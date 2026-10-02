/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Fuchs (Technical University of Munich), paul.fuchs@tum.de
------------------------------------------------------------------------- */

#include "pair_chemtrain_deploy_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "comm_kokkos.h"
#include "chemtrain_runtime.h"
#include "error.h"
#include "fix_chemtrain_topology.h"
#include "fix_chemtrain_topology_kokkos.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <string>
#include <utility>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

using namespace LAMMPS_NS;

namespace {

struct DlpackKokkosTensorContext {
  std::vector<int64_t> shape;
};

void delete_dlpack_kokkos_tensor(DLManagedTensor *tensor)
{
  delete static_cast<DlpackKokkosTensorContext *>(tensor->manager_ctx);
  delete tensor;
}

DLDataType dl_float32_kokkos()
{
  return DLDataType{2, 32, 1};
}

DLDataType dl_int32_kokkos()
{
  return DLDataType{0, 32, 1};
}

std::string lower_backend(std::string backend)
{
  std::transform(backend.begin(), backend.end(), backend.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return backend;
}

bool dl_device_type_for_backend(const std::string &backend, DLDeviceType &device_type)
{
  const std::string lower = lower_backend(backend);
  if (lower == "cuda" || lower.find("cuda") != std::string::npos) {
    device_type = kDLCUDA;
    return true;
  }
  if (lower == "rocm" || lower == "hip" || lower.find("rocm") != std::string::npos) {
    device_type = kDLROCM;
    return true;
  }
  if (lower == "sycl" || lower == "oneapi" || lower.find("sycl") != std::string::npos) {
    device_type = kDLOneAPI;
    return true;
  }
  return false;
}

template <class DeviceType>
void *kokkos_native_stream()
{
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<DeviceType, Kokkos::Cuda>)
    return reinterpret_cast<void *>(Kokkos::Cuda().cuda_stream());
#endif
  return nullptr;
}

#if defined(KOKKOS_ENABLE_CUDA)
bool cuda_stream_wait_for_stream(void *waiting_stream, void *producer_stream,
                                 const char *label, std::string &error)
{
  if (waiting_stream == nullptr || producer_stream == nullptr) {
    error = std::string("chemtrain/kk cannot bridge CUDA streams for ") +
        label + ": missing stream handle";
    return false;
  }

  cudaEvent_t event = nullptr;
  cudaError_t rc = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  if (rc != cudaSuccess) {
    error = std::string("chemtrain/kk failed to create CUDA event for ") +
        label + ": " + cudaGetErrorString(rc);
    return false;
  }

  rc = cudaEventRecord(event, static_cast<cudaStream_t>(producer_stream));
  if (rc != cudaSuccess) {
    cudaEventDestroy(event);
    error = std::string("chemtrain/kk failed to record CUDA event for ") +
        label + ": " + cudaGetErrorString(rc);
    return false;
  }

  rc = cudaStreamWaitEvent(static_cast<cudaStream_t>(waiting_stream), event, 0);
  if (rc != cudaSuccess) {
    cudaEventDestroy(event);
    error = std::string("chemtrain/kk failed to order CUDA streams for ") +
        label + ": " + cudaGetErrorString(rc);
    return false;
  }

  rc = cudaEventDestroy(event);
  if (rc != cudaSuccess) {
    error = std::string("chemtrain/kk failed to destroy CUDA event for ") +
        label + ": " + cudaGetErrorString(rc);
    return false;
  }
  return true;
}
#endif

}    // namespace

// ----------------------------------------------------------------------
// LAMMPS pair-style lifecycle
// ----------------------------------------------------------------------

template <class DeviceType>
ChemtrainDeployKokkos<DeviceType>::ChemtrainDeployKokkos(LAMMPS *lmp) :
    ChemtrainDeploy(lmp), atomKK(nullptr),
    communication_data(nullptr), communication_width(0)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | TAG_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  reverse_comm_device = 1;
}

template <class DeviceType>
ChemtrainDeployKokkos<DeviceType>::~ChemtrainDeployKokkos()
{
  if (copymode) return;
  clear_model_contexts();
  memoryKK->destroy_kokkos(k_eatom, eatom);
}

template <class DeviceType>
std::string ChemtrainDeployKokkos<DeviceType>::topology_fix_style() const
{
  if constexpr (std::is_same_v<DeviceType, LMPDeviceType>)
    return "CHEMTRAIN/TOPOLOGY/kk/device";
  return "CHEMTRAIN/TOPOLOGY/kk/host";
}

template <class DeviceType>
typename ChemtrainDeployKokkos<DeviceType>::KokkosModelContextStorage &
ChemtrainDeployKokkos<DeviceType>::storage_for(const ChemtrainModelContext &model_context)
{
  if (model_context_storages.size() != model_contexts.size())
    model_context_storages.resize(model_contexts.size());
  for (size_t i = 0; i < model_contexts.size(); ++i)
    if (model_contexts[i].get() == &model_context) return model_context_storages[i];
  error->all(FLERR, "chemtrain/kk internal error: unknown model context");
  return model_context_storages.front();
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::init_style()
{
  if (backend_name == "cpu")
    error->all(FLERR,
               "pair_style chemtrain/kk supports only the CUDA backend. "
               "Use pair_style chemtrain for CPU execution");

  if constexpr (!std::is_same_v<DeviceType, LMPDeviceType>) {
    ChemtrainDeploy::init_style();
    auto request = neighbor->find_request(this);
    request->set_kokkos_host(true);
    request->set_kokkos_device(false);
    return;
  } else {
    if (install_topology_zero_styles) install_missing_topology_styles();

    double max_comm_dist = 0.0;
    bool need_full = false;
    bool include_pair_type = false;
    int max_comm_width = 0;
    for (const auto &model_context : model_contexts) {
      const auto &properties = model_context->runtime->model_properties;
      if (properties.neighbor_format != JCN_NEIGHBOR_SIMPLE_SPARSE)
        error->all(FLERR,
                   "chemtrain/kk currently supports only SIMPLE_SPARSE "
                   "neighbor lists on Kokkos devices");
      max_comm_dist = std::max(max_comm_dist, properties.comm_dist);
      need_full = need_full || !properties.half_list;
      include_pair_type = include_pair_type || properties.include_pair_type;
      max_comm_width = std::max(max_comm_width, properties.communication_buffer_width);
    }
    if (use_communication && !force->newton_pair)
      error->all(FLERR,
                 "chemtrain/kk communication models require newton pair on");

    comm->cutghostuser = std::max(comm->cutghostuser, max_comm_dist + neighbor->skin);
    // Owned and ghost central rows make graph coverage independent of the
    // endpoint where a model collects edge values.
    int flags = NeighConst::REQ_DEFAULT | NeighConst::REQ_GHOST;
    if (force->newton_pair || need_full) {
      flags |= NeighConst::REQ_FULL;
    } else {
      flags |= NeighConst::REQ_NEWTON_OFF;
    }
    neighbor->add_request(this, flags);
    auto request = neighbor->find_request(this);
    if (include_pair_type) {
      request->enable_special();
      ensure_topology_lookup();
    } else release_topology_lookup();
    request->set_kokkos_host(false);
    request->set_kokkos_device(true);
    comm_forward = max_comm_width;
    comm_reverse = max_comm_width;

    if (use_communication) {
      auto *commKK = dynamic_cast<CommKokkos *>(comm);
      if (commKK == nullptr)
        error->all(FLERR, "chemtrain/kk device communication requires CommKokkos");
      if (comm->nprocs > 1 && !lmp->kokkos->gpu_aware_flag)
        error->all(FLERR,
                   "chemtrain/kk multi-rank device communication requires "
                   "GPU-aware MPI. Use package kokkos gpu/aware on or use the "
                   "non-Kokkos chemtrain pair style");
      if (force->newton_pair &&
          (lmp->kokkos->reverse_comm_legacy ||
           lmp->kokkos->reverse_comm_on_host))
        error->all(FLERR,
                   "chemtrain/kk communication models require device "
                   "Kokkos atom-force reverse communication. Use package kokkos "
                   "comm device gpu/aware on");
      if (lmp->kokkos->forward_pair_comm_legacy ||
          lmp->kokkos->reverse_pair_comm_legacy)
        error->all(FLERR,
                   "chemtrain/kk device communication requires Kokkos "
                   "device pair communication. Use package kokkos comm device "
                   "gpu/aware on");
    }
  }
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::compute(int eflag, int vflag)
{
  // LAMMPS Kokkos arrays are dual views. sync() makes the atom data current
  // before DLPack exposes those pointers. Mark writable arrays first, matching
  // native Kokkos pair styles, so device-side virial reduction sees device data.
  atomKK->sync(execution_space, datamask_read);
  if (eflag || vflag)
    atomKK->modified(execution_space, datamask_modify);
  else
    atomKK->modified(execution_space, F_MASK);
  ChemtrainDeploy::compute(eflag, vflag);
}

// ----------------------------------------------------------------------
// Model output application
// ----------------------------------------------------------------------

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::apply_forces(
    ChemtrainModelContext &model_context)
{
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    ChemtrainDeploy::apply_forces(model_context);
  } else {
    auto f = atomKK->k_f.template view<DeviceType>();
    const auto forces = storage_for(model_context).device_forces_f32;
    const double force_scale = model_context.scale;
    const int active_rows = atom->nlocal + (force->newton_pair ? atom->nghost : 0);
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, active_rows),
                         KOKKOS_LAMBDA(int i) {
                           f(i, 0) += force_scale * forces(3 * i + 0);
                           f(i, 1) += force_scale * forces(3 * i + 1);
                           f(i, 2) += force_scale * forces(3 * i + 2);
                         });
    DeviceType().fence();
    atomKK->modified(execution_space, F_MASK);
  }
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::sync_requested_outputs_to_host(
    ChemtrainModelContext &model_context)
{
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    ChemtrainDeploy::sync_requested_outputs_to_host(model_context);
  } else {
    for (auto &output : model_context.requested_outputs) {
      output.device_timestep = update->ntimestep;
      // A force evaluation can run more than once at the same LAMMPS timestep.
      // Invalidate the host mirror by evaluation, not only by timestep.
      output.host_timestep = -1;
    }
  }
}

template <class DeviceType>
const double *ChemtrainDeployKokkos<DeviceType>::chemtrain_output(
    const std::string &name, int &components, bigint &timestep)
{
  return chemtrain_output("", name, components, timestep);
}

template <class DeviceType>
const double *ChemtrainDeployKokkos<DeviceType>::chemtrain_output(
    const std::string &model_context_name, const std::string &name,
    int &components, bigint &timestep)
{
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    return ChemtrainDeploy::chemtrain_output(model_context_name, name, components,
                                             timestep);
  } else {
    auto &model_context = model_context_for_name(model_context_name);
    auto &storage = storage_for(model_context);
    for (size_t i = 0; i < model_context.requested_outputs.size(); ++i) {
      auto &output = model_context.requested_outputs[i];
      if (output.name != name) continue;
      components = output.components;
      if (output.device_timestep == update->ntimestep &&
          output.host_timestep != update->ntimestep) {
        if (i >= storage.host_requested_outputs.size())
          error->all(FLERR, "chemtrain/kk output {} has no host mirror",
                     name);
        // Host-side LAMMPS consumers copy a device output on their first access
        // after each successful force evaluation. The mirror persists until
        // capacity changes.
        Kokkos::deep_copy(storage.host_requested_outputs[i],
                          storage.device_requested_outputs[i]);
        const size_t records =
            output.scope == JCN_OUTPUT_PARTICLE ? atom->nlocal : 1;
        const size_t value_count = records * output.components;
        std::copy(storage.host_requested_outputs[i].data(),
                  storage.host_requested_outputs[i].data() + value_count,
                  output.host_values.begin());
        output.host_timestep = update->ntimestep;
      }
      timestep = output.host_timestep;
      return output.host_values.data();
    }
    components = 0;
    timestep = -1;
    return nullptr;
  }
}

template <class DeviceType>
typename ChemtrainDeployKokkos<DeviceType>::RequestedOutputDeviceView
ChemtrainDeployKokkos<DeviceType>::chemtrain_output_device(
    const std::string &model_context_name, const std::string &name,
    int &components, bigint &timestep)
{
  // The returned view aliases capacity-owned storage. The next force evaluation
  // overwrites its values, and a capacity change replaces the allocation. The
  // returned timestep lets callers reject stale output.
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    error->all(FLERR,
               "The chemtrain Kokkos host execution path does not provide "
               "device output views");
    components = 0;
    timestep = -1;
    return {};
  } else {
    auto &model_context = model_context_for_name(model_context_name);
    auto &storage = storage_for(model_context);
    for (size_t i = 0; i < model_context.requested_outputs.size(); ++i) {
      auto &output = model_context.requested_outputs[i];
      if (output.name != name) continue;
      components = output.components;
      timestep = output.device_timestep;
      if (i >= storage.device_requested_outputs.size())
        error->all(FLERR, "chemtrain/kk output {} has no device buffer",
                   name);
      return storage.device_requested_outputs[i];
    }
    components = 0;
    timestep = -1;
    return {};
  }
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::initialize_energy_virial(int eflag, int vflag)
{
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    ChemtrainDeploy::initialize_energy_virial(eflag, vflag);
  } else {
    ev_init(eflag, vflag, 0);
    if (eflag_atom) prepare_peratom_energy_storage();
  }
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::prepare_peratom_energy_storage()
{
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    ChemtrainDeploy::prepare_peratom_energy_storage();
  } else {
    if ((int) k_eatom.view_host().extent(0) < maxeatom) {
      memoryKK->destroy_kokkos(k_eatom, eatom);
      memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    }
    Kokkos::deep_copy(k_eatom.template view<DeviceType>(), 0.0);
    k_eatom.template modify<DeviceType>();
    d_eatom = k_eatom.template view<DeviceType>();
  }
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::apply_energy(const ChemtrainModelContext &model_context)
{
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    ChemtrainDeploy::apply_energy(model_context);
  } else {
    auto d_e = d_eatom;
    const auto model_energy = storage_for(model_context).device_energy_f32;
    const double energy_scale = model_context.scale;
    const int nlocal = atom->nlocal;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal),
                         KOKKOS_LAMBDA(int i) {
                           d_e(i) += energy_scale *
                               static_cast<double>(model_energy(i));
                         });
    DeviceType().fence();
    k_eatom.template modify<DeviceType>();
  }
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::apply_virial(
    const ChemtrainModelContext &model_context)
{
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    ChemtrainDeploy::apply_virial(model_context);
  } else {
    // The exported V output is already -dE/dstrain in LAMMPS order and energy
    // units. Copy its six components from the device and apply only the model
    // scale.
    auto &storage = storage_for(model_context);
    Kokkos::deep_copy(storage.host_virial_f32, storage.device_virial_f32);
    for (int i = 0; i < 6; ++i)
      virial[i] += model_context.scale *
          static_cast<double>(storage.host_virial_f32(i));
  }
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::finish_peratom_energy_storage()
{
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    ChemtrainDeploy::finish_peratom_energy_storage();
  } else {
    k_eatom.sync_host();
  }
}

template <class DeviceType>
bool ChemtrainDeployKokkos<DeviceType>::use_device_communication_callback() const
{
  return std::is_same_v<DeviceType, LMPDeviceType>;
}

template <class DeviceType>
bool ChemtrainDeployKokkos<DeviceType>::use_host_communication_callback() const
{
  return false;
}

template <class DeviceType>
jcn_tensor_dtype ChemtrainDeployKokkos<DeviceType>::engine_position_dtype() const
{
  if constexpr (std::is_same_v<DeviceType, LMPDeviceType>)
    return JCN_TENSOR_DTYPE_F32;
  return JCN_TENSOR_DTYPE_F64;
}

template <class DeviceType>
int ChemtrainDeployKokkos<DeviceType>::selected_backend_device_ordinal(
    const std::string &backend) const
{
  if constexpr (!std::is_same_v<DeviceType, LMPDeviceType>) {
    return ChemtrainDeploy::selected_backend_device_ordinal(backend);
  } else {
    if (lower_backend(backend).find("cuda") == std::string::npos)
      return ChemtrainDeploy::selected_backend_device_ordinal(backend);
#if defined(KOKKOS_ENABLE_CUDA)
    // PJRT imports must use Kokkos's active CUDA device rather than a device
    // ordinal inferred from the MPI rank.
    int device = -1;
    const cudaError_t rc = cudaGetDevice(&device);
    if (rc != cudaSuccess)
      error->all(FLERR, "chemtrain/kk failed to query active CUDA device: {}",
                 cudaGetErrorString(rc));
    return device;
#else
    error->all(FLERR,
               "chemtrain/kk CUDA backend requires a CUDA-enabled Kokkos build");
    return 0;
#endif
  }
}

template <class DeviceType>
double ChemtrainDeployKokkos<DeviceType>::reduce_energy(
    const ChemtrainModelContext &model_context) const
{
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    return ChemtrainDeploy::reduce_energy(model_context);
  } else {
    const auto energy =
        const_cast<ChemtrainDeployKokkos<DeviceType> *>(this)
            ->storage_for(model_context).device_energy_f32;
    const int nlocal = atom->nlocal;
    double potential = 0.0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0, nlocal),
                            KOKKOS_LAMBDA(int i, double &sum) {
                              sum += static_cast<double>(energy(i));
                            },
                            potential);
    DeviceType().fence();
    return potential;
  }
}

// ----------------------------------------------------------------------
// DLPack request construction
// ----------------------------------------------------------------------

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::observe_neighbors(
    ChemtrainModelContext &model_context)
{
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    ChemtrainDeploy::observe_neighbors(model_context);
  } else {
    auto &runtime = *model_context.runtime;
    // Connector graphs include LAMMPS rows for both owned and ghost central
    // atoms. Graph coverage does not depend on communication callbacks.
    const int row_count = list == nullptr ? 0 : list->inum + list->gnum;
    if (list != nullptr && !list->ghost)
      error->all(FLERR,
                 "chemtrain/kk graph construction requires ghost-I neighbor rows, "
                 "but LAMMPS did not build a ghost neighbor list");

    if (model_context.model_neighbors == nullptr ||
        model_context.model_neighbors->format() != runtime.model_properties.neighbor_format)
      model_context.model_neighbors =
          std::make_unique<ChemtrainNeighborSparseKokkos<DeviceType>>(execution_context->api);
    model_context.model_neighbors->observe({list, row_count, neighbor->ago == 0});
  }
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::build_request(
    ChemtrainModelContext &model_context, bool allow_internal_recompile,
    bool force_clear_capacities,
    jcn_force_request &request)
{
  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    ChemtrainDeploy::build_request(model_context, allow_internal_recompile,
                                   force_clear_capacities, request);
  } else {
    auto &runtime = *model_context.runtime;
    auto &storage = storage_for(model_context);
    const int nall = atom->nlocal + atom->nghost;
    const int nlocal = atom->nlocal;
    // Request construction uses the same owned-and-ghost row set as the earlier
    // count-only observation pass.
    DLDeviceType dl_device_type;
    if (!dl_device_type_for_backend(backend_name, dl_device_type))
      error->all(FLERR,
                 "chemtrain/kk cannot map backend to a DLPack device type");
    void *producer_stream = kokkos_native_stream<DeviceType>();
    if (lower_backend(backend_name).find("cuda") != std::string::npos &&
        producer_stream == nullptr)
      error->all(FLERR,
                 "chemtrain/kk CUDA DLPack import requires a Kokkos CUDA stream");
    if (lower_backend(backend_name).find("cuda") == std::string::npos)
      error->all(FLERR,
                 "chemtrain/kk device DLPack copy-back currently supports CUDA only");

    auto import_device_buffer = [&](void *data, DLDataType dtype,
                                    std::vector<int64_t> shape) {
      auto *context = new DlpackKokkosTensorContext{std::move(shape)};
      auto *tensor = new DLManagedTensor{};
      tensor->dl_tensor.data = data;
      tensor->dl_tensor.device = DLDevice{dl_device_type, backend_device_ordinal};
      tensor->dl_tensor.ndim = static_cast<int32_t>(context->shape.size());
      tensor->dl_tensor.dtype = dtype;
      tensor->dl_tensor.shape = context->shape.data();
      tensor->dl_tensor.strides = nullptr;
      tensor->dl_tensor.byte_offset = 0;
      tensor->manager_ctx = context;
      tensor->deleter = delete_dlpack_kokkos_tensor;

      jcn_status status{};
      jcn_buffer_import_options options{};
      options.copy_policy = JCN_DLPACK_NEVER_COPY;
      options.role = JCN_BUFFER_ROLE_INPUT_OUTPUT;
      options.debug_name = "chemtrain_kokkos_device_buffer";
      options.backend = backend_name.c_str();
      options.device_ordinal = backend_device_ordinal;
      options.producer_stream = producer_stream;
      jcn_buffer *buffer =
          execution_context->api->buffer_from_dlpack_options(runtime.executor, tensor,
                                                            &options, &status);
      if (status.code != JCN_STATUS_OK) {
        delete_dlpack_kokkos_tensor(tensor);
        fail_if_jcn_error("chemtrain/kk failed to import device DLPack buffer",
                          status);
      }
      return buffer;
    };

    auto x = atomKK->k_x.template view<DeviceType>();
    auto type = atomKK->k_type.template view<DeviceType>();
    const bool atom_capacity_changed =
        runtime.atom_capacity != runtime.capacities.max_atoms;
    // Recreate all capacity-dependent Kokkos views and DLPack handles together.
    // Keeping their lifetimes aligned prevents cached handles from referring to
    // replaced device allocations.
    if (atom_capacity_changed) {
      destroy_buffer(runtime.positions_buffer);
      for (auto &input : model_context.particle_inputs) destroy_buffer(input.cached_buffer);
      for (auto &input : model_context.global_inputs) destroy_buffer(input.cached_buffer);
      for (auto &output : model_context.requested_outputs) destroy_buffer(output.cached_buffer);
      destroy_buffer(runtime.forces_buffer);
      destroy_buffer(runtime.energy_buffer);
      destroy_buffer(runtime.virial_buffer);
      storage.device_positions_f32 =
          Kokkos::View<float *, typename DeviceType::array_layout, DeviceType>(
              "chemtrain:positions_f32", runtime.capacities.max_atoms * 3);
      storage.device_forces_f32 =
          Kokkos::View<float *, typename DeviceType::array_layout, DeviceType>(
              "chemtrain:forces_f32", runtime.capacities.max_atoms * 3);
      storage.device_energy_f32 =
          Kokkos::View<float *, typename DeviceType::array_layout, DeviceType>(
              "chemtrain:energy_f32", runtime.capacities.max_atoms);
      storage.device_virial_f32 =
          Kokkos::View<float *, typename DeviceType::array_layout, DeviceType>(
              "chemtrain:virial_f32", 6);
      storage.host_virial_f32 =
          Kokkos::create_mirror_view(storage.device_virial_f32);
      storage.device_particle_inputs.clear();
      storage.host_particle_inputs.clear();
      storage.device_particle_inputs.reserve(model_context.particle_inputs.size());
      storage.host_particle_inputs.reserve(model_context.particle_inputs.size());
      for (const auto &input : model_context.particle_inputs) {
        storage.device_particle_inputs.emplace_back(
            "chemtrain:particle_" + input.name,
            runtime.capacities.max_atoms);
        storage.host_particle_inputs.push_back(
            Kokkos::create_mirror_view(storage.device_particle_inputs.back()));
      }
      storage.device_global_inputs.resize(model_context.global_inputs.size());
      storage.device_requested_outputs.clear();
      storage.host_requested_outputs.clear();
      storage.device_requested_outputs.reserve(model_context.requested_outputs.size());
      storage.host_requested_outputs.reserve(model_context.requested_outputs.size());
      for (auto &output : model_context.requested_outputs) {
        const size_t records = output.scope == JCN_OUTPUT_PARTICLE
            ? static_cast<size_t>(runtime.capacities.max_atoms) : 1;
        const size_t value_count = records * output.components;
        storage.device_requested_outputs.emplace_back(
            "chemtrain:requested_output", value_count);
        storage.host_requested_outputs.push_back(
            Kokkos::create_mirror_view(storage.device_requested_outputs.back()));
        output.host_values.resize(value_count);
      }
      runtime.atom_capacity = runtime.capacities.max_atoms;
    }

    // Copy current positions into the capacity-sized model buffer and clear the
    // force and energy buffers that the executable will fill.
    auto positions_f32 = storage.device_positions_f32;
    auto forces_f32 = storage.device_forces_f32;
    auto energy_f32 = storage.device_energy_f32;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nall),
                         KOKKOS_LAMBDA(int i) {
                           positions_f32(3 * i + 0) = static_cast<float>(x(i, 0));
                           positions_f32(3 * i + 1) = static_cast<float>(x(i, 1));
                           positions_f32(3 * i + 2) = static_cast<float>(x(i, 2));
                           forces_f32(3 * i + 0) = 0.0f;
                           forces_f32(3 * i + 1) = 0.0f;
                           forces_f32(3 * i + 2) = 0.0f;
                           energy_f32(i) = 0.0f;
                         });
    for (size_t field = 0; field < model_context.particle_inputs.size(); ++field) {
      auto &input = model_context.particle_inputs[field];
      auto values = storage.device_particle_inputs[field];
      if (input.source_kind == ChemtrainParticleSource::CUSTOM_INT) {
        validate_custom_particle_input(input);
        auto host_values = storage.host_particle_inputs[field];
        for (int i = 0; i < nall; ++i)
          host_values(i) = atom->ivector[input.custom_index][i];
        Kokkos::deep_copy(
            Kokkos::subview(values, std::make_pair(0, nall)),
            Kokkos::subview(host_values, std::make_pair(0, nall)));
      } else {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<DeviceType>(0, nall),
            KOKKOS_LAMBDA(int i) { values(i) = type(i) - 1; });
      }
    }
    DeviceType().fence();

    // Bind the position and particle input buffers after their device storage
    // contains the current atom data.
    request.atoms.num_owned_atoms = nlocal;
    request.atoms.num_valid_atoms = nall;
    if (runtime.positions_buffer == nullptr)
      runtime.positions_buffer =
          import_device_buffer(storage.device_positions_f32.data(), dl_float32_kokkos(),
                               {runtime.capacities.max_atoms, 3});
    for (size_t i = 0; i < model_context.particle_inputs.size(); ++i) {
      auto &input = model_context.particle_inputs[i];
      if (input.cached_buffer == nullptr)
        input.cached_buffer = import_device_buffer(
            storage.device_particle_inputs[i].data(), dl_int32_kokkos(),
            {runtime.capacities.max_atoms});
      runtime.particle_inputs[i] =
          jcn_named_input{input.name.c_str(), input.cached_buffer};
    }

    // Evaluate configuration inputs on the host, copy each scalar to persistent
    // device storage, and bind the resulting DLPack handles.
    evaluate_global_inputs(model_context);
    for (size_t i = 0; i < model_context.global_inputs.size(); ++i) {
      auto &input = model_context.global_inputs[i];
      auto &device_input = storage.device_global_inputs[i];
      void *data = nullptr;
      DLDataType dtype{};
      if (input.dtype == JCN_TENSOR_DTYPE_F32) {
        if (device_input.f32.extent(0) == 0)
          device_input.f32 = Kokkos::View<float *, typename DeviceType::array_layout, DeviceType>(
              "chemtrain:global_f32", 1);
        Kokkos::deep_copy(device_input.f32, input.float_value);
        data = device_input.f32.data();
        dtype = dl_float32_kokkos();
      } else if (input.dtype == JCN_TENSOR_DTYPE_F64) {
        if (device_input.f64.extent(0) == 0)
          device_input.f64 = Kokkos::View<double *, typename DeviceType::array_layout, DeviceType>(
              "chemtrain:global_f64", 1);
        Kokkos::deep_copy(device_input.f64, input.double_value);
        data = device_input.f64.data();
        dtype = DLDataType{2, 64, 1};
      } else if (input.dtype == JCN_TENSOR_DTYPE_S32) {
        if (device_input.s32.extent(0) == 0)
          device_input.s32 = Kokkos::View<int32_t *, typename DeviceType::array_layout, DeviceType>(
              "chemtrain:global_s32", 1);
        Kokkos::deep_copy(device_input.s32, input.int_value);
        data = device_input.s32.data();
        dtype = dl_int32_kokkos();
      }
      if (input.cached_buffer == nullptr)
        input.cached_buffer = import_device_buffer(data, dtype, {});
      runtime.global_inputs[i] =
          jcn_named_input{input.name.c_str(), input.cached_buffer};
    }

    // Bind the built-in force, energy, and virial output buffers.
    if (runtime.forces_buffer == nullptr)
      runtime.forces_buffer =
          import_device_buffer(storage.device_forces_f32.data(), dl_float32_kokkos(),
                               {runtime.capacities.max_atoms, 3});
    if (runtime.energy_buffer == nullptr)
      runtime.energy_buffer =
          import_device_buffer(storage.device_energy_f32.data(), dl_float32_kokkos(),
                               {runtime.capacities.max_atoms});
    if (runtime.virial_buffer == nullptr)
      runtime.virial_buffer = import_device_buffer(
          storage.device_virial_f32.data(), dl_float32_kokkos(), {6});

    // Bind auxiliary outputs. PARTICLE outputs include the atom-capacity axis,
    // while LOCAL and GLOBAL outputs contain one configuration value.
    for (size_t i = 0; i < model_context.requested_outputs.size(); ++i) {
      auto &output = model_context.requested_outputs[i];
      if (output.cached_buffer == nullptr) {
        auto shape = output.dimensions;
        if (output.scope == JCN_OUTPUT_PARTICLE)
          shape.insert(shape.begin(), runtime.capacities.max_atoms);
        output.cached_buffer =
            import_device_buffer(storage.device_requested_outputs[i].data(),
                                 dl_float32_kokkos(), shape);
      }
    }

    // Assemble request metadata only after every referenced buffer has stable
    // storage.
    request.atoms.positions = runtime.positions_buffer;
    request.atoms.particle_inputs = runtime.particle_inputs.data();
    request.atoms.num_particle_inputs =
        static_cast<int32_t>(runtime.particle_inputs.size());
    request.atoms.global_inputs = runtime.global_inputs.data();
    request.atoms.num_global_inputs =
        static_cast<int32_t>(runtime.global_inputs.size());
    request.capacities = runtime.capacities;
    request.allow_internal_recompile = allow_internal_recompile ? 1 : 0;
    const int output_force_rows = nlocal + (force->newton_pair ? atom->nghost : 0);
    append_named_outputs(model_context, output_force_rows);
    request.outputs = runtime.outputs.data();
    request.num_outputs = static_cast<int32_t>(runtime.outputs.size());

    // Build the neighbor request from the same owned-and-ghost row set used
    // during neighbor observation.
    if (runtime.model_properties.include_pair_type && topology_lookup == nullptr)
      error->all(FLERR, "chemtrain/kk topology lookup is unavailable");
    ChemtrainNeighborBuild neighbor_build{
        runtime.capacities,
        runtime.model_properties.include_pair_type != 0,
        {},
        import_device_buffer,
        [this](const char *context, const jcn_status &status) {
          fail_if_jcn_error(context, status);
        },
        error,
        nlocal,
        topology_lookup,
        atomKK,
        domain};
    const auto neighbor_update = model_context.model_neighbors->update(neighbor_build);
    request.neighbors = neighbor_update.neighbors;
    // Capacity and neighbor changes invalidate separate connector caches.
    request.clear_capacities =
        (force_clear_capacities || atom_capacity_changed ||
         neighbor_update.capacity_changed) ? 1 : 0;
    request.clear_neighbors = neighbor_update.rebuilt ? 1 : 0;
  }
}

// ----------------------------------------------------------------------
// PJRT communication callbacks
// ----------------------------------------------------------------------

template <class DeviceType>
int ChemtrainDeployKokkos<DeviceType>::exchange_device_features(
    void *data, int64_t rows, int64_t cols, jcn_communication_scalar_type type,
    int reverse, const char *backend, int device_ordinal, void *pjrt_stream,
    const char **callback_error)
{
  if constexpr (!std::is_same_v<DeviceType, LMPDeviceType>) {
    communication_error =
        "chemtrain/kk host style received a device communication callback";
    if (callback_error != nullptr) *callback_error = communication_error.c_str();
    return 1;
  } else {
    if (data == nullptr || cols <= 0) {
      communication_error =
          "chemtrain/kk received an empty device communication buffer";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    if (rows < atom->nlocal + atom->nghost) {
      communication_error =
          "chemtrain/kk device communication buffer has fewer rows than LAMMPS atoms";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    if (type != JCN_COMMUNICATION_F64) {
      communication_error =
          "chemtrain/kk device communication expects f64 buffers";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    if (cols > INT_MAX) {
      communication_error =
          "chemtrain/kk device communication width exceeds integer range";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    if (backend == nullptr ||
        lower_backend(backend).find("cuda") == std::string::npos) {
      communication_error =
          "chemtrain/kk device communication currently supports CUDA only";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    if (pjrt_stream == nullptr) {
      communication_error =
          "chemtrain/kk device communication callback did not receive a PJRT stream";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    if (device_ordinal != backend_device_ordinal) {
      communication_error =
          "chemtrain/kk device communication callback used a different "
          "PJRT device than the force request";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
#if defined(KOKKOS_ENABLE_CUDA)
    int active_cuda_device = -1;
    const cudaError_t active_device_status = cudaGetDevice(&active_cuda_device);
    if (active_device_status != cudaSuccess) {
      communication_error =
          std::string("chemtrain/kk failed to query active CUDA device "
                      "during feature exchange: ") +
          cudaGetErrorString(active_device_status);
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    if (active_cuda_device != device_ordinal) {
      communication_error =
          "chemtrain/kk active Kokkos CUDA device differs from the PJRT "
          "feature communication device";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    void *kokkos_stream = kokkos_native_stream<DeviceType>();
    if (kokkos_stream == nullptr) {
      communication_error =
          "chemtrain/kk device communication could not get the Kokkos CUDA stream";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
#else
    communication_error =
        "chemtrain/kk device communication requires a CUDA-enabled Kokkos build";
    if (callback_error != nullptr) *callback_error = communication_error.c_str();
    return 1;
#endif

    // Device communication occurs inside an XLA FFI custom call. The temporary
    // communication buffer is not imported through DLPack like the persistent
    // atom buffers. CUDA events therefore reproduce the same stream-ordering
    // contract: Kokkos waits for the PJRT copy, and PJRT waits for Kokkos
    // communication.
#if defined(KOKKOS_ENABLE_CUDA)
    if (!cuda_stream_wait_for_stream(kokkos_stream, pjrt_stream,
                                     "PJRT-to-Kokkos feature exchange",
                                     communication_error)) {
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
#endif

    const int width = static_cast<int>(cols);
    auto *buffer = static_cast<double *>(data);
    const int rc = reverse ? reverse_comm(buffer, buffer, width)
                           : forward_comm(buffer, buffer, width);

#if defined(KOKKOS_ENABLE_CUDA)
    if (rc == 0 &&
        !cuda_stream_wait_for_stream(pjrt_stream, kokkos_stream,
                                     "Kokkos-to-PJRT feature exchange",
                                     communication_error)) {
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
#endif
    return rc;
  }
}

template <class DeviceType>
int ChemtrainDeployKokkos<DeviceType>::reduce_device_features(
    void *data, int64_t count, jcn_communication_scalar_type type,
    const char *backend, int device_ordinal, void *pjrt_stream,
    const char **callback_error)
{
  if constexpr (!std::is_same_v<DeviceType, LMPDeviceType>) {
    communication_error =
        "chemtrain/kk host style received a device reduce callback";
    if (callback_error != nullptr) *callback_error = communication_error.c_str();
    return 1;
  } else {
    if (data == nullptr || count <= 0 || count > INT_MAX) {
      communication_error =
          "chemtrain/kk received an invalid device reduce buffer";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    if (backend == nullptr ||
        lower_backend(backend).find("cuda") == std::string::npos) {
      communication_error =
          "chemtrain/kk device reduce currently supports CUDA only";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    if (pjrt_stream == nullptr) {
      communication_error =
          "chemtrain/kk device reduce callback did not receive a PJRT stream";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    if (device_ordinal != backend_device_ordinal) {
      communication_error =
          "chemtrain/kk device reduce callback used a different PJRT "
          "device than the force request";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
#if defined(KOKKOS_ENABLE_CUDA)
    int active_cuda_device = -1;
    const cudaError_t active_device_status = cudaGetDevice(&active_cuda_device);
    if (active_device_status != cudaSuccess) {
      communication_error =
          std::string("chemtrain/kk failed to query active CUDA device "
                      "during reduce: ") +
          cudaGetErrorString(active_device_status);
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    if (active_cuda_device != device_ordinal) {
      communication_error =
          "chemtrain/kk active Kokkos CUDA device differs from the PJRT "
          "reduce device";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    void *kokkos_stream = kokkos_native_stream<DeviceType>();
    if (kokkos_stream == nullptr) {
      communication_error =
          "chemtrain/kk device reduce could not get the Kokkos CUDA stream";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
#else
    communication_error =
        "chemtrain/kk device reduce requires a CUDA-enabled Kokkos build";
    if (callback_error != nullptr) *callback_error = communication_error.c_str();
    return 1;
#endif

#if defined(KOKKOS_ENABLE_CUDA)
    // CUDA-aware MPI consumes the Kokkos device pointer directly, but MPI is
    // not ordered on the PJRT stream. Make PJRT writes visible before Allreduce.
    if (!cuda_stream_wait_for_stream(kokkos_stream, pjrt_stream,
                                     "PJRT-to-Kokkos reduce",
                                     communication_error)) {
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    const cudaError_t ready_status =
        cudaStreamSynchronize(static_cast<cudaStream_t>(kokkos_stream));
    if (ready_status != cudaSuccess) {
      communication_error =
          std::string("chemtrain/kk failed to synchronize CUDA stream "
                      "before device reduce: ") +
          cudaGetErrorString(ready_status);
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
#endif

    if (type != JCN_COMMUNICATION_F64) {
      communication_error = "chemtrain/kk reduce expects f64 buffers";
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
    int rc = reduce_device_buffer(static_cast<double *>(data), count);

#if defined(KOKKOS_ENABLE_CUDA)
    // Make the reduced device buffer visible to the PJRT stream before XLA
    // consumes the reduce result.
    if (rc == 0 &&
        !cuda_stream_wait_for_stream(pjrt_stream, kokkos_stream,
                                     "Kokkos-to-PJRT reduce",
                                     communication_error)) {
      if (callback_error != nullptr) *callback_error = communication_error.c_str();
      return 1;
    }
#endif
    if (rc != 0 && callback_error != nullptr)
      *callback_error = communication_error.c_str();
    return rc;
  }
}

template <class DeviceType>
int ChemtrainDeployKokkos<DeviceType>::reduce_device_buffer(double *data,
                                                            int64_t count)
{
  if (comm->nprocs == 1) return 0;

  // CUDA-aware MPI reduces the connector scratch buffer in place on device.
  const int rc =
      MPI_Allreduce(MPI_IN_PLACE, data, static_cast<int>(count), MPI_DOUBLE,
                    MPI_SUM, world);
  if (rc != MPI_SUCCESS) {
    communication_error =
        "chemtrain/kk CUDA-aware MPI_Allreduce failed";
    return 1;
  }
  return 0;
}

// ----------------------------------------------------------------------
// LAMMPS Kokkos communication hooks
// ----------------------------------------------------------------------

// The exported executable writes intermediate per-atom values to a contiguous
// buffer during PJRT execution. The pair style stores the buffer address and
// width while CommKokkos fills ghost rows during forward communication or
// adds ghost-row contributions to owned rows during reverse communication. The
// callbacks are synchronous and non-reentrant, so these temporary fields remain
// valid until CommKokkos returns and are then cleared.
template <class DeviceType>
int ChemtrainDeployKokkos<DeviceType>::forward_comm(double *copy_from,
                                                    double *copy_to,
                                                    int width)
{
  if (width <= 0)
    error->all(FLERR, "chemtrain/kk invalid forward communication width");
  communication_data = copy_to;
  communication_width = width;
  comm_forward = width;

  if (copy_from != copy_to) {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, atom->nlocal * width),
                         KOKKOS_LAMBDA(int i) { copy_to[i] = copy_from[i]; });
  }

  auto *commKK = dynamic_cast<CommKokkos *>(comm);
  if (commKK == nullptr || commKK->forward_pair_comm_legacy)
    error->all(FLERR,
               "chemtrain/kk device forward communication requires "
               "Kokkos device pair communication");
  // CommKokkos chooses its pack and unpack path from execution_space. Use the
  // device hooks for the callback, then restore the pair's previous setting.
  const auto saved_execution_space = std::exchange(execution_space, Device);
  comm->forward_comm(this, width);
  execution_space = saved_execution_space;

  communication_data = nullptr;
  communication_width = 0;
  return 0;
}

template <class DeviceType>
int ChemtrainDeployKokkos<DeviceType>::reverse_comm(double *copy_from,
                                                    double *copy_to,
                                                    int width)
{
  if (width <= 0)
    error->all(FLERR, "chemtrain/kk invalid reverse communication width");
  communication_data = copy_to;
  communication_width = width;
  comm_reverse = width;

  if (copy_from != copy_to) {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(
                             0, (atom->nlocal + atom->nghost) * width),
                         KOKKOS_LAMBDA(int i) { copy_to[i] = copy_from[i]; });
  }

  auto *commKK = dynamic_cast<CommKokkos *>(comm);
  if (commKK == nullptr || commKK->reverse_pair_comm_legacy)
    error->all(FLERR,
               "chemtrain/kk device reverse communication requires "
               "Kokkos device pair communication");
  // CommKokkos chooses its pack and unpack path from execution_space. Use the
  // device hooks for the callback, then restore the pair's previous setting.
  const auto saved_execution_space = std::exchange(execution_space, Device);
  comm->reverse_comm(this, width);
  execution_space = saved_execution_space;

  communication_data = nullptr;
  communication_width = 0;
  return 0;
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::require_comm_ready() const
{
  if (communication_data == nullptr || communication_width <= 0)
    error->all(FLERR, "chemtrain/kk communication buffer was not initialized");
}

template <class DeviceType>
int ChemtrainDeployKokkos<DeviceType>::pack_forward_comm_kokkos(
    int n, DAT::tdual_int_1d list, DAT::tdual_double_1d &buf, int, int *)
{
  require_comm_ready();
  if (n > INT_MAX / communication_width)
    error->all(FLERR, "chemtrain/kk forward communication buffer is too large");
  const double *data = communication_data;
  auto idx = list.template view<DeviceType>();
  auto val = buf.template view<DeviceType>();
  const int width = communication_width;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, n * width),
                       KOKKOS_LAMBDA(int k) {
                         const int i = k / width;
                         const int j = k - i * width;
                         val(k) = data[idx(i) * width + j];
                       });
  return n * width;
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::unpack_forward_comm_kokkos(
    int n, int first, DAT::tdual_double_1d &buf)
{
  require_comm_ready();
  if (n > INT_MAX / communication_width)
    error->all(FLERR, "chemtrain/kk forward communication buffer is too large");
  double *data = communication_data;
  auto val = buf.template view<DeviceType>();
  const int width = communication_width;
  const int64_t offset = static_cast<int64_t>(first) * width;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, n * width),
                       KOKKOS_LAMBDA(int k) { data[offset + k] = val(k); });
}

template <class DeviceType>
int ChemtrainDeployKokkos<DeviceType>::pack_reverse_comm_kokkos(
    int n, int first, DAT::tdual_double_1d &buf)
{
  require_comm_ready();
  if (n > INT_MAX / communication_width)
    error->all(FLERR, "chemtrain/kk reverse communication buffer is too large");
  const double *data = communication_data;
  auto val = buf.template view<DeviceType>();
  const int width = communication_width;
  const int64_t offset = static_cast<int64_t>(first) * width;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, n * width),
                       KOKKOS_LAMBDA(int k) { val(k) = data[offset + k]; });
  return n * width;
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::unpack_reverse_comm_kokkos(
    int n, DAT::tdual_int_1d list, DAT::tdual_double_1d &buf)
{
  require_comm_ready();
  if (n > INT_MAX / communication_width)
    error->all(FLERR, "chemtrain/kk reverse communication buffer is too large");
  double *data = communication_data;
  auto idx = list.template view<DeviceType>();
  auto val = buf.template view<DeviceType>();
  const int width = communication_width;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, n * width),
                       KOKKOS_LAMBDA(int k) {
                         const int i = k / width;
                         const int j = k - i * width;
                         double *dst = data + idx(i) * width + j;
                         Kokkos::atomic_add(dst, val(k));
                       });
}

template <class DeviceType>
int ChemtrainDeployKokkos<DeviceType>::pack_forward_comm(int, int *, double *,
                                                         int, int *)
{
  error->all(FLERR,
             "chemtrain/kk host pair communication is unsupported. "
             "Use Kokkos device pair communication");
  return 0;
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::unpack_forward_comm(int, int, double *)
{
  error->all(FLERR,
             "chemtrain/kk host pair communication is unsupported. "
             "Use Kokkos device pair communication");
}

template <class DeviceType>
int ChemtrainDeployKokkos<DeviceType>::pack_reverse_comm(int, int, double *)
{
  error->all(FLERR,
             "chemtrain/kk host pair communication is unsupported. "
             "Use Kokkos device pair communication");
  return 0;
}

template <class DeviceType>
void ChemtrainDeployKokkos<DeviceType>::unpack_reverse_comm(int, int *, double *)
{
  error->all(FLERR,
             "chemtrain/kk host pair communication is unsupported. "
             "Use Kokkos device pair communication");
}

namespace LAMMPS_NS {
template class ChemtrainDeployKokkos<LMPDeviceType>;
}
