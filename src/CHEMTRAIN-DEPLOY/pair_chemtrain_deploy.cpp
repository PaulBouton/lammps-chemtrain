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

#include "pair_chemtrain_deploy.h"

#include "angle.h"
#include "arg_info.h"
#include "atom.h"
#include "atom_masks.h"
#include "bond.h"
#include "chemtrain_model_io.h"
#include "chemtrain_neighbor.h"
#include "chemtrain_runtime.h"
#include "chemtrain_settings.h"
#include "citeme.h"
#include "comm.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix_chemtrain_topology.h"
#include "force.h"
#include "improper.h"
#include "input.h"
#include "lmptype.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "special.h"
#include "update.h"
#include "utils.h"
#include "variable.h"

#if defined(LMP_KOKKOS)
#include "atom_kokkos.h"
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace LAMMPS_NS;

static const char cite_chemtrain_deploy[] =
    "pair_style chemtrain command: https://doi.org/10.1021/acs.jctc.5c00996 and "
    "https://doi.org/10.1016/j.cpc.2025.109512\n\n"
    "@Article{fuchsChemtrainDeploy2025,\n"
    " author = {Fuchs, Paul and Chen, Weilong and Thaler, Stephan and Zavadlav, Julija},\n"
    " title = {{chemtrain-deploy}: A Parallel and Scalable Framework for Machine Learning "
    "Potentials in Million-Atom {MD} Simulations},\n"
    " journal = {J. Chem. Theory Comput.},\n"
    " year = {2025},\n"
    " volume = {21},\n"
    " number = {15},\n"
    " pages = {7550--7560},\n"
    " doi = {10.1021/acs.jctc.5c00996}\n"
    "}\n\n"
    "@Article{fuchs2025chemtrain,\n"
    " author = {Fuchs, Paul and Thaler, Stephan and R{\\\"o}cken, Sebastien and Zavadlav, "
    "Julija},\n"
    " title = {{chemtrain}: Learning Deep Potential Models via Automatic Differentiation and "
    "Statistical Physics},\n"
    " journal = {Comput. Phys. Commun.},\n"
    " year = {2025},\n"
    " volume = {310},\n"
    " pages = {109512},\n"
    " doi = {10.1016/j.cpc.2025.109512}\n"
    "}\n\n";

namespace {

// File-local DLPack definitions used by the host pair style.
constexpr uint8_t kDLInt = 0;
constexpr uint8_t kDLFloat = 2;

// MPI_MAX over AttemptDecision gives one collective decision for all ranks:
// any fatal error wins over resize retry, and retry wins over local success.
enum AttemptDecision { ATTEMPT_OK = 0, ATTEMPT_RETRY = 1, ATTEMPT_FATAL = 2 };

struct DlpackHostTensorContext {
  std::vector<int64_t> shape;
};

void delete_dlpack_host_tensor(DLManagedTensor *tensor)
{
  delete static_cast<DlpackHostTensorContext *>(tensor->manager_ctx);
  delete tensor;
}

DLDataType dl_float64()
{
  return DLDataType{kDLFloat, 64, 1};
}

DLDataType dl_int32()
{
  return DLDataType{kDLInt, 32, 1};
}

}    // namespace

// ----------------------------------------------------------------------
// Standard LAMMPS pair-style lifecycle
// ----------------------------------------------------------------------

ChemtrainDeploy::ChemtrainDeploy(LAMMPS *lmp) :
    Pair(lmp), cut(nullptr), backend_device_ordinal(0),
    execution_context(std::make_unique<ChemtrainExecutionContext>()), capacity_atom_growth(1.20),
    capacity_edge_growth(1.50), capacity_dense_neighbor_growth(1.25),
    install_topology_zero_styles(false), communication_data(nullptr), communication_cols(0),
    use_communication(false), topology_lookup(nullptr)
{
  if (lmp->citeme) lmp->citeme->add(cite_chemtrain_deploy);

  writedata = 0;
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 0;
  manybody_flag = 1;
  // The model returns the strain-derived virial directly, so LAMMPS must not
  // add a second force-dot-position contribution after compute().
  no_virial_fdotr_compute = 1;
}

ChemtrainDeploy::~ChemtrainDeploy()
{
  if (copymode) return;
  release_topology_lookup();
  destroy_jcn();
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
  }
}

// ----------------------------------------------------------------------
// Connector object ownership
// ----------------------------------------------------------------------

void ChemtrainDeploy::ensure_api()
{
  static_assert(JCN_API_VERSION == 10, "CHEMTRAIN-DEPLOY requires JCN API version 10");
  if (execution_context->api != nullptr) return;
  execution_context->api = jcn_get_api(JCN_API_VERSION);
  if (execution_context->api == nullptr)
    error->all(FLERR, "chemtrain: could not load compatible JCN API");
  if (execution_context->api->struct_size < sizeof(JCN_Api))
    error->all(FLERR, "chemtrain: loaded JCN API table is too old");
  jcn_runtime_info info{};
  execution_context->api->get_runtime_info(&info);
  if (comm->me == 0) {
    utils::logmesg(lmp, "chemtrain: JCN API {}, build {}, XLA {} ({})\n", info.api_version,
                   info.connector_build_version != nullptr ? info.connector_build_version
                                                           : "unknown",
                   info.xla_commit != nullptr ? info.xla_commit : "unknown",
                   info.xla_sha256 != nullptr ? info.xla_sha256 : "unknown");
  }
}

void ChemtrainDeploy::destroy_jcn()
{
  clear_model_contexts();
  if (execution_context->api != nullptr && execution_context->client != nullptr)
    execution_context->api->client_destroy(execution_context->client);
  execution_context->client = nullptr;
}

void ChemtrainDeploy::destroy_model_context(ChemtrainModelContext &model_context)
{
  clear_request_buffers(model_context);
  auto &runtime = *model_context.runtime;
  if (execution_context->api != nullptr) {
    if (runtime.executor != nullptr) execution_context->api->executor_destroy(runtime.executor);
    if (runtime.model != nullptr) execution_context->api->model_destroy(runtime.model);
  }
  runtime.executor = nullptr;
  runtime.model = nullptr;
  runtime.model_properties = {};
  runtime.capacities = {};
}

void ChemtrainDeploy::destroy_buffer(jcn_buffer *&buffer)
{
  if (execution_context->api != nullptr && buffer != nullptr)
    execution_context->api->buffer_destroy(buffer);
  buffer = nullptr;
}

void ChemtrainDeploy::clear_request_buffers(ChemtrainModelContext &model_context)
{
  auto &runtime = *model_context.runtime;
  if (model_context.model_neighbors != nullptr) model_context.model_neighbors->clear();
  model_context.model_neighbors.reset();
  destroy_buffer(runtime.positions_buffer);
  for (auto &input : model_context.particle_inputs) destroy_buffer(input.cached_buffer);
  for (auto &input : model_context.global_inputs) destroy_buffer(input.cached_buffer);
  for (auto &output : model_context.requested_outputs) destroy_buffer(output.cached_buffer);
  destroy_buffer(runtime.forces_buffer);
  destroy_buffer(runtime.energy_buffer);
  destroy_buffer(runtime.virial_buffer);
  runtime.particle_inputs.clear();
  runtime.global_inputs.clear();
  runtime.outputs.clear();
  runtime.atom_capacity = 0;
}

void ChemtrainDeploy::clear_model_contexts()
{
  for (auto &model_context : model_contexts)
    if (model_context != nullptr) destroy_model_context(*model_context);
  model_contexts.clear();
}

void ChemtrainDeploy::ensure_topology_lookup()
{
  if (topology_lookup != nullptr) return;
  if (atom->molecular == Atom::MOLECULAR && !force->special_onefive &&
      force->special_lj[3] == 1.0 && force->special_coul[3] == 1.0) {
#if defined(LMP_KOKKOS)
    auto *atomKK = dynamic_cast<AtomKokkos *>(atom);
    if (atomKK != nullptr) atomKK->sync(Host, TAG_MASK | BOND_MASK | ANGLE_MASK | DIHEDRAL_MASK);
#endif
    Special(lmp).build(true);
#if defined(LMP_KOKKOS)
    if (atomKK != nullptr) atomKK->modified(Host, SPECIAL_MASK);
#endif
  }
  topology_fix_id = "CHEMTRAIN_TOPOLOGY_" + std::to_string(instance_me);
  Fix *fix = modify->add_fix(topology_fix_id + " all " + topology_fix_style(), 0);
  auto *resolved = dynamic_cast<FixChemtrainTopology *>(fix);
  if (resolved == nullptr)
    error->all(FLERR, "Internal chemtrain topology fix ID is already in use");
  topology_lookup = resolved;
}

std::string ChemtrainDeploy::topology_fix_style() const
{
  return "CHEMTRAIN/TOPOLOGY";
}

void ChemtrainDeploy::release_topology_lookup()
{
  if (topology_lookup == nullptr) return;
  if (modify->nfix && modify->get_fix_by_id(topology_fix_id) != nullptr)
    modify->delete_fix(topology_fix_id);
  topology_lookup = nullptr;
  topology_fix_id.clear();
}

int ChemtrainDeploy::topology_category(int sender, int receiver) const
{
  if (topology_lookup == nullptr) return 0;
  const double dx = atom->x[sender][0] - atom->x[receiver][0];
  const double dy = atom->x[sender][1] - atom->x[receiver][1];
  const double dz = atom->x[sender][2] - atom->x[receiver][2];
  if (domain->minimum_image_check(dx, dy, dz)) return 0;
  const int category = topology_lookup->category(sender, atom->tag[receiver]);
  return category != 0 ? category : topology_lookup->onefive(sender, receiver);
}

int ChemtrainDeploy::topology_onefive_category(int sender, int receiver) const
{
  if (topology_lookup == nullptr) return 0;
  const double dx = atom->x[sender][0] - atom->x[receiver][0];
  const double dy = atom->x[sender][1] - atom->x[receiver][1];
  const double dz = atom->x[sender][2] - atom->x[receiver][2];
  if (domain->minimum_image_check(dx, dy, dz)) return 0;
  return topology_lookup->onefive(sender, receiver);
}

void ChemtrainDeploy::install_missing_topology_styles()
{
  char all[] = "*";
  char *coeff[] = {all};

  if (atom->nbonds > 0 && force->bond == nullptr) {
    force->create_bond("zero", 0);
    force->bond->datamask_read = force->bond->datamask_modify = EMPTY_MASK;
    force->bond->settings(0, nullptr);
    force->bond->coeff(1, coeff);
  }
  if (atom->nangles > 0 && force->angle == nullptr) {
    force->create_angle("zero", 0);
    force->angle->datamask_read = force->angle->datamask_modify = EMPTY_MASK;
    force->angle->settings(0, nullptr);
    force->angle->coeff(1, coeff);
  }
  if (atom->ndihedrals > 0 && force->dihedral == nullptr) {
    force->create_dihedral("zero", 0);
    force->dihedral->datamask_read = force->dihedral->datamask_modify = EMPTY_MASK;
    force->dihedral->settings(0, nullptr);
    force->dihedral->coeff(1, coeff);
  }
  if (atom->nimpropers > 0 && force->improper == nullptr) {
    force->create_improper("zero", 0);
    force->improper->datamask_read = force->improper->datamask_modify = EMPTY_MASK;
    force->improper->settings(0, nullptr);
    force->improper->coeff(1, coeff);
  }
}

int ChemtrainDeploy::selected_backend_device_ordinal(const std::string &backend) const
{
  if (backend == "cpu") return 0;
  return chemtrain_local_rank(lmp);
}

jcn_tensor_dtype ChemtrainDeploy::engine_position_dtype() const
{
  return JCN_TENSOR_DTYPE_F64;
}

// ----------------------------------------------------------------------
// DLPack import helpers
// ----------------------------------------------------------------------

jcn_buffer *ChemtrainDeploy::import_host_buffer(ChemtrainModelContext &model_context, void *data,
                                                DLDataType dtype, const std::vector<int64_t> &shape)
{
  auto &runtime = *model_context.runtime;
  auto *context = new DlpackHostTensorContext{shape};
  auto *tensor = new DLManagedTensor{};
  tensor->dl_tensor.data = data;
  tensor->dl_tensor.device = DLDevice{kDLCPU, 0};
  tensor->dl_tensor.ndim = static_cast<int32_t>(context->shape.size());
  tensor->dl_tensor.dtype = dtype;
  tensor->dl_tensor.shape = context->shape.data();
  tensor->dl_tensor.strides = nullptr;
  tensor->dl_tensor.byte_offset = 0;
  tensor->manager_ctx = context;
  tensor->deleter = delete_dlpack_host_tensor;

  jcn_status status{};
  jcn_buffer_import_options options{};
  options.copy_policy = JCN_DLPACK_COPY_IF_NEEDED;
  options.role = JCN_BUFFER_ROLE_INPUT_OUTPUT;
  options.debug_name = "chemtrain_host_buffer";
  options.device_ordinal = -1;
  jcn_buffer *buffer = execution_context->api->buffer_from_dlpack_options(runtime.executor, tensor,
                                                                          &options, &status);
  if (status.code != JCN_STATUS_OK) {
    delete_dlpack_host_tensor(tensor);
    fail_if_jcn_error("chemtrain: failed to import DLPack buffer", status);
  }
  return buffer;
}

void ChemtrainDeploy::fail_if_jcn_error(const char *context, const jcn_status &status) const
{
  if (status.code == JCN_STATUS_OK) return;
  const char *message = status.message != nullptr ? status.message : "unknown";
  error->all(FLERR, "{}: {}", context, message);
}

void ChemtrainDeploy::prepare_host_requested_outputs(ChemtrainModelContext &model_context)
{
  auto &runtime = *model_context.runtime;
  for (auto &output : model_context.requested_outputs) {
    // Buffers persist across timesteps and use their maximum capacity. Only
    // PARTICLE outputs carry an atom axis.
    const size_t records =
        output.scope == JCN_OUTPUT_PARTICLE ? static_cast<size_t>(runtime.capacities.max_atoms) : 1;
    destroy_buffer(output.cached_buffer);
    output.host_values.resize(records * output.components);
    std::fill(output.host_values.begin(), output.host_values.end(), 0.0);
    std::vector<int64_t> shape = output.dimensions;
    if (output.scope == JCN_OUTPUT_PARTICLE)
      shape.insert(shape.begin(), runtime.capacities.max_atoms);
    output.cached_buffer =
        import_host_buffer(model_context, output.host_values.data(), dl_float64(), shape);
  }
}

void ChemtrainDeploy::append_named_outputs(ChemtrainModelContext &model_context,
                                           int output_force_rows)
{
  auto &runtime = *model_context.runtime;
  runtime.outputs.clear();
  // JCN counts describe valid records rather than buffer capacities. Energy
  // has one record per owned atom, force may include ghost atoms, and each
  // configuration output has one record.
  runtime.outputs.push_back(jcn_named_output{"F", runtime.forces_buffer, output_force_rows});
  runtime.outputs.push_back(jcn_named_output{"U", runtime.energy_buffer, atom->nlocal});
  runtime.outputs.push_back(jcn_named_output{"V", runtime.virial_buffer, 1});
  for (auto &output : model_context.requested_outputs)
    runtime.outputs.push_back(
        jcn_named_output{output.name.c_str(), output.cached_buffer,
                         output.scope == JCN_OUTPUT_PARTICLE ? atom->nlocal : 1});
}

void ChemtrainDeploy::sync_requested_outputs_to_host(ChemtrainModelContext &model_context)
{
  for (auto &output : model_context.requested_outputs) {
    // Stamp only after successful collective execution. Compute styles use the
    // timestep to reject stale output.
    output.host_timestep = update->ntimestep;
    output.device_timestep = update->ntimestep;
  }
}

// ----------------------------------------------------------------------
// Force evaluation
// ----------------------------------------------------------------------

void ChemtrainDeploy::compute(int eflag, int vflag)
{
  initialize_energy_virial(eflag, vflag);
  if (vflag_atom || cvflag_atom)
    error->all(FLERR, "Pair chemtrain does not provide per-atom virial output");

  // All ranks must make the same retry decision for each model. Capacity checks
  // that LAMMPS can perform happen before request buffers are created. After
  // request construction, every rank must enter connector execution in the
  // same order.
  constexpr int max_capacity_trials = 10;
  for (size_t model_index = 0; model_index < model_contexts.size(); ++model_index) {
    auto &model_context = *model_contexts[model_index];
    auto &runtime = *model_context.runtime;
    jcn_force_result result{};
    jcn_status status{};
    int collective_decision = ATTEMPT_OK;
    bool force_collective_recompile = false;
    bool allow_internal_recompile = false;
    int trial = 0;
    for (; trial < max_capacity_trials; ++trial) {
      const int64_t atoms = atom->nlocal + atom->nghost;
      observe_neighbors(model_context);
      const bool local_bootstrap = capacities_are_uninitialized(model_context);
      int collective_bootstrap = 0;
      const int bootstrap_value = local_bootstrap ? 1 : 0;
      if (MPI_Allreduce(&bootstrap_value, &collective_bootstrap, 1, MPI_INT, MPI_MAX, world) !=
          MPI_SUCCESS)
        error->all(FLERR, "chemtrain MPI_Allreduce failed");
      const int local_external_overflow = !local_bootstrap &&
          (atoms > runtime.capacities.max_atoms ||
           model_context.model_neighbors->has_capacity_overflow(runtime.capacities));
      int collective_external_overflow = 0;
      if (MPI_Allreduce(&local_external_overflow, &collective_external_overflow, 1, MPI_INT,
                        MPI_MAX, world) != MPI_SUCCESS)
        error->all(FLERR, "chemtrain MPI_Allreduce failed");

      if (local_bootstrap) {
        initialize_atom_capacity(model_context, atoms);
        model_context.model_neighbors->initialize_capacity(runtime.capacities, capacity_edge_growth,
                                                           capacity_dense_neighbor_growth);
      }
      if (collective_bootstrap != 0) force_collective_recompile = true;
      if (collective_external_overflow != 0) {
        // When any rank exceeds a capacity, every rank enters the growth path.
        // Each rank grows only a buffer that overflowed or used at least half
        // of its spare capacity. The common path keeps later collective
        // connector calls in the same order without forcing every buffer to grow.
        grow_atom_capacity_after_collective_overflow(model_context, atoms);
        model_context.model_neighbors->grow_capacity_after_collective_overflow(
            runtime.capacities, capacity_edge_growth, capacity_dense_neighbor_growth);
        force_collective_recompile = true;
        allow_internal_recompile = true;
      }

      jcn_force_request request{};
      build_request(model_context, allow_internal_recompile, force_collective_recompile, request);
      execution_context->api->executor_set_capacities(runtime.executor, &runtime.capacities,
                                                      &status);
      const int local_setup_failed = status.code == JCN_STATUS_OK ? 0 : 1;
      int collective_setup_failed = 0;
      if (MPI_Allreduce(&local_setup_failed, &collective_setup_failed, 1, MPI_INT, MPI_MAX,
                        world) != MPI_SUCCESS)
        error->all(FLERR, "chemtrain MPI_Allreduce failed");
      if (collective_setup_failed != 0) {
        if (local_setup_failed != 0)
          fail_if_jcn_error("chemtrain: failed to set capacities", status);
        error->all(FLERR, "chemtrain: failed to set capacities collectively");
      }
      if (MPI_Barrier(world) != MPI_SUCCESS) error->all(FLERR, "chemtrain MPI_Barrier failed");
      execution_context->api->executor_compute_forces(runtime.executor, &request, &result, &status);

      int local_decision = ATTEMPT_FATAL;
      if (status.code == JCN_STATUS_OK && result.code == JCN_COMPUTE_OK) {
        local_decision = ATTEMPT_OK;
      } else if (status.code == JCN_STATUS_OK && result.code == JCN_COMPUTE_NEEDS_CAPACITY_CHANGE) {
        local_decision = ATTEMPT_RETRY;
      }
      collective_decision = ATTEMPT_OK;
      // MPI_MAX selects the most severe result reported by any rank: fatal
      // error, capacity retry, or success.
      const int allreduce_status =
          MPI_Allreduce(&local_decision, &collective_decision, 1, MPI_INT, MPI_MAX, world);
      if (allreduce_status != MPI_SUCCESS) error->all(FLERR, "chemtrain MPI_Allreduce failed");
      if (collective_decision == ATTEMPT_FATAL) {
        if (status.code != JCN_STATUS_OK)
          fail_if_jcn_error("chemtrain: force computation failed", status);
        error->all(FLERR, "chemtrain: force computation failed collectively");
      }
      if (collective_decision == ATTEMPT_OK) break;

      // Each rank applies the concrete lower bounds returned by its connector.
      // The collective retry keeps all ranks in the same execution order even
      // when only one rank needs larger adapter-owned buffers.
      if (local_decision == ATTEMPT_RETRY) {
        runtime.capacities.max_atoms =
            std::max(runtime.capacities.max_atoms, result.required_capacities.max_atoms);
        runtime.capacities.raw_edges =
            std::max(runtime.capacities.raw_edges, result.required_capacities.raw_edges);
        runtime.capacities.max_neighbors_per_atom =
            std::max(runtime.capacities.max_neighbors_per_atom,
                     result.required_capacities.max_neighbors_per_atom);
      }

      // The connector reports an internal capacity overflow only after
      // execution. On the next attempt, all ranks authorize internal
      // recompilation. Buffers that did not overflow grow only after crossing
      // the allowed usage threshold.
      force_collective_recompile = true;
      allow_internal_recompile = true;
    }
    if (collective_decision == ATTEMPT_RETRY) {
      error->all(FLERR, "chemtrain exceeded capacity retry limit of {} attempts",
                 max_capacity_trials);
    }

    sync_requested_outputs_to_host(model_context);
  }

  // Apply model results only after every context has finished. Equal-style
  // variables used as global inputs then observe identical pre-application
  // atom forces in every context.
  for (auto &model_context_ptr : model_contexts) {
    auto &model_context = *model_context_ptr;
    if (model_context.scale == 0.0) continue;
    apply_forces(model_context);
    if (eflag) eng_vdwl += model_context.scale * reduce_energy(model_context);
    if (eflag_atom) apply_energy(model_context);
    if (vflag_global) apply_virial(model_context);
  }
  if (eflag_atom) finish_peratom_energy_storage();
}

void ChemtrainDeploy::apply_forces(ChemtrainModelContext &model_context)
{
  // Host DLPack output buffers use contiguous std::vector storage. Runner has
  // already copied the PJRT force output into host_forces. Local rows are added
  // to owned atoms. With Newton pair on, ghost rows are added too, and LAMMPS
  // later returns those force contributions to their owners.
  const int active_rows = atom->nlocal + (force->newton_pair ? atom->nghost : 0);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int i = 0; i < active_rows; ++i) {
    atom->f[i][0] += model_context.scale * model_context.host_forces[3 * i + 0];
    atom->f[i][1] += model_context.scale * model_context.host_forces[3 * i + 1];
    atom->f[i][2] += model_context.scale * model_context.host_forces[3 * i + 2];
  }
}

double ChemtrainDeploy::reduce_energy(const ChemtrainModelContext &model_context) const
{
  double potential = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : potential)
#endif
  for (int i = 0; i < atom->nlocal; ++i) potential += model_context.host_energy[i];
  return potential;
}

void ChemtrainDeploy::prepare_peratom_energy_storage() {}

void ChemtrainDeploy::initialize_energy_virial(int eflag, int vflag)
{
  ev_init(eflag, vflag);
}

void ChemtrainDeploy::apply_energy(const ChemtrainModelContext &model_context)
{
  for (int i = 0; i < atom->nlocal; ++i)
    eatom[i] += model_context.scale * model_context.host_energy[i];
}

void ChemtrainDeploy::apply_virial(const ChemtrainModelContext &model_context)
{
  // The exported V output is -dE/dstrain in LAMMPS component order: xx, yy,
  // zz, xy, xz, yz. The exporter has already applied the virial sign
  // convention, and the result has energy units. Only the model scale remains
  // to be applied.
  for (int i = 0; i < 6; ++i) virial[i] += model_context.scale * model_context.host_virial[i];
}

void ChemtrainDeploy::finish_peratom_energy_storage() {}

bool ChemtrainDeploy::use_device_communication_callback() const
{
  return false;
}

bool ChemtrainDeploy::use_host_communication_callback() const
{
  return true;
}

bool ChemtrainDeploy::capacities_are_uninitialized(const ChemtrainModelContext &model_context) const
{
  return model_context.runtime->capacities.max_atoms <= 0;
}

void ChemtrainDeploy::initialize_atom_capacity(ChemtrainModelContext &model_context, int64_t atoms)
{
  auto &capacities = model_context.runtime->capacities;
  capacities.max_atoms = std::max<int64_t>(
      atoms, static_cast<int64_t>(std::ceil(std::max<int64_t>(atoms, 1) * capacity_atom_growth)));
}

void ChemtrainDeploy::grow_atom_capacity_after_collective_overflow(
    ChemtrainModelContext &model_context, int64_t atoms)
{
  int64_t &capacity = model_context.runtime->capacities.max_atoms;
  const bool overflow = atoms > capacity;
  const bool half_padding_consumed =
      static_cast<long double>(atoms) * 2.0L * capacity_atom_growth >=
      static_cast<long double>(capacity) * (capacity_atom_growth + 1.0L);
  if (overflow || half_padding_consumed)
    capacity =
        std::max<int64_t>(atoms, static_cast<int64_t>(std::ceil(atoms * capacity_atom_growth)));
}

// ----------------------------------------------------------------------
// Request construction and neighbor-list conversion
// ----------------------------------------------------------------------

void ChemtrainDeploy::observe_neighbors(ChemtrainModelContext &model_context)
{
  auto &runtime = *model_context.runtime;
  // Connector graphs include LAMMPS rows for both owned and ghost central
  // atoms. Graph coverage does not depend on communication callbacks.
  const int row_count = list == nullptr ? 0 : list->inum + list->gnum;
  if (list != nullptr && !list->ghost)
    error->all(FLERR,
               "chemtrain graph construction requires ghost-I neighbor rows, "
               "but LAMMPS did not build a ghost neighbor list");
  if (model_context.model_neighbors == nullptr ||
      model_context.model_neighbors->format() != runtime.model_properties.neighbor_format) {
    model_context.model_neighbors = make_host_chemtrain_neighbor(
        runtime.model_properties.neighbor_format, execution_context->api);
    if (model_context.model_neighbors == nullptr)
      error->all(FLERR, "chemtrain model has unsupported neighbor format");
  }
  model_context.model_neighbors->observe({list, row_count, neighbor->ago == 0});
}

void ChemtrainDeploy::build_request(ChemtrainModelContext &model_context,
                                    bool allow_internal_recompile, bool force_clear_capacities,
                                    jcn_force_request &request)
{
  auto &runtime = *model_context.runtime;
  const int nall = atom->nlocal + atom->nghost;

  // Request construction uses the same owned-and-ghost row set as the earlier
  // count-only observation pass.
  const int64_t atom_capacity = runtime.capacities.max_atoms;
  const bool atom_capacity_changed = runtime.atom_capacity != atom_capacity;
  // Virial storage is initialized once with its fixed six-component shape.
  // Its DLPack handle therefore survives changes to atom-sized capacities.
  if (model_context.host_virial.empty()) model_context.host_virial.resize(6);
  if (atom_capacity_changed) {
    destroy_buffer(runtime.positions_buffer);
    for (auto &input : model_context.particle_inputs) {
      destroy_buffer(input.cached_buffer);
      input.host_values.resize(static_cast<size_t>(atom_capacity));
    }
    for (auto &input : model_context.global_inputs) destroy_buffer(input.cached_buffer);
    destroy_buffer(runtime.forces_buffer);
    destroy_buffer(runtime.energy_buffer);
    model_context.host_positions.resize(static_cast<size_t>(atom_capacity) * 3);
    model_context.host_forces.resize(static_cast<size_t>(atom_capacity) * 3);
    model_context.host_energy.resize(static_cast<size_t>(atom_capacity));
    runtime.atom_capacity = atom_capacity;
  }

  std::fill(model_context.host_positions.begin(), model_context.host_positions.end(), 0.0);
  std::fill(model_context.host_forces.begin(), model_context.host_forces.end(), 0.0);
  std::fill(model_context.host_energy.begin(), model_context.host_energy.end(), 0.0);
  std::fill(model_context.host_virial.begin(), model_context.host_virial.end(), 0.0);
  for (auto &input : model_context.particle_inputs)
    std::fill(input.host_values.begin(), input.host_values.end(), 0);
  for (auto &input : model_context.particle_inputs)
    if (input.source_kind == ChemtrainParticleSource::CUSTOM_INT)
      validate_custom_particle_input(input);
  evaluate_global_inputs(model_context);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int i = 0; i < nall; ++i) {
    model_context.host_positions[3 * i + 0] = atom->x[i][0];
    model_context.host_positions[3 * i + 1] = atom->x[i][1];
    model_context.host_positions[3 * i + 2] = atom->x[i][2];
    for (auto &input : model_context.particle_inputs) {
      if (input.source_kind == ChemtrainParticleSource::TYPE_ZERO_BASED)
        input.host_values[i] = atom->type[i] - 1;
      else
        input.host_values[i] = atom->ivector[input.custom_index][i];
    }
  }

  if (runtime.positions_buffer == nullptr)
    runtime.positions_buffer = import_host_buffer(
        model_context, model_context.host_positions.data(), dl_float64(), {atom_capacity, 3});
  for (size_t i = 0; i < model_context.particle_inputs.size(); ++i) {
    auto &input = model_context.particle_inputs[i];
    if (input.cached_buffer == nullptr)
      input.cached_buffer =
          import_host_buffer(model_context, input.host_values.data(), dl_int32(), {atom_capacity});
    runtime.particle_inputs[i] = jcn_named_input{input.name.c_str(), input.cached_buffer};
  }
  if (runtime.forces_buffer == nullptr)
    runtime.forces_buffer = import_host_buffer(model_context, model_context.host_forces.data(),
                                               dl_float64(), {atom_capacity, 3});
  if (runtime.energy_buffer == nullptr)
    runtime.energy_buffer = import_host_buffer(model_context, model_context.host_energy.data(),
                                               dl_float64(), {atom_capacity});
  if (runtime.virial_buffer == nullptr)
    runtime.virial_buffer =
        import_host_buffer(model_context, model_context.host_virial.data(), dl_float64(), {6});
  if (atom_capacity_changed) prepare_host_requested_outputs(model_context);

  for (size_t i = 0; i < model_context.global_inputs.size(); ++i) {
    auto &input = model_context.global_inputs[i];
    void *data = nullptr;
    DLDataType dtype{};
    if (input.dtype == JCN_TENSOR_DTYPE_F32) {
      data = &input.float_value;
      dtype = DLDataType{kDLFloat, 32, 1};
    } else if (input.dtype == JCN_TENSOR_DTYPE_F64) {
      data = &input.double_value;
      dtype = dl_float64();
    } else if (input.dtype == JCN_TENSOR_DTYPE_S32) {
      data = &input.int_value;
      dtype = dl_int32();
    }
    if (input.cached_buffer == nullptr)
      input.cached_buffer = import_host_buffer(model_context, data, dtype, {});
    runtime.global_inputs[i] = jcn_named_input{input.name.c_str(), input.cached_buffer};
  }

  // The pair style owns the storage and neighbor conversion. The host path
  // copies atom->x into contiguous capacity-sized storage and clears the force
  // output buffer written by the exported executable. After execution,
  // apply_forces() accumulates that buffer into atom->f. DLPack handles remain
  // valid until a capacity change recreates the storage.
  request.atoms.num_owned_atoms = atom->nlocal;
  request.atoms.num_valid_atoms = nall;
  request.atoms.positions = runtime.positions_buffer;
  request.atoms.particle_inputs = runtime.particle_inputs.data();
  request.atoms.num_particle_inputs = static_cast<int32_t>(runtime.particle_inputs.size());
  request.atoms.global_inputs = runtime.global_inputs.data();
  request.atoms.num_global_inputs = static_cast<int32_t>(runtime.global_inputs.size());
  request.capacities = runtime.capacities;
  request.allow_internal_recompile = allow_internal_recompile ? 1 : 0;
  const int output_force_rows = atom->nlocal + (force->newton_pair ? atom->nghost : 0);
  append_named_outputs(model_context, output_force_rows);
  request.outputs = runtime.outputs.data();
  request.num_outputs = static_cast<int32_t>(runtime.outputs.size());

  // Encoded special-neighbor bits take precedence. When they are absent,
  // owned rows need only the additional 1-5 lookup, while ghost rows require
  // the identity-based topology lookup.
  ChemtrainNeighborBuild neighbor_build{
      runtime.capacities,
      runtime.model_properties.include_pair_type != 0,
      [this](int sender, int receiver, int encoded) {
        if (encoded != 0) return encoded;
        if (sender < atom->nlocal) return topology_onefive_category(sender, receiver);
        return topology_category(sender, receiver);
      },
      [this, &model_context](void *data, DLDataType dtype, const std::vector<int64_t> &shape) {
        return import_host_buffer(model_context, data, dtype, shape);
      },
      [this](const char *context, const jcn_status &status) {
        fail_if_jcn_error(context, status);
      },
      error,
      atom->nlocal,
      topology_lookup,
      atom,
      domain};
  const auto neighbor_update = model_context.model_neighbors->update(neighbor_build);
  request.neighbors = neighbor_update.neighbors;
  // Capacity and neighbor changes invalidate separate connector caches.
  request.clear_capacities =
      (force_clear_capacities || atom_capacity_changed || neighbor_update.capacity_changed) ? 1 : 0;
  request.clear_neighbors = neighbor_update.rebuilt ? 1 : 0;
}

// ----------------------------------------------------------------------
// LAMMPS coefficient storage
// ----------------------------------------------------------------------

void ChemtrainDeploy::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cut, n + 1, n + 1, "pair:cut");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++) setflag[i][j] = 0;
}

// ----------------------------------------------------------------------
// LAMMPS input parsing and model attachment
// ----------------------------------------------------------------------

void ChemtrainDeploy::settings(int narg, char **arg)
{
  const ChemtrainSettings parsed = parse_chemtrain_settings(lmp, narg, arg);
  ensure_api();
  destroy_jcn();

  backend_name = parsed.backend;
  capacity_atom_growth = parsed.capacity_atom_growth;
  capacity_edge_growth = parsed.capacity_edge_growth;
  capacity_dense_neighbor_growth = parsed.capacity_dense_neighbor_growth;
  use_communication = parsed.use_communication;
  install_topology_zero_styles = parsed.install_topology_zero_styles;
  configured_particle_inputs = parsed.particle_inputs;
  configured_global_inputs = parsed.global_inputs;
  configured_outputs = parsed.outputs;
  clear_model_contexts();
  if (parsed.model_contexts.empty()) {
    auto model_context = std::make_unique<ChemtrainModelContext>();
    model_context->runtime = std::make_unique<ChemtrainModelRuntime>();
    model_context->scale = 1.0;
    model_contexts.push_back(std::move(model_context));
  } else {
    for (const auto &declared : parsed.model_contexts) {
      auto model_context = std::make_unique<ChemtrainModelContext>();
      model_context->name = declared.name;
      model_context->scale = declared.scale;
      model_context->runtime = std::make_unique<ChemtrainModelRuntime>();
      model_contexts.push_back(std::move(model_context));
    }
  }

  jcn_client_options options{};
  options.backend = backend_name.c_str();
  options.device_ordinal = selected_backend_device_ordinal(backend_name);
  options.memory_fraction = static_cast<float>(parsed.memory_fraction);
  backend_device_ordinal = options.device_ordinal;

  jcn_status status{};
  execution_context->client = execution_context->api->client_create(&options, &status);
  fail_if_jcn_error("chemtrain: failed to create JCN client", status);
}

void ChemtrainDeploy::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  const int expected_args = multiple_model_contexts() ? 2 + static_cast<int>(model_contexts.size())
                                                      : 3;
  if (narg != expected_args)
    error->all(FLERR, Error::ARGZERO, "Incorrect number of arguments for pair_coeff chemtrain");
  if (execution_context->api == nullptr || execution_context->client == nullptr)
    error->all(FLERR, "chemtrain settings must be called before coeff");

  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "chemtrain requires pair_coeff * *");

  for (size_t model_index = 0; model_index < model_contexts.size(); ++model_index) {
    auto &model_context = *model_contexts[model_index];
    auto &runtime = *model_context.runtime;
    const char *filename = arg[2 + model_index];
    const std::vector<char> bytes = read_chemtrain_model(lmp, filename);
    jcn_status status{};
    if (runtime.model != nullptr) destroy_model_context(model_context);
    runtime.executor = execution_context->api->executor_create(execution_context->client, &status);
    fail_if_jcn_error("chemtrain: failed to create JCN executor", status);
    runtime.model = execution_context->api->model_load_from_protobuf(
        bytes.data(), bytes.size(), &runtime.model_properties, &status);
    fail_if_jcn_error("chemtrain: failed to load model", status);
    model_context.filename = filename;

    // Communication callbacks must be registered first because the connector
    // checks communication requirements during model attachment.
    if (use_communication) {
      jcn_communication_callbacks callbacks{};
      callbacks.context = this;
      callbacks.exchange = use_host_communication_callback()
          ? &ChemtrainDeploy::communication_exchange_callback
          : nullptr;
      callbacks.reduce = use_host_communication_callback()
          ? &ChemtrainDeploy::communication_reduce_callback
          : nullptr;
      callbacks.device_exchange = use_device_communication_callback()
          ? &ChemtrainDeploy::communication_device_exchange_callback
          : nullptr;
      callbacks.device_reduce = use_device_communication_callback()
          ? &ChemtrainDeploy::communication_device_reduce_callback
          : nullptr;
      // Host execution registers the host exchange and reduction callbacks
      // together. The Kokkos override registers the corresponding device
      // callbacks. Mixing host and device callbacks would pass buffers to the
      // wrong communication path.
      execution_context->api->executor_set_communication_callbacks(runtime.executor, &callbacks,
                                                                   &status);
      fail_if_jcn_error("chemtrain: failed to register communication callbacks", status);
    }

    jcn_engine_abi_options engine_abi{};
    engine_abi.position_dtype = engine_position_dtype();
    engine_abi.species_dtype = JCN_TENSOR_DTYPE_S32;
    engine_abi.default_output_dtype = engine_position_dtype();
    // The engine ABI uses f64 for both gather and reduction buffers so both
    // communication paths use the same scalar type.
    engine_abi.communication_dtype = JCN_TENSOR_DTYPE_F64;
    execution_context->api->executor_set_engine_abi_options(runtime.executor, &engine_abi, &status);
    fail_if_jcn_error("chemtrain: failed to configure engine ABI", status);

    jcn_model_options options{};
    options.newton_pair = force->newton_pair ? 1 : 0;
    options.use_communication = use_communication ? 1 : 0;
    execution_context->api->executor_set_model_options(runtime.executor, runtime.model, &options,
                                                       &runtime.model_properties, &status);
    fail_if_jcn_error("chemtrain: failed to attach model", status);

    // Validate metadata again at the LAMMPS adapter boundary. The checks keep
    // a mismatched connector implementation from exposing invalid pointers or
    // shapes to the pair style.
    if (runtime.model_properties.num_outputs < 0 ||
        (runtime.model_properties.num_outputs > 0 && runtime.model_properties.outputs == nullptr))
      error->all(FLERR, "chemtrain model has an invalid output table");
    for (int i = 0; i < runtime.model_properties.num_outputs; ++i) {
      const auto &output = runtime.model_properties.outputs[i];
      if (output.name == nullptr || output.name[0] == '\0')
        error->all(FLERR, "chemtrain model output {} has no name", i);
      for (int j = 0; j < i; ++j)
        if (strcmp(output.name, runtime.model_properties.outputs[j].name) == 0)
          error->all(FLERR, "chemtrain model output {} is declared more than once", output.name);
      if (output.scope != JCN_OUTPUT_PARTICLE && output.scope != JCN_OUTPUT_LOCAL &&
          output.scope != JCN_OUTPUT_GLOBAL)
        error->all(FLERR, "chemtrain model output {} has invalid scope {}", output.name,
                   static_cast<int>(output.scope));
      if (output.rank < 0 || (output.rank > 0 && output.dimensions == nullptr))
        error->all(FLERR, "chemtrain model output {} has invalid dimensions", output.name);

      int64_t component_count = 1;
      for (int j = 0; j < output.rank; ++j) {
        const int64_t dimension = output.dimensions[j];
        if (dimension < 1 || component_count > INT_MAX / dimension)
          error->all(FLERR, "chemtrain model output {} has invalid dimensions", output.name);
        component_count *= dimension;
      }
      if (output.components < 1 || output.components != component_count)
        error->all(FLERR, "chemtrain model output {} has inconsistent shape", output.name);
    }

    bool provides_virial = false;
    for (int i = 0; i < runtime.model_properties.num_outputs; ++i) {
      const auto &output = runtime.model_properties.outputs[i];
      if (strcmp(output.name, "V") == 0 && output.scope == JCN_OUTPUT_LOCAL && output.rank == 1 &&
          output.dimensions[0] == 6 && output.components == 6 && output.extensive)
        provides_virial = true;
    }
    if (!provides_virial)
      error->all(FLERR, "chemtrain model must provide extensive LOCAL output V[6]");
    if (runtime.model_properties.unit_style != nullptr &&
        strlen(runtime.model_properties.unit_style) > 0 &&
        strcmp(runtime.model_properties.unit_style, update->unit_style) != 0)
      error->all(FLERR, "chemtrain model {} uses {} units, but LAMMPS is using {} units", filename,
                 runtime.model_properties.unit_style, update->unit_style);
    resolve_particle_inputs(model_context);
    resolve_global_inputs(model_context);
    resolve_requested_outputs(model_context);
  }

  int ilo, ihi, jlo, jhi;
  utils::bounds_typelabel(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, lmp, Atom::ATOM);
  utils::bounds_typelabel(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, lmp, Atom::ATOM);
  for (int i = ilo; i <= ihi; i++)
    for (int j = MAX(jlo, i); j <= jhi; j++) setflag[i][j] = 1;

  int max_comm_width = 0;
  for (const auto &model_context : model_contexts)
    max_comm_width = std::max(max_comm_width,
                              model_context->runtime->model_properties.communication_buffer_width);
  comm_forward = max_comm_width;
  comm_reverse = max_comm_width;
}

// ----------------------------------------------------------------------
// Neighbor-list request setup
// ----------------------------------------------------------------------

void ChemtrainDeploy::init_style()
{
  if (install_topology_zero_styles) install_missing_topology_styles();

  double max_comm_dist = 0.0;
  bool need_full = false;
  bool include_pair_type = false;
  int max_comm_width = 0;
  jcn_neighbor_format neighbor_format = JCN_NEIGHBOR_UNSPECIFIED;
  for (const auto &model_context : model_contexts) {
    const auto &properties = model_context->runtime->model_properties;
    max_comm_dist = std::max(max_comm_dist, properties.comm_dist);
    need_full = need_full || !properties.half_list;
    include_pair_type = include_pair_type || properties.include_pair_type;
    max_comm_width = std::max(max_comm_width, properties.communication_buffer_width);
    if (neighbor_format == JCN_NEIGHBOR_UNSPECIFIED)
      neighbor_format = properties.neighbor_format;
    else if (neighbor_format != properties.neighbor_format)
      error->all(FLERR, "chemtrain model contexts require the same neighbor format");
  }

  if (use_communication && !force->newton_pair)
    error->all(FLERR, "chemtrain communication models require newton pair on");

  comm->cutghostuser = std::max(comm->cutghostuser, max_comm_dist + neighbor->skin);
  // Owned and ghost central rows make graph coverage independent of the
  // endpoint where a model collects edge values.
  int request = NeighConst::REQ_DEFAULT | NeighConst::REQ_GHOST;
  if (need_full) request |= NeighConst::REQ_FULL;
  neighbor->add_request(this, request);
  if (include_pair_type) {
    neighbor->find_request(this)->enable_special();
    ensure_topology_lookup();
  } else
    release_topology_lookup();
  comm_forward = max_comm_width;
  comm_reverse = max_comm_width;
}

double ChemtrainDeploy::init_one(int i, int j)
{
  if (!allocated) allocate();
  if (setflag[i][j] == 0) error->all(FLERR, "Not all pair coeffs are set");
  double cutoff = 0.0;
  for (const auto &model_context : model_contexts)
    cutoff = std::max(cutoff, model_context->runtime->model_properties.cutoff);
  return cutoff;
}

// ----------------------------------------------------------------------
// LAMMPS feature communication hooks
// ----------------------------------------------------------------------

int ChemtrainDeploy::communication_exchange_callback(void *context, void *data, int64_t rows,
                                                     int64_t cols,
                                                     jcn_communication_scalar_type type,
                                                     int32_t reverse, const char **error)
{
  auto *pair = static_cast<ChemtrainDeploy *>(context);
  if (pair == nullptr) {
    if (error != nullptr) *error = "chemtrain communication context is null";
    return 1;
  }
  return pair->exchange_features(data, rows, cols, type, reverse, error);
}

int ChemtrainDeploy::communication_device_exchange_callback(
    void *context, void *data, int64_t rows, int64_t cols, jcn_communication_scalar_type type,
    int32_t reverse, const char *backend, int32_t device_ordinal, void *stream, const char **error)
{
  auto *pair = static_cast<ChemtrainDeploy *>(context);
  if (pair == nullptr) {
    if (error != nullptr) *error = "chemtrain communication context is null";
    return 1;
  }
  return pair->exchange_device_features(data, rows, cols, type, reverse, backend, device_ordinal,
                                        stream, error);
}

int ChemtrainDeploy::communication_reduce_callback(void *context, void *data, int64_t count,
                                                   jcn_communication_scalar_type type,
                                                   const char **error)
{
  auto *pair = static_cast<ChemtrainDeploy *>(context);
  if (pair == nullptr) {
    if (error != nullptr) *error = "chemtrain communication context is null";
    return 1;
  }
  return pair->reduce(data, count, type, error);
}

int ChemtrainDeploy::communication_device_reduce_callback(void *context, void *data, int64_t count,
                                                          jcn_communication_scalar_type type,
                                                          const char *backend,
                                                          int32_t device_ordinal, void *stream,
                                                          const char **error)
{
  auto *pair = static_cast<ChemtrainDeploy *>(context);
  if (pair == nullptr) {
    if (error != nullptr) *error = "chemtrain communication context is null";
    return 1;
  }
  return pair->reduce_device_features(data, count, type, backend, device_ordinal, stream, error);
}

int ChemtrainDeploy::reduce(void *data, int64_t count, jcn_communication_scalar_type type,
                            const char **callback_error)
{
  if (data == nullptr || count <= 0 || count > INT_MAX) {
    communication_error = "chemtrain received an invalid reduce buffer";
    if (callback_error != nullptr) *callback_error = communication_error.c_str();
    return 1;
  }

  if (type != JCN_COMMUNICATION_F64) {
    communication_error = "chemtrain reduce expects f64 buffers";
    if (callback_error != nullptr) *callback_error = communication_error.c_str();
    return 1;
  }

  auto *send = static_cast<double *>(data);
  std::vector<double> recv(static_cast<size_t>(count));
  const int rc =
      MPI_Allreduce(send, recv.data(), static_cast<int>(count), MPI_DOUBLE, MPI_SUM, world);
  if (rc != MPI_SUCCESS) {
    communication_error = "chemtrain MPI_Allreduce failed";
    if (callback_error != nullptr) *callback_error = communication_error.c_str();
    return 1;
  }
  std::copy(recv.begin(), recv.end(), send);
  return 0;
}

int ChemtrainDeploy::exchange_features(void *data, int64_t rows, int64_t cols,
                                       jcn_communication_scalar_type type, int reverse,
                                       const char **callback_error)
{
  if (data == nullptr || cols <= 0) {
    communication_error = "chemtrain received an empty communication buffer";
    if (callback_error != nullptr) *callback_error = communication_error.c_str();
    return 1;
  }
  if (type != JCN_COMMUNICATION_F64) {
    communication_error = "chemtrain host communication expects f64 buffers";
    if (callback_error != nullptr) *callback_error = communication_error.c_str();
    return 1;
  }
  if (cols > INT_MAX) {
    communication_error = "chemtrain communication width exceeds integer range";
    if (callback_error != nullptr) *callback_error = communication_error.c_str();
    return 1;
  }
  if (rows < atom->nlocal + atom->nghost) {
    communication_error = "chemtrain communication buffer has fewer rows than LAMMPS atoms";
    if (callback_error != nullptr) *callback_error = communication_error.c_str();
    return 1;
  }

  // Runner calls ChemtrainDeploy::exchange_features from the PJRT service loop
  // while execution is paused at an XLA FFI communication call. The data
  // pointer addresses the executable's value buffer. LAMMPS communication
  // fills ghost rows in the forward direction or adds ghost-row contributions
  // to owned rows in the reverse direction.
  communication_data = data;
  communication_cols = static_cast<int>(cols);
  // LAMMPS reads these widths during the synchronous pack and unpack calls.
  // The next connector callback overwrites them before communication starts.
  comm_forward = communication_cols;
  comm_reverse = communication_cols;

  if (reverse)
    comm->reverse_comm(this);
  else
    comm->forward_comm(this);
  communication_data = nullptr;
  communication_cols = 0;
  return 0;
}

int ChemtrainDeploy::exchange_device_features(void *, int64_t, int64_t,
                                              jcn_communication_scalar_type, int, const char *, int,
                                              void *, const char **callback_error)
{
  communication_error = "chemtrain device communication requires the Kokkos device pair style";
  if (callback_error != nullptr) *callback_error = communication_error.c_str();
  return 1;
}

int ChemtrainDeploy::reduce_device_features(void *, int64_t, jcn_communication_scalar_type,
                                            const char *, int, void *, const char **callback_error)
{
  communication_error = "chemtrain device reduce requires the Kokkos device pair style";
  if (callback_error != nullptr) *callback_error = communication_error.c_str();
  return 1;
}

int ChemtrainDeploy::pack_forward_comm(int n, int *list, double *buf, int, int *)
{
  if (communication_data == nullptr || communication_cols <= 0)
    error->all(FLERR, "chemtrain communication buffer was not initialized");
  if (n > INT_MAX / communication_cols)
    error->all(FLERR, "chemtrain forward communication buffer is too large");

  auto *data = static_cast<double *>(communication_data);
  for (int i = 0; i < n; ++i) {
    int64_t src = static_cast<int64_t>(list[i]) * communication_cols;
    int64_t dst = static_cast<int64_t>(i) * communication_cols;
    for (int j = 0; j < communication_cols; ++j) { buf[dst++] = data[src++]; }
  }
  return n * communication_cols;
}

void ChemtrainDeploy::unpack_forward_comm(int n, int first, double *buf)
{
  if (communication_data == nullptr || communication_cols <= 0)
    error->all(FLERR, "chemtrain communication buffer was not initialized");

  auto *data = static_cast<double *>(communication_data);
  for (int i = 0; i < n; ++i) {
    int64_t dst = (static_cast<int64_t>(first) + i) * communication_cols;
    int64_t src = static_cast<int64_t>(i) * communication_cols;
    for (int j = 0; j < communication_cols; ++j) { data[dst++] = buf[src++]; }
  }
}

int ChemtrainDeploy::pack_reverse_comm(int n, int first, double *buf)
{
  if (communication_data == nullptr || communication_cols <= 0)
    error->all(FLERR, "chemtrain communication buffer was not initialized");
  if (n > INT_MAX / communication_cols)
    error->all(FLERR, "chemtrain reverse communication buffer is too large");

  auto *data = static_cast<double *>(communication_data);
  for (int i = 0; i < n; ++i) {
    int64_t src = (static_cast<int64_t>(first) + i) * communication_cols;
    int64_t dst = static_cast<int64_t>(i) * communication_cols;
    for (int j = 0; j < communication_cols; ++j) { buf[dst++] = data[src++]; }
  }
  return n * communication_cols;
}

void ChemtrainDeploy::unpack_reverse_comm(int n, int *list, double *buf)
{
  if (communication_data == nullptr || communication_cols <= 0)
    error->all(FLERR, "chemtrain communication buffer was not initialized");

  auto *data = static_cast<double *>(communication_data);
  for (int i = 0; i < n; ++i) {
    int64_t dst = static_cast<int64_t>(list[i]) * communication_cols;
    int64_t src = static_cast<int64_t>(i) * communication_cols;
    for (int j = 0; j < communication_cols; ++j) { data[dst++] += buf[src++]; }
  }
}
