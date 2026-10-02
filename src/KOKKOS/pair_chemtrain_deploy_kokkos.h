/* -*- c++ -*- ----------------------------------------------------------
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

#if !defined(LMP_KOKKOS)
#else

#ifdef PAIR_CLASS
// clang-format off
PairStyle(chemtrain/kk,ChemtrainDeployKokkos<LMPDeviceType>);
PairStyle(chemtrain/kk/device,ChemtrainDeployKokkos<LMPDeviceType>);
// clang-format on
#else

#ifndef LMP_PAIR_CHEMTRAIN_DEPLOY_KOKKOS_H
#define LMP_PAIR_CHEMTRAIN_DEPLOY_KOKKOS_H

#include "chemtrain_neighbor_sparse_kokkos.h"
#include "kokkos_base.h"
#include "kokkos_type.h"
#include "pair_chemtrain_deploy.h"
#include "pair_kokkos.h"

#include <type_traits>

namespace LAMMPS_NS {

template <class DeviceType> class ChemtrainDeployKokkos :
    public ChemtrainDeploy, public KokkosBase {
 public:
  using device_type = DeviceType;
  using AT = ArrayTypes<DeviceType>;
  using RequestedOutputDeviceView =
      Kokkos::View<float *, typename DeviceType::array_layout, DeviceType>;

  ChemtrainDeployKokkos(class LAMMPS *);
  ~ChemtrainDeployKokkos() override;

  void compute(int, int) override;
  void init_style() override;

  // The methods exchange intermediate per-atom values for communication-enabled
  // executables. The width is known only at runtime, so the Kokkos hooks use the
  // pointer and width stored for the active callback.
  int forward_comm(double *copy_from, double *copy_to, int width);
  int reverse_comm(double *copy_from, double *copy_to, int width);

  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d,
                               DAT::tdual_double_1d &, int, int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d &) override;
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_double_1d &) override;
  void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d,
                                  DAT::tdual_double_1d &) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  void observe_neighbors(ChemtrainModelContext &) override;
  void build_request(ChemtrainModelContext &, bool allow_internal_recompile,
                     bool force_clear_capacities,
                     jcn_force_request &) override;
  void initialize_energy_virial(int, int) override;
  void apply_forces(ChemtrainModelContext &) override;
  void sync_requested_outputs_to_host(ChemtrainModelContext &) override;
  double reduce_energy(const ChemtrainModelContext &) const override;
  void prepare_peratom_energy_storage() override;
  void apply_energy(const ChemtrainModelContext &) override;
  void apply_virial(const ChemtrainModelContext &) override;
  void finish_peratom_energy_storage() override;

  bool use_host_communication_callback() const override;
  bool use_device_communication_callback() const override;
  int selected_backend_device_ordinal(const std::string &) const override;
  jcn_tensor_dtype engine_position_dtype() const override;
  std::string topology_fix_style() const override;

  int exchange_device_features(void *, int64_t, int64_t,
                               jcn_communication_scalar_type, int,
                               const char *, int, void *,
                               const char **) override;
  int reduce_device_features(void *, int64_t, jcn_communication_scalar_type,
                             const char *, int, void *,
                             const char **) override;
  const double *chemtrain_output(const std::string &, int &, bigint &) override;
  const double *chemtrain_output(const std::string &, const std::string &,
                                 int &, bigint &) override;
  RequestedOutputDeviceView chemtrain_output_device(
      const std::string &, const std::string &, int &, bigint &);

  // Reduces connector-owned CUDA scratch without staging through host memory.
  int reduce_device_buffer(double *, int64_t);

 private:
  void require_comm_ready() const;
  struct KokkosModelContextStorage;
  KokkosModelContextStorage &storage_for(const ChemtrainModelContext &);

  class AtomKokkos *atomKK;
  double *communication_data;
  int communication_width;

  using DeviceIntView =
      Kokkos::View<int32_t *, typename DeviceType::array_layout, DeviceType>;
  using HostIntView = Kokkos::View<int32_t *, Kokkos::HostSpace>;
  struct DeviceGlobalInput {
    Kokkos::View<float *, typename DeviceType::array_layout, DeviceType> f32;
    Kokkos::View<double *, typename DeviceType::array_layout, DeviceType> f64;
    Kokkos::View<int32_t *, typename DeviceType::array_layout, DeviceType> s32;
  };
  struct KokkosModelContextStorage {
    std::vector<DeviceIntView> device_particle_inputs;
    std::vector<HostIntView> host_particle_inputs;
    std::vector<DeviceGlobalInput> device_global_inputs;
    std::vector<Kokkos::View<float *, typename DeviceType::array_layout, DeviceType>>
        device_requested_outputs;
    std::vector<Kokkos::View<float *, Kokkos::HostSpace>>
        host_requested_outputs;

    Kokkos::View<float *, typename DeviceType::array_layout, DeviceType> device_positions_f32;
    Kokkos::View<float *, typename DeviceType::array_layout, DeviceType> device_forces_f32;

    Kokkos::View<float *, typename DeviceType::array_layout, DeviceType> device_energy_f32;
    Kokkos::View<float *, typename DeviceType::array_layout, DeviceType> device_virial_f32;
    Kokkos::View<float *, Kokkos::HostSpace> host_virial_f32;
  };
  std::vector<KokkosModelContextStorage> model_context_storages;
  DAT::ttransform_kkacc_1d k_eatom;
  typename AT::t_kkacc_1d d_eatom;
};

}    // namespace LAMMPS_NS

#endif
#endif

#endif
