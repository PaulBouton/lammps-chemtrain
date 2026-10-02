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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(chemtrain,ChemtrainDeploy);
// clang-format on
#else

#ifndef LMP_PAIR_CHEMTRAIN_DEPLOY_H
#define LMP_PAIR_CHEMTRAIN_DEPLOY_H

#include "pair.h"

#include "chemtrain_deploy_model.h"
#include "chemtrain_settings.h"
#include "connector/jcn_api.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace LAMMPS_NS {

class NeighList;
class FixChemtrainTopology;
class ChemtrainNeighbor;
struct ChemtrainExecutionContext;
struct ChemtrainModelRuntime;

class ChemtrainDeploy : public Pair {
 public:
  ChemtrainDeploy(class LAMMPS *);
  ~ChemtrainDeploy() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  double **cut;
  std::string backend_name;
  int backend_device_ordinal;

  std::vector<ChemtrainSettings::InputMapping> configured_particle_inputs;
  std::vector<ChemtrainSettings::InputMapping> configured_global_inputs;
  std::vector<ChemtrainSettings::OutputRequest> configured_outputs;
  std::unique_ptr<ChemtrainExecutionContext> execution_context;
  std::vector<std::unique_ptr<ChemtrainModelContext>> model_contexts;

  double capacity_atom_growth;
  double capacity_edge_growth;
  double capacity_dense_neighbor_growth;
  bool install_topology_zero_styles;

  // Connector callbacks enter LAMMPS communication while PJRT execution
  // waits. These fields describe only the callback that is currently active.
  void *communication_data;
  int communication_cols;

  bool use_communication;
  // The connector borrows callback error text through a const char pointer.
  // Persistent string storage keeps the text valid until the callback returns.
  std::string communication_error;

  FixChemtrainTopology *topology_lookup;
  std::string topology_fix_id;

  virtual void allocate();

  void resolve_particle_inputs(ChemtrainModelContext &);
  void resolve_global_inputs(ChemtrainModelContext &);
  void resolve_requested_outputs(ChemtrainModelContext &);
  void validate_custom_particle_input(ChemtrainParticleInputBinding &);
  void evaluate_global_inputs(ChemtrainModelContext &);
  void prepare_host_requested_outputs(ChemtrainModelContext &);
  void append_named_outputs(ChemtrainModelContext &, int output_force_rows);

  void ensure_topology_lookup();
  void release_topology_lookup();
  int topology_category(int, int) const;
  int topology_onefive_category(int, int) const;
  void install_missing_topology_styles();
  virtual std::string topology_fix_style() const;

  virtual void initialize_energy_virial(int eflag, int vflag);
  virtual void observe_neighbors(ChemtrainModelContext &);
  // The host implementation imports contiguous buffers. The Kokkos override
  // imports device views while preserving the same connector request layout.
  virtual void build_request(ChemtrainModelContext &, bool allow_internal_recompile,
                             bool force_clear_capacities, jcn_force_request &request);
  virtual void apply_forces(ChemtrainModelContext &);
  virtual void sync_requested_outputs_to_host(ChemtrainModelContext &);
  virtual double reduce_energy(const ChemtrainModelContext &) const;
  virtual void prepare_peratom_energy_storage();
  virtual void apply_energy(const ChemtrainModelContext &);
  virtual void apply_virial(const ChemtrainModelContext &);
  virtual void finish_peratom_energy_storage();
  virtual bool use_host_communication_callback() const;
  virtual bool use_device_communication_callback() const;
  virtual int selected_backend_device_ordinal(const std::string &backend) const;
  virtual jcn_tensor_dtype engine_position_dtype() const;

  // The C callback table delegates here so derived pair styles can provide a
  // device-native implementation without changing connector registration.
  virtual int exchange_features(void *data, int64_t rows, int64_t cols,
                                jcn_communication_scalar_type type, int reverse,
                                const char **error);
  virtual int reduce(void *data, int64_t count, jcn_communication_scalar_type type,
                     const char **error);
  virtual int exchange_device_features(void *data, int64_t rows, int64_t cols,
                                       jcn_communication_scalar_type type, int reverse,
                                       const char *backend, int device_ordinal, void *stream,
                                       const char **error);
  virtual int reduce_device_features(void *data, int64_t count, jcn_communication_scalar_type type,
                                     const char *backend, int device_ordinal, void *stream,
                                     const char **error);

  static int communication_exchange_callback(void *context, void *data, int64_t rows, int64_t cols,
                                             jcn_communication_scalar_type type, int32_t reverse,
                                             const char **error);
  static int communication_device_exchange_callback(void *context, void *data, int64_t rows,
                                                    int64_t cols,
                                                    jcn_communication_scalar_type type,
                                                    int32_t reverse, const char *backend,
                                                    int32_t device_ordinal, void *stream,
                                                    const char **error);
  static int communication_reduce_callback(void *context, void *data, int64_t count,
                                           jcn_communication_scalar_type type, const char **error);
  static int communication_device_reduce_callback(void *context, void *data, int64_t count,
                                                  jcn_communication_scalar_type type,
                                                  const char *backend, int32_t device_ordinal,
                                                  void *stream, const char **error);

  void destroy_buffer(jcn_buffer *&);
  void clear_request_buffers(ChemtrainModelContext &);
  void clear_model_contexts();
  bool capacities_are_uninitialized(const ChemtrainModelContext &) const;
  void initialize_atom_capacity(ChemtrainModelContext &, int64_t atoms);
  void grow_atom_capacity_after_collective_overflow(ChemtrainModelContext &, int64_t atoms);
  jcn_buffer *import_host_buffer(ChemtrainModelContext &, void *data, DLDataType dtype,
                                 const std::vector<int64_t> &shape);
  void ensure_api();
  void destroy_jcn();
  void destroy_model_context(ChemtrainModelContext &);

  void fail_if_jcn_error(const char *context, const jcn_status &status) const;
  ChemtrainModelContext &model_context_for_name(const std::string &);
  const ChemtrainModelContext &model_context_for_name(const std::string &) const;
  bool multiple_model_contexts() const;

 public:
  // Compute styles read the outputs cached by the latest pair evaluation. A
  // query never executes the exported model a second time.
  virtual const double *chemtrain_output(const std::string &, int &, bigint &);
  virtual const double *chemtrain_output(const std::string &, const std::string &, int &, bigint &);
  int chemtrain_output_components(const std::string &, jcn_output_scope &, bool &) const;
  int chemtrain_output_components(const std::string &, const std::string &, jcn_output_scope &,
                                  bool &) const;
};

}    // namespace LAMMPS_NS

#endif
#endif
