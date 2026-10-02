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

#include "chemtrain_neighbor.h"

#include "chemtrain_neighbor_dense.h"
#include "chemtrain_neighbor_sparse.h"
#include "neigh_list.h"

#include <algorithm>
#include <utility>

using namespace LAMMPS_NS;

namespace {

class HostChemtrainNeighbor : public ChemtrainNeighbor {
 public:
  HostChemtrainNeighbor(int format, const JCN_Api *api) : format_(format), api_(api) {}
  ~HostChemtrainNeighbor() override = default;

  void observe(const ChemtrainNeighborObserve &input) override
  {
    row_count_ = input.row_count;
    pending_rebuild_ =
        input.geometric_rebuild || input.list != list_ || neighbors_ == nullptr;
    list_ = input.list;
    if (!pending_rebuild_) return;

    observed_ = {};
    if (list_ != nullptr) {
      for (int ii = 0; ii < row_count_; ++ii) {
        const int central = list_->ilist[ii];
        const int count = list_->numneigh[central];
        observed_.raw_edges += static_cast<int64_t>(count);
        observed_.max_neighbors = std::max(observed_.max_neighbors,
                                           static_cast<int64_t>(count));
      }
    }
  }

  [[nodiscard]] bool has_capacity_overflow(
      const jcn_requested_capacities &capacities) const override
  {
    if (format_ == JCN_NEIGHBOR_SIMPLE_SPARSE)
      return observed_.raw_edges > capacities.raw_edges;
    return observed_.max_neighbors > capacities.max_neighbors_per_atom;
  }

  void initialize_capacity(jcn_requested_capacities &capacities,
                           double sparse_growth, double dense_growth) override
  {
    if (format_ == JCN_NEIGHBOR_SIMPLE_SPARSE)
      capacities.raw_edges = grown_capacity(0, observed_.raw_edges, sparse_growth);
    else
      capacities.max_neighbors_per_atom =
          grown_capacity(0, observed_.max_neighbors, dense_growth);
  }

  void grow_capacity_after_collective_overflow(
      jcn_requested_capacities &capacities, double sparse_growth,
      double dense_growth) override
  {
    int64_t &capacity = format_ == JCN_NEIGHBOR_SIMPLE_SPARSE
        ? capacities.raw_edges : capacities.max_neighbors_per_atom;
    const int64_t observed = format_ == JCN_NEIGHBOR_SIMPLE_SPARSE
        ? observed_.raw_edges : observed_.max_neighbors;
    const double growth = format_ == JCN_NEIGHBOR_SIMPLE_SPARSE
        ? sparse_growth : dense_growth;
    if (observed > capacity || half_padding_consumed(capacity, observed, growth))
      capacity = grown_capacity(capacity, observed, growth);
  }

  ChemtrainNeighborUpdate update(const ChemtrainNeighborBuild &build) override
  {
    const bool capacity_changed =
        max_atoms_ != build.capacities.max_atoms ||
        raw_edges_ != build.capacities.raw_edges ||
        max_neighbors_ != build.capacities.max_neighbors_per_atom;
    const bool mode_changed =
        include_pair_type_ != static_cast<int>(build.include_pair_type);
    const bool rebuild = pending_rebuild_ || capacity_changed || mode_changed;

    if (rebuild) {
      release_jcn_objects();
      build_neighbor(build);
      max_atoms_ = build.capacities.max_atoms;
      raw_edges_ = build.capacities.raw_edges;
      max_neighbors_ = build.capacities.max_neighbors_per_atom;
      include_pair_type_ = build.include_pair_type;
      pending_rebuild_ = false;
    }
    return {neighbors_, rebuild, capacity_changed};
  }

  void clear() override
  {
    release_jcn_objects();
    clear_storage();
    list_ = nullptr;
    row_count_ = 0;
    observed_ = {};
    pending_rebuild_ = true;
    max_atoms_ = raw_edges_ = max_neighbors_ = 0;
    include_pair_type_ = -1;
  }

  [[nodiscard]] int format() const override { return format_; }

 protected:
  virtual void build_neighbor(const ChemtrainNeighborBuild &) = 0;
  virtual void clear_storage() = 0;

  void destroy_buffer(jcn_buffer *&buffer)
  {
    if (api_ != nullptr && buffer != nullptr) api_->buffer_destroy(buffer);
    buffer = nullptr;
  }

  void release_jcn_objects()
  {
    if (api_ != nullptr && neighbors_ != nullptr)
      api_->neighbor_list_destroy(neighbors_);
    neighbors_ = nullptr;
    for (auto *&buffer : buffers_) destroy_buffer(buffer);
    buffers_.clear();
  }

  int format_;
  const JCN_Api *api_;
  NeighList *list_ = nullptr;
  int row_count_ = 0;
  ChemtrainNeighborObserved observed_{};
  bool pending_rebuild_ = true;
  jcn_neighbor_list *neighbors_ = nullptr;
  std::vector<jcn_buffer *> buffers_;

 private:
  int64_t max_atoms_ = 0;
  int64_t raw_edges_ = 0;
  int64_t max_neighbors_ = 0;
  int include_pair_type_ = -1;
};

class HostChemtrainNeighborSparse : public HostChemtrainNeighbor {
 public:
  explicit HostChemtrainNeighborSparse(const JCN_Api *api) :
      HostChemtrainNeighbor(JCN_NEIGHBOR_SIMPLE_SPARSE, api) {}
  ~HostChemtrainNeighborSparse() override { clear(); }

 protected:
  void build_neighbor(const ChemtrainNeighborBuild &build) override
  {
    storage_.build(list_, row_count_, build.capacities.raw_edges,
                   build.capacities.max_atoms, build.include_pair_type,
                   build.topology_category, build.error);
    descriptor_ = {};
    descriptor_.senders = build.import_buffer(
        storage_.senders(), DLDataType{0, 32, 1}, {build.capacities.raw_edges});
    descriptor_.receivers = build.import_buffer(
        storage_.receivers(), DLDataType{0, 32, 1}, {build.capacities.raw_edges});
    buffers_.push_back(descriptor_.senders);
    buffers_.push_back(descriptor_.receivers);
    if (build.include_pair_type) {
      descriptor_.pair_type = build.import_buffer(
          storage_.pair_type(), DLDataType{0, 32, 1},
          {build.capacities.raw_edges});
      buffers_.push_back(descriptor_.pair_type);
    }
    jcn_status status{};
    neighbors_ = const_cast<jcn_neighbor_list *>(
        api_->neighbor_list_create_simple_sparse(&descriptor_, &status));
    build.fail("chemtrain: failed to create sparse neighbor list", status);
  }

  void clear_storage() override
  {
    descriptor_ = {};
    storage_.clear();
  }

 private:
  ChemtrainNeighborSparse storage_;
  jcn_sparse_neighbors descriptor_{};
};

class HostChemtrainNeighborDense : public HostChemtrainNeighbor {
 public:
  explicit HostChemtrainNeighborDense(const JCN_Api *api) :
      HostChemtrainNeighbor(JCN_NEIGHBOR_SIMPLE_DENSE, api) {}
  ~HostChemtrainNeighborDense() override { clear(); }

 protected:
  void build_neighbor(const ChemtrainNeighborBuild &build) override
  {
    storage_.build(list_, row_count_, build.capacities.max_atoms,
                   build.capacities.max_neighbors_per_atom,
                   build.include_pair_type, build.topology_category);
    descriptor_ = {};
    descriptor_.neighbors = build.import_buffer(
        storage_.neighbors(), DLDataType{0, 32, 1},
        {build.capacities.max_atoms, build.capacities.max_neighbors_per_atom});
    buffers_.push_back(descriptor_.neighbors);
    if (build.include_pair_type) {
      descriptor_.pair_type = build.import_buffer(
          storage_.pair_type(), DLDataType{0, 32, 1},
          {build.capacities.max_atoms,
           build.capacities.max_neighbors_per_atom});
      buffers_.push_back(descriptor_.pair_type);
    }
    jcn_status status{};
    neighbors_ = const_cast<jcn_neighbor_list *>(
        api_->neighbor_list_create_simple_dense(&descriptor_, &status));
    build.fail("chemtrain: failed to create dense neighbor list", status);
  }

  void clear_storage() override
  {
    descriptor_ = {};
    storage_.clear();
  }

 private:
  ChemtrainNeighborDense storage_;
  jcn_dense_neighbors descriptor_{};
};

}    // namespace

std::unique_ptr<ChemtrainNeighbor> LAMMPS_NS::make_host_chemtrain_neighbor(
    int format, const JCN_Api *api)
{
  if (format == JCN_NEIGHBOR_SIMPLE_SPARSE)
    return std::make_unique<HostChemtrainNeighborSparse>(api);
  if (format == JCN_NEIGHBOR_SIMPLE_DENSE)
    return std::make_unique<HostChemtrainNeighborDense>(api);
  return nullptr;
}
