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

#ifndef LMP_CHEMTRAIN_NEIGHBOR_SPARSE_KOKKOS_H
#define LMP_CHEMTRAIN_NEIGHBOR_SPARSE_KOKKOS_H

#include "atom_kokkos.h"
#include "chemtrain_neighbor.h"
#include "domain.h"
#include "fix_chemtrain_topology_kokkos.h"
#include "kokkos_type.h"
#include "neigh_list_kokkos.h"
#include "neighbor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace LAMMPS_NS {

template <class DeviceType>
class ChemtrainNeighborSparseKokkos : public ChemtrainNeighbor {
 public:
  using AT = ArrayTypes<DeviceType>;

  explicit ChemtrainNeighborSparseKokkos(const JCN_Api *api) : api_(api) {}
  ~ChemtrainNeighborSparseKokkos() override { clear(); }

  void observe(const ChemtrainNeighborObserve &input) override
  {
    auto *next = input.list == nullptr ? nullptr :
        static_cast<NeighListKokkos<DeviceType> *>(input.list);
    row_count_ = input.row_count;
    pending_rebuild_ =
        input.geometric_rebuild || next != list_ || neighbors_ == nullptr;
    list_ = next;
    if (!pending_rebuild_) return;
    observed_ = {};
    if (list_ != nullptr) {
      int max_neighbors = 1;
      analyze(list_, row_count_, observed_.raw_edges, max_neighbors);
      observed_.max_neighbors = max_neighbors;
    }
  }

  [[nodiscard]] bool has_capacity_overflow(
      const jcn_requested_capacities &capacities) const override
  {
    return observed_.raw_edges > capacities.raw_edges;
  }

  void initialize_capacity(jcn_requested_capacities &capacities,
                           double sparse_growth, double) override
  {
    capacities.raw_edges =
        grown_capacity(0, observed_.raw_edges, sparse_growth);
  }

  void grow_capacity_after_collective_overflow(
      jcn_requested_capacities &capacities, double sparse_growth,
      double) override
  {
    if (observed_.raw_edges > capacities.raw_edges ||
        half_padding_consumed(capacities.raw_edges, observed_.raw_edges,
                              sparse_growth))
      capacities.raw_edges = grown_capacity(capacities.raw_edges,
                                             observed_.raw_edges,
                                             sparse_growth);
  }

  ChemtrainNeighborUpdate update(const ChemtrainNeighborBuild &input) override
  {
    const bool capacity_changed =
        max_atoms_ != input.capacities.max_atoms ||
        raw_edges_ != input.capacities.raw_edges ||
        max_neighbors_ != input.capacities.max_neighbors_per_atom;
    const bool mode_changed =
        include_pair_type_ != static_cast<int>(input.include_pair_type);
    const bool rebuild = pending_rebuild_ || capacity_changed || mode_changed;
    if (rebuild) {
      release_jcn_objects();
      build(list_, row_count_, input.nlocal,
            input.capacities.raw_edges, input.capacities.max_atoms,
            input.include_pair_type,
            static_cast<FixChemtrainTopology *>(input.topology),
            static_cast<AtomKokkos *>(input.atom),
            static_cast<Domain *>(input.domain));
      descriptor_ = {};
      descriptor_.senders = input.import_buffer(
          senders_.data(), DLDataType{0, 32, 1}, {input.capacities.raw_edges});
      descriptor_.receivers = input.import_buffer(
          receivers_.data(), DLDataType{0, 32, 1}, {input.capacities.raw_edges});
      buffers_.push_back(descriptor_.senders);
      buffers_.push_back(descriptor_.receivers);
      if (input.include_pair_type) {
        descriptor_.pair_type = input.import_buffer(
            pair_type_.data(), DLDataType{0, 32, 1},
            {input.capacities.raw_edges});
        buffers_.push_back(descriptor_.pair_type);
      }
      jcn_status status{};
      neighbors_ = const_cast<jcn_neighbor_list *>(
          api_->neighbor_list_create_simple_sparse(&descriptor_, &status));
      input.fail("chemtrain/kk failed to create sparse neighbor list",
                 status);
      max_atoms_ = input.capacities.max_atoms;
      raw_edges_ = input.capacities.raw_edges;
      max_neighbors_ = input.capacities.max_neighbors_per_atom;
      include_pair_type_ = input.include_pair_type;
      pending_rebuild_ = false;
    }
    return {neighbors_, rebuild, capacity_changed};
  }

  void clear() override
  {
    release_jcn_objects();
    descriptor_ = {};
    list_ = nullptr;
    observed_ = {};
    pending_rebuild_ = true;
    max_atoms_ = raw_edges_ = max_neighbors_ = 0;
    include_pair_type_ = -1;
  }

  [[nodiscard]] int format() const override
  {
    return JCN_NEIGHBOR_SIMPLE_SPARSE;
  }

  void analyze(NeighListKokkos<DeviceType> *list, int row_count,
               int64_t &raw_edges, int &max_neighbors)
  {
    auto ilist = list->d_ilist;
    auto numneigh = list->d_numneigh;

    raw_edges = 0;
    max_neighbors = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<DeviceType>(0, row_count),
        KOKKOS_LAMBDA(int ii, int64_t &sum, int &maximum) {
          const int central = ilist(ii);
          const int count = numneigh(central);
          if (count > maximum) maximum = count;
          sum += count;
        },
        raw_edges,
        Kokkos::Max<int>(max_neighbors));
    max_neighbors = std::max(max_neighbors, 1);

    if (static_cast<int>(edge_counts_.extent(0)) < std::max(row_count, 1))
      edge_counts_ = typename AT::t_int_1d(
          "chemtrain:edge_counts", std::max(row_count, 1));
    if (static_cast<int>(edge_offsets_.extent(0)) < std::max(row_count + 1, 1))
      edge_offsets_ =
          Kokkos::View<int64_t *, typename DeviceType::array_layout, DeviceType>(
              "chemtrain:edge_offsets", std::max(row_count + 1, 1));
    auto edge_counts = edge_counts_;
    auto edge_offsets = edge_offsets_;
    // The connector graph interface uses the neighbor-list row atom as sender
    // and the listed neighbor as receiver.
    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType>(0, row_count),
        KOKKOS_LAMBDA(int ii) {
          const int central = ilist(ii);
          edge_counts(ii) = numneigh(central);
        });
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<DeviceType>(0, row_count),
        KOKKOS_LAMBDA(int ii, int64_t &update, const bool final) {
          if (final) edge_offsets(ii) = update;
          update += edge_counts(ii);
          if (final && ii == row_count - 1) edge_offsets(row_count) = update;
        });
    if (row_count == 0) Kokkos::deep_copy(edge_offsets_, 0);
  }

  void build(NeighListKokkos<DeviceType> *list, int row_count, int nlocal,
             int64_t raw_edges, int64_t max_atoms,
             bool include_pair_type,
             FixChemtrainTopology *topology, AtomKokkos *atom, Domain *domain)
  {
    if (include_pair_type) bind_topology(topology);

    senders_ =
        typename AT::t_int_1d("chemtrain:sparse_senders", raw_edges);
    receivers_ =
        typename AT::t_int_1d("chemtrain:sparse_receivers", raw_edges);
    if (include_pair_type)
      pair_type_ =
          typename AT::t_int_1d("chemtrain:sparse_pair_type", raw_edges);
    const int padding_index = static_cast<int>(max_atoms);
    Kokkos::deep_copy(senders_, padding_index);
    Kokkos::deep_copy(receivers_, padding_index);
    if (include_pair_type) Kokkos::deep_copy(pair_type_, 0);
    if (list == nullptr) return;

    auto ilist = list->d_ilist;
    auto numneigh = list->d_numneigh;
    auto neighbors = list->d_neighbors;
    auto offsets = edge_offsets_;
    auto senders = senders_;
    auto receivers = receivers_;
    auto pair_type = pair_type_;
    auto topology_counts = topology_counts_;
    auto topology_partners = topology_partners_;
    auto tag = atom->k_tag.template view<DeviceType>();
    auto x = atom->k_x.template view<DeviceType>();
    const int topology_maxspecial = topology_maxspecial_;
    const int xperiodic = domain->xperiodic;
    const int yperiodic = domain->yperiodic;
    const int zperiodic = domain->zperiodic;
    const double xprd_half = domain->xprd_half;
    const double yprd_half = domain->yprd_half;
    const double zprd_half = domain->zprd_half;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType>(0, row_count),
        KOKKOS_LAMBDA(int ii) {
          const int central = ilist(ii);
          const int count = numneigh(central);
          int64_t edge = offsets(ii);
          for (int jj = 0; jj < count; ++jj) {
            const int entry = neighbors(central, jj);
            const int neighbor = entry & NEIGHMASK;
            senders(edge) = central;
            receivers(edge) = neighbor;
            if (include_pair_type)
              pair_type(edge) = category(central, neighbor, entry >> SBBITS & 3,
                                         nlocal, x, tag, topology_counts,
                                         topology_partners, topology_maxspecial,
                                         xperiodic, yperiodic, zperiodic,
                                         xprd_half, yprd_half, zprd_half);
            ++edge;
          }
        });
  }

  [[nodiscard]] typename AT::t_int_1d senders() const { return senders_; }
  [[nodiscard]] typename AT::t_int_1d receivers() const { return receivers_; }
  [[nodiscard]] typename AT::t_int_1d pair_type() const { return pair_type_; }

 private:
  void release_jcn_objects()
  {
    if (api_ != nullptr && neighbors_ != nullptr)
      api_->neighbor_list_destroy(neighbors_);
    neighbors_ = nullptr;
    if (api_ != nullptr)
      for (auto *buffer : buffers_)
        if (buffer != nullptr) api_->buffer_destroy(buffer);
    buffers_.clear();
  }

 public:
  template <class PositionView, class TagView, class CountView, class PartnerView>
  KOKKOS_INLINE_FUNCTION static int category(
      int central, int neighbor, int encoded, int nlocal,
      const PositionView &x,
      const TagView &tag,
      const CountView &counts, const PartnerView &partners, int maxspecial,
      int xperiodic, int yperiodic, int zperiodic, double xprd_half,
      double yprd_half, double zprd_half)
  {
    if (encoded) return encoded;
    const double dx = x(central, 0) - x(neighbor, 0);
    const double dy = x(central, 1) - x(neighbor, 1);
    const double dz = x(central, 2) - x(neighbor, 2);
    if ((xperiodic && fabs(dx) > xprd_half) ||
        (yperiodic && fabs(dy) > yprd_half) ||
        (zperiodic && fabs(dz) > zprd_half))
      return 0;
    const int n2 = counts(3 * central + 1);
    const int n3 = counts(3 * central + 2);
    if (central >= nlocal) {
      const int n1 = counts(3 * central);
      for (int k = 0; k < n3; ++k) {
        if (partners(central * maxspecial + k) != tag(neighbor)) continue;
        return k < n1 ? 1 : (k < n2 ? 2 : 3);
      }
    }

    const int neighbor_n1 = counts(3 * neighbor);
    const int neighbor_offset = neighbor * maxspecial;
    const int central_offset = central * maxspecial;
    for (int ik = n2; ik < n3; ++ik)
      for (int jk = 0; jk < neighbor_n1; ++jk)
        if (partners(central_offset + ik) == partners(neighbor_offset + jk)) return 4;
    return 0;
  }

 private:
  void bind_topology(FixChemtrainTopology *topology)
  {
    auto *topologyKK =
        dynamic_cast<FixChemtrainTopologyKokkos<DeviceType> *>(topology);
    if (topologyKK == nullptr)
      throw std::runtime_error(
          "chemtrain/kk topology fix has incompatible execution space");
    topology_maxspecial_ = topology->maxspecial();
    topology_counts_ = topologyKK->counts_view();
    topology_partners_ = topologyKK->partners_view();
  }

  typename AT::t_int_1d edge_counts_;
  Kokkos::View<int64_t *, typename DeviceType::array_layout, DeviceType> edge_offsets_;
  typename AT::t_int_1d senders_;
  typename AT::t_int_1d receivers_;
  typename AT::t_int_1d pair_type_;
  typename AT::t_int_1d topology_counts_;
  typename AT::t_tagint_1d topology_partners_;
  int topology_maxspecial_ = 0;
  const JCN_Api *api_;
  NeighListKokkos<DeviceType> *list_ = nullptr;
  int row_count_ = 0;
  bool pending_rebuild_ = true;
  ChemtrainNeighborObserved observed_{};
  jcn_sparse_neighbors descriptor_{};
  jcn_neighbor_list *neighbors_ = nullptr;
  std::vector<jcn_buffer *> buffers_;
  int64_t max_atoms_ = 0;
  int64_t raw_edges_ = 0;
  int64_t max_neighbors_ = 0;
  int include_pair_type_ = -1;
};

}    // namespace LAMMPS_NS

#endif
