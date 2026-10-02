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

#ifndef LMP_CHEMTRAIN_NEIGHBOR_H
#define LMP_CHEMTRAIN_NEIGHBOR_H

#include "connector/jcn_api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace LAMMPS_NS {

class Error;
class NeighList;

struct ChemtrainNeighborObserve {
  NeighList *list;
  int row_count;
  bool geometric_rebuild;
};

struct ChemtrainNeighborObserved {
  int64_t raw_edges = 0;
  int64_t max_neighbors = 1;
};

struct ChemtrainNeighborBuild {
  const jcn_requested_capacities &capacities;
  bool include_pair_type;

  std::function<int(int, int, int)> topology_category;
  std::function<jcn_buffer *(void *, DLDataType,
                             const std::vector<int64_t> &)> import_buffer;
  std::function<void(const char *, const jcn_status &)> fail;

  Error *error;
  int nlocal;
  void *topology;
  void *atom;
  void *domain;
};

struct ChemtrainNeighborUpdate {
  jcn_neighbor_list *neighbors = nullptr;
  bool rebuilt = false;
  bool capacity_changed = false;
};

class ChemtrainNeighbor {
 public:
  virtual ~ChemtrainNeighbor() = default;

  // Observation is a count-only pass used for the collective capacity
  // decision. update() materializes connector buffers only after ranks agree.
  virtual void observe(const ChemtrainNeighborObserve &) = 0;
  [[nodiscard]] virtual bool has_capacity_overflow(
      const jcn_requested_capacities &) const = 0;
  virtual void initialize_capacity(jcn_requested_capacities &, double sparse_growth,
                                   double dense_growth) = 0;
  virtual void grow_capacity_after_collective_overflow(
      jcn_requested_capacities &, double sparse_growth, double dense_growth) = 0;
  virtual ChemtrainNeighborUpdate update(const ChemtrainNeighborBuild &) = 0;

  virtual void clear() = 0;
  [[nodiscard]] virtual int format() const = 0;

 protected:
  static int64_t grown_capacity(int64_t current, int64_t observed, double growth)
  {
    observed = std::max<int64_t>(observed, 1);
    return std::max(
        {current, observed,
         static_cast<int64_t>(std::ceil(observed * growth))});
  }

  static bool half_padding_consumed(int64_t capacity, int64_t observed,
                                    double growth)
  {
    return static_cast<long double>(observed) * 2.0L * growth >=
        static_cast<long double>(capacity) * (growth + 1.0L);
  }
};

std::unique_ptr<ChemtrainNeighbor> make_host_chemtrain_neighbor(
    int format, const JCN_Api *api);

}    // namespace LAMMPS_NS

#endif
