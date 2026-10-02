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

#ifndef LMP_CHEMTRAIN_NEIGHBOR_SPARSE_H
#define LMP_CHEMTRAIN_NEIGHBOR_SPARSE_H

#include <cstdint>
#include <functional>
#include <vector>

namespace LAMMPS_NS {

class Error;
class NeighList;

class ChemtrainNeighborSparse {
 public:
  using TopologyCategory = std::function<int(int, int, int)>;

  // Builds capacity-sized connector arrays. Unused entries contain the model
  // padding index rather than a LAMMPS atom index.
  void build(NeighList *list, int row_count, int64_t raw_edges,
             int64_t max_atoms, bool include_pair_type,
             const TopologyCategory &topology_category, Error *error);
  void clear();

  [[nodiscard]] int32_t *senders() { return senders_.data(); }
  [[nodiscard]] int32_t *receivers() { return receivers_.data(); }
  [[nodiscard]] int32_t *pair_type() { return pair_type_.data(); }

 private:
  std::vector<int32_t> senders_;
  std::vector<int32_t> receivers_;
  std::vector<int32_t> pair_type_;
};

}    // namespace LAMMPS_NS

#endif
