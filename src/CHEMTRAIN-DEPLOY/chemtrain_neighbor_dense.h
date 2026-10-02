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

#ifndef LMP_CHEMTRAIN_NEIGHBOR_DENSE_H
#define LMP_CHEMTRAIN_NEIGHBOR_DENSE_H

#include <cstdint>
#include <functional>
#include <vector>

namespace LAMMPS_NS {

class NeighList;

class ChemtrainNeighborDense {
 public:
  using TopologyCategory = std::function<int(int, int, int)>;

  void build(NeighList *, int, int64_t, int64_t, bool,
             const TopologyCategory &);
  void clear();

  [[nodiscard]] int32_t *neighbors() { return neighbors_.data(); }
  [[nodiscard]] int32_t *pair_type() { return pair_type_.data(); }

 private:
  std::vector<int32_t> neighbors_;
  std::vector<int32_t> pair_type_;
};

}    // namespace LAMMPS_NS

#endif
