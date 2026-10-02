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

#include "chemtrain_neighbor_dense.h"

#include "neigh_list.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

void ChemtrainNeighborDense::build(NeighList *list, int row_count,
                                   int64_t max_atoms, int64_t max_neighbors,
                                   bool include_pair_type,
                                   const TopologyCategory &topology_category)
{
  const auto entries = static_cast<std::size_t>(max_atoms) * max_neighbors;
  neighbors_.assign(entries, static_cast<int32_t>(max_atoms));
  if (include_pair_type)
    pair_type_.assign(entries, 0);
  else
    pair_type_.clear();

  if (list == nullptr) return;
  for (int ii = 0; ii < row_count; ++ii) {
    const int sender = list->ilist[ii];
    const auto row = static_cast<std::size_t>(sender) * max_neighbors;
    for (int jj = 0; jj < list->numneigh[sender]; ++jj) {
      const int entry = list->firstneigh[sender][jj];
      const int receiver = entry & NEIGHMASK;
      neighbors_[row + jj] = receiver;
      if (include_pair_type)
        pair_type_[row + jj] = topology_category(sender, receiver, entry >> SBBITS & 3);
    }
  }
}

void ChemtrainNeighborDense::clear()
{
  neighbors_.clear();
  pair_type_.clear();
}
