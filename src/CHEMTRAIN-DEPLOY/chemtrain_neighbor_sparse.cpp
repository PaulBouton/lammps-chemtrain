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

#include "chemtrain_neighbor_sparse.h"

#include "error.h"
#include "neigh_list.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

void ChemtrainNeighborSparse::build(NeighList *list, int row_count,
                                    int64_t raw_edges, int64_t max_atoms,
                                    bool include_pair_type,
                                    const TopologyCategory &topology_category,
                                    Error *error)
{
  senders_.assign(static_cast<std::size_t>(raw_edges),
                  static_cast<int32_t>(max_atoms));
  receivers_.assign(static_cast<std::size_t>(raw_edges),
                    static_cast<int32_t>(max_atoms));
  if (include_pair_type)
    pair_type_.assign(static_cast<std::size_t>(raw_edges), 0);
  else
    pair_type_.clear();

  int64_t edge = 0;
  if (list == nullptr) return;
  // The connector graph interface uses the neighbor-list row atom as sender
  // and the listed neighbor as receiver.
  for (int ii = 0; ii < row_count; ++ii) {
    const int central = list->ilist[ii];
    for (int jj = 0; jj < list->numneigh[central]; ++jj) {
      const int entry = list->firstneigh[central][jj];
      const int neighbor = entry & NEIGHMASK;
      if (edge >= raw_edges)
        error->all(FLERR,
                   "chemtrain sparse neighbor list exceeded "
                   "capacity/edge while filling request");
      senders_[edge] = central;
      receivers_[edge] = neighbor;
      if (include_pair_type)
        pair_type_[edge] = topology_category(central, neighbor, entry >> SBBITS & 3);
      ++edge;
    }
  }
}

void ChemtrainNeighborSparse::clear()
{
  senders_.clear();
  receivers_.clear();
  pair_type_.clear();
}
