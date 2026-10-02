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

#include "fix_chemtrain_topology.h"

#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "error.h"
#include "molecule.h"
#include <cstddef>

using namespace LAMMPS_NS;

FixChemtrainTopology::FixChemtrainTopology(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR, "Illegal internal chemtrain topology fix");
}

int FixChemtrainTopology::setmask()
{
  return FixConst::PRE_NEIGHBOR | FixConst::MIN_PRE_NEIGHBOR;
}

void FixChemtrainTopology::setup_pre_neighbor()
{
  refresh();
}

void FixChemtrainTopology::pre_neighbor()
{
  refresh();
}

void FixChemtrainTopology::min_pre_neighbor()
{
  refresh();
}

void FixChemtrainTopology::refresh()
{
  snapshot_owned();
  forward_host();
  ++generation_;
}

void FixChemtrainTopology::snapshot_owned()
{
  counts_.assign(static_cast<std::size_t>(atom->nmax), {0, 0, 0});
  maxspecial_ = atom->molecular == Atom::TEMPLATE ? 0 : atom->maxspecial;

  if (atom->molecular == Atom::TEMPLATE) {
    const int nset = atom->avec->nset;
    for (int imol = 0; imol < nset; ++imol)
      maxspecial_ = MAX(maxspecial_, atom->avec->onemols[imol]->maxspecial);

    partners_.assign(static_cast<std::size_t>(atom->nmax) * maxspecial_, 0);
    for (int i = 0; i < atom->nlocal; ++i) {
      const int imol = atom->molindex[i];
      const int iatom = atom->molatom[i];
      if (imol < 0 || imol >= nset || iatom < 0 ||
          iatom >= atom->avec->onemols[imol]->natoms)
        continue;
      const Molecule *molecule = atom->avec->onemols[imol];
      counts_[i] = {molecule->nspecial[iatom][0], molecule->nspecial[iatom][1],
                    molecule->nspecial[iatom][2]};
      const tagint tag_offset = atom->tag[i] - iatom - 1;
      for (int j = 0; j < counts_[i][2]; ++j)
        partners_[static_cast<std::size_t>(i) * maxspecial_ + j] =
            tag_offset + molecule->special[iatom][j];
    }
  } else if (atom->molecular == Atom::MOLECULAR) {
    partners_.assign(static_cast<std::size_t>(atom->nmax) * maxspecial_, 0);
    for (int i = 0; i < atom->nlocal; ++i) {
      counts_[i] = {atom->nspecial[i][0], atom->nspecial[i][1], atom->nspecial[i][2]};
      for (int j = 0; j < counts_[i][2]; ++j)
        partners_[static_cast<std::size_t>(i) * maxspecial_ + j] = atom->special[i][j];
    }
  } else {
    partners_.clear();
  }
}

void FixChemtrainTopology::forward_host()
{
  const int width = 3 + maxspecial_;
  communication_storage_.assign(
      static_cast<std::size_t>(atom->nmax) * width, 0.0);
  communication_rows_.resize(atom->nmax);
  for (int i = 0; i < atom->nmax; ++i)
    communication_rows_[i] =
        communication_storage_.data() + static_cast<std::size_t>(i) * width;
  for (int i = 0; i < atom->nlocal; ++i) {
    communication_rows_[i][0] = ubuf(counts_[i][0]).d;
    communication_rows_[i][1] = ubuf(counts_[i][1]).d;
    communication_rows_[i][2] = ubuf(counts_[i][2]).d;
    const std::size_t offset = static_cast<std::size_t>(i) * maxspecial_;
    for (int j = 0; j < maxspecial_; ++j)
      communication_rows_[i][3 + j] = ubuf(partners_[offset + j]).d;
  }
  if (atom->nghost)
    comm->forward_comm_array(width, communication_rows_.data());
  for (int i = atom->nlocal; i < atom->nlocal + atom->nghost; ++i) {
    counts_[i][0] = static_cast<int>(ubuf(communication_rows_[i][0]).i);
    counts_[i][1] = static_cast<int>(ubuf(communication_rows_[i][1]).i);
    counts_[i][2] = static_cast<int>(ubuf(communication_rows_[i][2]).i);
    const std::size_t offset = static_cast<std::size_t>(i) * maxspecial_;
    for (int j = 0; j < maxspecial_; ++j)
      partners_[offset + j] =
          static_cast<tagint>(ubuf(communication_rows_[i][3 + j]).i);
  }
}

int FixChemtrainTopology::category(int i, tagint partner) const
{
  if (i < 0 || static_cast<std::size_t>(i) >= counts_.size()) return 0;
  const auto &count = counts_[i];
  const std::size_t offset = static_cast<std::size_t>(i) * maxspecial_;
  for (int j = 0; j < count[2]; ++j) {
    if (partners_[offset + j] != partner) continue;
    if (j < count[0]) return 1;
    if (j < count[1]) return 2;
    return 3;
  }
  return 0;
}

int FixChemtrainTopology::onefive(int i, int j) const
{
  if (i < 0 || j < 0 || static_cast<std::size_t>(i) >= counts_.size() ||
      static_cast<std::size_t>(j) >= counts_.size())
    return 0;

  const auto &icount = counts_[i];
  const auto &jcount = counts_[j];
  const std::size_t ioffset = static_cast<std::size_t>(i) * maxspecial_;
  const std::size_t joffset = static_cast<std::size_t>(j) * maxspecial_;

  // A four-bond path i-a-b-c-j exists exactly when an exact 1-4 partner of
  // i is directly bonded to j. Both rows retain LAMMPS's canonical topology.
  for (int ik = icount[1]; ik < icount[2]; ++ik)
    for (int jk = 0; jk < jcount[0]; ++jk)
      if (partners_[ioffset + ik] == partners_[joffset + jk]) return 4;
  return 0;
}

double FixChemtrainTopology::memory_usage()
{
  return static_cast<double>(counts_.capacity() * sizeof(std::array<int, 3>) +
                             partners_.capacity() * sizeof(tagint) +
                             communication_storage_.capacity() * sizeof(double) +
                             communication_rows_.capacity() * sizeof(double *));
}
