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

#ifdef FIX_CLASS
// clang-format off
FixStyle(CHEMTRAIN/TOPOLOGY,FixChemtrainTopology);
// clang-format on
#else

#ifndef LMP_FIX_CHEMTRAIN_TOPOLOGY_H
#define LMP_FIX_CHEMTRAIN_TOPOLOGY_H

#include "fix.h"
#include "lmptype.h"

#include <array>
#include <vector>

namespace LAMMPS_NS {

class FixChemtrainTopology : public Fix {
 public:
  FixChemtrainTopology(class LAMMPS *, int, char **);

  int setmask() override;
  void setup_pre_neighbor() override;
  void pre_neighbor() override;
  void min_pre_neighbor() override;

  int category(int, tagint) const;
  int onefive(int, int) const;
  int generation() const { return generation_; }
  int maxspecial() const { return maxspecial_; }
  const std::vector<std::array<int, 3>> &counts() const { return counts_; }
  const std::vector<tagint> &partners() const { return partners_; }
  double memory_usage() override;

 protected:
  int generation_ = 0;
  int maxspecial_ = 0;
  std::vector<std::array<int, 3>> counts_;
  std::vector<tagint> partners_;
  std::vector<double> communication_storage_;
  std::vector<double *> communication_rows_;

  virtual void refresh();
  void snapshot_owned();
  void forward_host();
};

}    // namespace LAMMPS_NS

#endif
#endif
