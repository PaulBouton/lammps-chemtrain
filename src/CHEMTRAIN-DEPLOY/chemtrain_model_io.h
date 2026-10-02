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

#ifndef LMP_CHEMTRAIN_MODEL_IO_H
#define LMP_CHEMTRAIN_MODEL_IO_H

#include <vector>

namespace LAMMPS_NS {

class LAMMPS;

std::vector<char> read_chemtrain_model(LAMMPS *, const char *);

}    // namespace LAMMPS_NS

#endif
