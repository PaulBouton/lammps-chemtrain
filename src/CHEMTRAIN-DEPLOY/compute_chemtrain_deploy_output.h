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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(chemtrain/output,ComputeChemtrainDeployOutput);
// clang-format on
#else

#ifndef LMP_COMPUTE_CHEMTRAIN_DEPLOY_OUTPUT_H
#define LMP_COMPUTE_CHEMTRAIN_DEPLOY_OUTPUT_H

#include "compute.h"
#include "connector/jcn_api.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

class ChemtrainDeploy;

class ComputeChemtrainDeployOutput : public Compute {
 public:
  ComputeChemtrainDeployOutput(class LAMMPS *, int, char **);
  ~ComputeChemtrainDeployOutput() override;

  void init() override;
  void compute_peratom() override;

  double compute_scalar() override;
  void compute_vector() override;

 private:
  std::string model_context_name;
  std::string output_name;
  ChemtrainDeploy *pair;
  jcn_output_scope scope;
  bool extensive;

  int components;
  int nmax;
  std::vector<double> values;
};

}    // namespace LAMMPS_NS

#endif
#endif
