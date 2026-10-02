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

#if !defined(LMP_KOKKOS)
#else

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(chemtrain/output/kk,ComputeChemtrainDeployOutputKokkos<LMPDeviceType>);
ComputeStyle(chemtrain/output/kk/device,ComputeChemtrainDeployOutputKokkos<LMPDeviceType>);
// clang-format on
#else

#ifndef LMP_COMPUTE_CHEMTRAIN_DEPLOY_OUTPUT_KOKKOS_H
#define LMP_COMPUTE_CHEMTRAIN_DEPLOY_OUTPUT_KOKKOS_H

#include "compute.h"
#include "connector/jcn_api.h"
#include "kokkos_type.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

template <class DeviceType> class ChemtrainDeployKokkos;

template <class DeviceType> class ComputeChemtrainDeployOutputKokkos : public Compute {
 public:
  using device_type = DeviceType;
  using AT = ArrayTypes<DeviceType>;

  ComputeChemtrainDeployOutputKokkos(class LAMMPS *, int, char **);
  ~ComputeChemtrainDeployOutputKokkos() override;

  void init() override;
  void compute_peratom() override;

  double compute_scalar() override;
  void compute_vector() override;

 private:
  std::string model_context_name;
  std::string output_name;
  ChemtrainDeployKokkos<LMPDeviceType> *pair;
  jcn_output_scope scope;
  bool extensive;

  int components;
  int nmax;
  std::vector<double> values;

  DAT::ttransform_kkfloat_1d k_vector_atom;
  typename AT::t_kkfloat_1d d_vector_atom;
  DAT::ttransform_kkfloat_2d k_array_atom;
  typename AT::t_kkfloat_2d d_array_atom;
};

}    // namespace LAMMPS_NS

#endif
#endif
#endif
