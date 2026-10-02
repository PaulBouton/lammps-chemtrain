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

#ifndef LMP_CHEMTRAIN_RUNTIME_H
#define LMP_CHEMTRAIN_RUNTIME_H

#include "connector/jcn_api.h"

#include <cstdint>
#include <vector>

namespace LAMMPS_NS {

struct ChemtrainExecutionContext {
  // The API table is borrowed from the connector. The client is shared by all
  // model contexts owned by one pair style.
  const JCN_Api *api = nullptr;
  jcn_client *client = nullptr;
};

struct ChemtrainModelRuntime {
  // The pair style destroys these connector handles in dependency order.
  jcn_model *model = nullptr;
  jcn_executor *executor = nullptr;
  jcn_model_properties model_properties{};
  jcn_requested_capacities capacities{};

  std::vector<jcn_named_input> particle_inputs;
  std::vector<jcn_named_input> global_inputs;
  std::vector<jcn_named_output> outputs;

  jcn_buffer *positions_buffer = nullptr;
  jcn_buffer *forces_buffer = nullptr;
  jcn_buffer *energy_buffer = nullptr;
  jcn_buffer *virial_buffer = nullptr;

  int64_t atom_capacity = 0;
};

}    // namespace LAMMPS_NS

#endif
