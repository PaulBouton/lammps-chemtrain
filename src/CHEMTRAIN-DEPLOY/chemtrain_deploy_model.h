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

#ifndef LMP_CHEMTRAIN_DEPLOY_MODEL_H
#define LMP_CHEMTRAIN_DEPLOY_MODEL_H

#include "connector/jcn_api.h"
#include "lmptype.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class ChemtrainNeighbor;
struct ChemtrainModelRuntime;

enum class ChemtrainParticleSource { TYPE_ZERO_BASED, CUSTOM_INT };
enum class ChemtrainGlobalSource { LITERAL, EQUAL_VARIABLE };

struct ChemtrainParticleInputBinding {
  ChemtrainParticleInputBinding() :
      source_kind(ChemtrainParticleSource::CUSTOM_INT), custom_index(-1), cached_buffer(nullptr)
  {
  }

  std::string name;
  std::string source;
  ChemtrainParticleSource source_kind;
  int custom_index;
  std::vector<int32_t> host_values;
  jcn_buffer *cached_buffer;
};

struct ChemtrainGlobalInputBinding {
  ChemtrainGlobalInputBinding() :
      dtype(JCN_TENSOR_DTYPE_MODEL_DEFAULT), source_kind(ChemtrainGlobalSource::LITERAL),
      variable_index(-1), double_value(0.0), float_value(0.0f), int_value(0), cached_buffer(nullptr)
  {
  }

  std::string name;
  std::string source;
  jcn_tensor_dtype dtype;
  ChemtrainGlobalSource source_kind;
  int variable_index;
  double double_value;
  float float_value;
  int32_t int_value;
  jcn_buffer *cached_buffer;
};

struct ChemtrainRequestedOutputBinding {
  ChemtrainRequestedOutputBinding() :
      scope(JCN_OUTPUT_PARTICLE), extensive(false), components(1), cached_buffer(nullptr),
      host_timestep(-1), device_timestep(-1)
  {
  }

  std::string name;
  jcn_output_scope scope;
  bool extensive;
  // Declared dimensions of one output value. PARTICLE storage adds the
  // atom-capacity axis separately.
  std::vector<int64_t> dimensions;
  int components;

  std::vector<double> host_values;
  jcn_buffer *cached_buffer;

  bigint host_timestep;
  bigint device_timestep;
};

struct ChemtrainModelContext {
  std::string name;
  std::string filename;
  double scale = 1.0;
  std::unique_ptr<ChemtrainModelRuntime> runtime;

  std::vector<ChemtrainParticleInputBinding> particle_inputs;
  std::vector<ChemtrainGlobalInputBinding> global_inputs;
  std::vector<ChemtrainRequestedOutputBinding> requested_outputs;

  std::vector<double> host_positions;
  std::vector<double> host_forces;
  std::vector<double> host_energy;
  std::vector<double> host_virial;
  std::unique_ptr<ChemtrainNeighbor> model_neighbors;
};

}    // namespace LAMMPS_NS

#endif
