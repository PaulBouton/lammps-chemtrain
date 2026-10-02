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

#include "pair_chemtrain_deploy.h"

#include "arg_info.h"
#include "atom.h"
#include "chemtrain_runtime.h"
#include "error.h"
#include "input.h"
#include "utils.h"
#include "variable.h"

#include <utility>

using namespace LAMMPS_NS;

// ----------------------------------------------------------------------
// Particle inputs
// ----------------------------------------------------------------------

void ChemtrainDeploy::resolve_particle_inputs(ChemtrainModelContext &model_context)
{
  auto &runtime = *model_context.runtime;
  model_context.particle_inputs.clear();
  runtime.particle_inputs.clear();

  // Validate configured names first so a misspelled field cannot silently use
  // a default source.
  for (const auto &mapping : configured_particle_inputs) {
    if (mapping.model_context != model_context.name) continue;
    bool declared = false;
    for (int i = 0; i < runtime.model_properties.num_particle_fields; ++i)
      declared |= mapping.name == runtime.model_properties.particle_fields[i].name;
    if (!declared)
      error->all(FLERR, "chemtrain atom/input names undeclared model field {}", mapping.name);
    if (mapping.name == "species")
      error->all(FLERR, "chemtrain species is always supplied from zero-based atom types");
  }

  // Species comes from the zero-based LAMMPS atom type. Other particle fields
  // default to scalar custom integer properties with matching names.
  for (int i = 0; i < runtime.model_properties.num_particle_fields; ++i) {
    const auto &descriptor = runtime.model_properties.particle_fields[i];
    if (descriptor.dtype != JCN_TENSOR_DTYPE_S32)
      error->all(FLERR, "chemtrain particle field {} requires unsupported dtype {}",
                 descriptor.name, static_cast<int>(descriptor.dtype));

    ChemtrainParticleInputBinding binding;
    binding.name = descriptor.name;
    for (const auto &mapping : configured_particle_inputs)
      if (mapping.model_context == model_context.name && mapping.name == binding.name)
        binding.source = mapping.source;
    if (binding.source.empty()) {
      if (binding.name == "species")
        binding.source = "type/zero-based";
      else
        binding.source = "i_" + binding.name;
    }

    if (binding.name == "species") {
      binding.source_kind = ChemtrainParticleSource::TYPE_ZERO_BASED;
    } else if (binding.source.rfind("i_", 0) == 0) {
      binding.source_kind = ChemtrainParticleSource::CUSTOM_INT;
      validate_custom_particle_input(binding);
    } else {
      error->all(FLERR, "chemtrain unsupported atom/input source {}", binding.source);
    }
    model_context.particle_inputs.push_back(std::move(binding));
  }
  runtime.particle_inputs.resize(model_context.particle_inputs.size());
}

// ----------------------------------------------------------------------
// Global inputs
// ----------------------------------------------------------------------

void ChemtrainDeploy::resolve_global_inputs(ChemtrainModelContext &model_context)
{
  auto &runtime = *model_context.runtime;
  model_context.global_inputs.clear();
  runtime.global_inputs.clear();

  for (const auto &mapping : configured_global_inputs) {
    if (mapping.model_context != model_context.name) continue;
    bool declared = false;
    for (int i = 0; i < runtime.model_properties.num_global_fields; ++i)
      declared |= mapping.name == runtime.model_properties.global_fields[i].name;
    if (!declared)
      error->all(FLERR, "chemtrain global/input names undeclared model field {}", mapping.name);
  }

  for (int i = 0; i < runtime.model_properties.num_global_fields; ++i) {
    const auto &descriptor = runtime.model_properties.global_fields[i];
    ChemtrainGlobalInputBinding binding;
    binding.name = descriptor.name;
    binding.dtype = descriptor.dtype;
    if (binding.dtype != JCN_TENSOR_DTYPE_F32 && binding.dtype != JCN_TENSOR_DTYPE_F64 &&
        binding.dtype != JCN_TENSOR_DTYPE_S32)
      error->all(FLERR, "chemtrain global field {} requires unsupported dtype {}", binding.name,
                 static_cast<int>(binding.dtype));
    for (const auto &mapping : configured_global_inputs)
      if (mapping.model_context == model_context.name && mapping.name == binding.name)
        binding.source = mapping.source;
    // A configuration-wide source cannot be inferred from the field name, so
    // every global field requires an explicit mapping.
    if (binding.source.empty())
      error->all(FLERR, "chemtrain global field {} requires an explicit global/input mapping",
                 binding.name);

    if (binding.source.rfind("v_", 0) == 0) {
      binding.source_kind = ChemtrainGlobalSource::EQUAL_VARIABLE;
      binding.variable_index = input->variable->find(binding.source.c_str() + 2);
      if (binding.variable_index < 0)
        error->all(FLERR, "Variable {} for chemtrain global/input does not exist", binding.source);
      if (!input->variable->equalstyle(binding.variable_index))
        error->all(FLERR, "Variable {} for chemtrain global/input must be equal-style",
                   binding.source);
    } else {
      binding.source_kind = ChemtrainGlobalSource::LITERAL;
      binding.double_value = utils::numeric(FLERR, binding.source, false, lmp);
    }
    model_context.global_inputs.push_back(std::move(binding));
  }
  runtime.global_inputs.resize(model_context.global_inputs.size());
}

// ----------------------------------------------------------------------
// Auxiliary outputs
// ----------------------------------------------------------------------

void ChemtrainDeploy::resolve_requested_outputs(ChemtrainModelContext &model_context)
{
  auto &runtime = *model_context.runtime;
  model_context.requested_outputs.clear();
  // Cache each output shape during pair_coeff. Compute commands inspect their
  // output shape while the input script is parsed, before force evaluation.
  for (const auto &request : configured_outputs) {
    if (request.model_context != model_context.name) continue;
    const auto &name = request.name;
    if (name == "F" || name == "U" || name == "V")
      error->all(FLERR,
                 "chemtrain output {} is built in and cannot be requested as an "
                 "auxiliary output",
                 name);

    bool declared = false;
    jcn_output_scope scope = JCN_OUTPUT_PARTICLE;
    bool extensive = false;
    std::vector<int64_t> dimensions;
    int components = 0;
    for (int i = 0; i < runtime.model_properties.num_outputs; ++i) {
      const auto &descriptor = runtime.model_properties.outputs[i];
      if (name == descriptor.name) {
        declared = true;
        scope = descriptor.scope;
        extensive = descriptor.extensive != 0;
        if (descriptor.rank > 0)
          dimensions.assign(descriptor.dimensions, descriptor.dimensions + descriptor.rank);
        components = descriptor.components;
      }
    }

    if (!declared) error->all(FLERR, "chemtrain output {} is not declared by the model", name);

    ChemtrainRequestedOutputBinding output;
    output.name = name;
    output.scope = scope;
    output.extensive = extensive;
    output.dimensions = std::move(dimensions);
    output.components = components;

    model_context.requested_outputs.push_back(std::move(output));
  }
}

// ----------------------------------------------------------------------
// Input validation and evaluation
// ----------------------------------------------------------------------

void ChemtrainDeploy::validate_custom_particle_input(ChemtrainParticleInputBinding &binding)
{
  const ArgInfo source(binding.source, ArgInfo::INAME);
  if (source.get_type() != ArgInfo::INAME || source.get_dim() != 0 ||
      binding.source.rfind("i_", 0) != 0)
    error->all(FLERR,
               "chemtrain atom/input source {} for field {} must name an "
               "unindexed scalar integer custom property (i_NAME)",
               binding.source, binding.name);

  int flag = -1;
  int cols = -1;
  int ghost = 0;
  const int index = atom->find_custom_ghost(source.get_name(), flag, cols, ghost);
  if (index < 0)
    error->all(FLERR,
               "chemtrain atom/input source {} for field {} does not exist or "
               "its fix property/atom is not active",
               binding.source, binding.name);
  if (flag != 0 || cols != 0)
    error->all(FLERR,
               "chemtrain atom/input source {} for field {} must be a scalar "
               "integer property created by fix property/atom",
               binding.source, binding.name);
  if (!ghost)
    error->all(FLERR,
               "chemtrain atom/input source {} for field {} requires "
               "fix property/atom ghost yes before pair_coeff",
               binding.source, binding.name);
  if (atom->ivector[index] == nullptr)
    error->all(FLERR,
               "chemtrain atom/input source {} for field {} is unavailable "
               "because its fix property/atom is no longer active",
               binding.source, binding.name);
  binding.custom_index = index;
}

void ChemtrainDeploy::evaluate_global_inputs(ChemtrainModelContext &model_context)
{
  for (auto &binding : model_context.global_inputs) {
    double value = binding.double_value;
    if (binding.source_kind == ChemtrainGlobalSource::EQUAL_VARIABLE)
      value = input->variable->compute_equal(binding.variable_index);
    if (binding.dtype == JCN_TENSOR_DTYPE_F32) {
      binding.float_value = static_cast<float>(value);
    } else if (binding.dtype == JCN_TENSOR_DTYPE_F64) {
      binding.double_value = value;
    } else if (binding.dtype == JCN_TENSOR_DTYPE_S32) {
      binding.int_value = static_cast<int32_t>(value);
    } else {
      error->all(FLERR, "chemtrain global field {} has unsupported dtype {}", binding.name,
                 static_cast<int>(binding.dtype));
    }
  }
}

// ----------------------------------------------------------------------
// Model-context output lookup
// ----------------------------------------------------------------------

bool ChemtrainDeploy::multiple_model_contexts() const
{
  return model_contexts.size() > 1 || (!model_contexts.empty() && !model_contexts[0]->name.empty());
}

ChemtrainModelContext &ChemtrainDeploy::model_context_for_name(const std::string &name)
{
  if (name.empty()) {
    if (multiple_model_contexts())
      error->all(
          FLERR,
          "compute chemtrain/output requires a model context name when multiple models are active");
    if (model_contexts.empty()) error->all(FLERR, "chemtrain has no loaded model context");
    return *model_contexts.front();
  }
  for (auto &model_context : model_contexts)
    if (model_context->name == name) return *model_context;
  error->all(FLERR, "chemtrain has no model context named {}", name);
  return *model_contexts.front();
}

const ChemtrainModelContext &ChemtrainDeploy::model_context_for_name(const std::string &name) const
{
  if (name.empty()) {
    if (multiple_model_contexts())
      error->all(
          FLERR,
          "compute chemtrain/output requires a model context name when multiple models are active");
    if (model_contexts.empty()) error->all(FLERR, "chemtrain has no loaded model context");
    return *model_contexts.front();
  }
  for (const auto &model_context : model_contexts)
    if (model_context->name == name) return *model_context;
  error->all(FLERR, "chemtrain has no model context named {}", name);
  return *model_contexts.front();
}

const double *ChemtrainDeploy::chemtrain_output(const std::string &name, int &components,
                                                bigint &timestep)
{
  return chemtrain_output("", name, components, timestep);
}

const double *ChemtrainDeploy::chemtrain_output(const std::string &model_context,
                                                const std::string &name, int &components,
                                                bigint &timestep)
{
  auto &model = model_context_for_name(model_context);
  for (auto &output : model.requested_outputs) {
    if (output.name == name) {
      components = output.components;
      timestep = output.host_timestep;
      return output.host_values.data();
    }
  }
  components = 0;
  timestep = -1;
  return nullptr;
}

int ChemtrainDeploy::chemtrain_output_components(const std::string &name, jcn_output_scope &scope,
                                                 bool &extensive) const
{
  return chemtrain_output_components("", name, scope, extensive);
}

int ChemtrainDeploy::chemtrain_output_components(const std::string &model_context,
                                                 const std::string &name, jcn_output_scope &scope,
                                                 bool &extensive) const
{
  const auto &model = model_context_for_name(model_context);
  if (model.runtime == nullptr || model.runtime->model == nullptr)
    error->all(FLERR,
               "Compute chemtrain/output must be defined after pair_coeff "
               "loads the chemtrain model");
  if (name == "F" || name == "U" || name == "V")
    error->all(FLERR,
               "chemtrain output {} is built in and cannot be requested as an "
               "auxiliary output",
               name);

  for (const auto &output : model.requested_outputs) {
    if (name == output.name) {
      if (output.components < 1)
        error->all(FLERR, "chemtrain output {} has invalid component count {}", name,
                   output.components);
      scope = output.scope;
      extensive = output.extensive;
      return output.components;
    }
  }
  error->all(FLERR, "chemtrain output {} was not requested on the pair style", name);
  return 0;
}
