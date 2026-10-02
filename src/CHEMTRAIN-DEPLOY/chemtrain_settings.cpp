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

#include "chemtrain_settings.h"

#include "error.h"
#include "lammps.h"
#include "utils.h"

#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

using namespace LAMMPS_NS;

namespace {

bool chemtrain_setting_keyword(const char *value)
{
  return strcmp(value, "comm") == 0 || strcmp(value, "device") == 0 ||
      strcmp(value, "capacity/atom") == 0 ||
      strcmp(value, "capacity/edge") == 0 ||
      strcmp(value, "capacity/dense-neighbor") == 0 ||
      strcmp(value, "topology") == 0 || strcmp(value, "models") == 0 ||
      strcmp(value, "atom/input") == 0 ||
      strcmp(value, "global/input") == 0 || strcmp(value, "output") == 0;
}

}    // namespace

// ----------------------------------------------------------------------
// Pair-style settings
// ----------------------------------------------------------------------

ChemtrainSettings LAMMPS_NS::parse_chemtrain_settings(LAMMPS *lmp, int narg,
                                                       char **arg)
{
  if (narg < 1)
    lmp->error->all(FLERR, Error::ARGZERO,
                    "Illegal pair_style chemtrain command: missing backend");

  ChemtrainSettings settings;
  settings.backend = arg[0];

  int iarg = 1;
  bool saw_model_sensitive_setting = false;
  if (iarg < narg && utils::is_double(arg[iarg])) {
    settings.memory_fraction = utils::numeric(FLERR, arg[iarg], false, lmp);
    ++iarg;
  }

  while (iarg < narg) {
    if (strcmp(arg[iarg], "comm") == 0) {
      // Parse the runtime choice here; model compatibility is checked later.
      if (iarg + 1 >= narg)
        lmp->error->all(FLERR, iarg,
                        "Missing value for chemtrain comm keyword");
      settings.use_communication =
          utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "device") == 0) {
      lmp->error->all(FLERR, iarg,
                      "chemtrain no longer supports the device option. "
                      "Select GPUs with the runtime environment, MPI rank mapping, "
                      "and LAMMPS/Kokkos options");
    } else if (strcmp(arg[iarg], "capacity/atom") == 0) {
      if (iarg + 1 >= narg)
        lmp->error->all(FLERR, iarg,
                        "Missing value for chemtrain capacity/atom keyword");
      settings.capacity_atom_growth =
          utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (settings.capacity_atom_growth <= 1.0)
        lmp->error->all(FLERR, iarg + 1,
                        "chemtrain capacity/atom must be greater than 1.0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "capacity/edge") == 0) {
      if (iarg + 1 >= narg)
        lmp->error->all(FLERR, iarg,
                        "Missing value for chemtrain capacity/edge keyword");
      settings.capacity_edge_growth =
          utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (settings.capacity_edge_growth <= 1.0)
        lmp->error->all(FLERR, iarg + 1,
                        "chemtrain capacity/edge must be greater than 1.0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "capacity/dense-neighbor") == 0) {
      if (iarg + 1 >= narg)
        lmp->error->all(
            FLERR, iarg,
            "Missing value for chemtrain capacity/dense-neighbor keyword");
      settings.capacity_dense_neighbor_growth =
          utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (settings.capacity_dense_neighbor_growth <= 1.0)
        lmp->error->all(
            FLERR, iarg + 1,
            "chemtrain capacity/dense-neighbor must be greater than 1.0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "topology") == 0) {
      settings.install_topology_zero_styles = true;
      ++iarg;

    } else if (strcmp(arg[iarg], "models") == 0) {
      // Model contexts must be known before their inputs or outputs are parsed.
      if (saw_model_sensitive_setting)
        lmp->error->all(FLERR, iarg,
                        "chemtrain models must be declared before atom/input, "
                        "global/input, or output");
      ++iarg;
      if (iarg + 1 >= narg || chemtrain_setting_keyword(arg[iarg]))
        lmp->error->all(FLERR, iarg,
                        "The chemtrain models keyword requires name/scale pairs");
      std::set<std::string> names;
      while (iarg < narg && !chemtrain_setting_keyword(arg[iarg])) {
        if (iarg + 1 >= narg || chemtrain_setting_keyword(arg[iarg + 1]))
          lmp->error->all(FLERR, iarg,
                          "The chemtrain models keyword requires name/scale pairs");
        const std::string name = arg[iarg];
        if (!names.insert(name).second)
          lmp->error->all(FLERR, iarg,
                          "Duplicate chemtrain model context {}", name);
        ChemtrainSettings::ModelContextSpec model_context;
        model_context.name = name;
        model_context.scale = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        settings.model_contexts.push_back(model_context);
        iarg += 2;
      }

    } else if (strcmp(arg[iarg], "atom/input") == 0) {
      // Input mappings use either the implicit model or one named context.
      saw_model_sensitive_setting = true;
      if (iarg + 2 >= narg)
        lmp->error->all(FLERR, iarg,
                        "chemtrain atom/input requires a name and source");
      ChemtrainSettings::InputMapping request;
      if (settings.model_contexts.empty()) {
        request.name = arg[iarg + 1];
        request.source = arg[iarg + 2];
        iarg += 3;
      } else {
        if (iarg + 3 >= narg)
          lmp->error->all(FLERR, iarg,
                          "chemtrain model atom/input requires a model context, name, and source");
        request.model_context = arg[iarg + 1];
        request.name = arg[iarg + 2];
        request.source = arg[iarg + 3];
        bool known_model_context = false;
        for (const auto &model_context : settings.model_contexts)
          known_model_context |= model_context.name == request.model_context;
        if (!known_model_context)
          lmp->error->all(FLERR, iarg + 1,
                          "chemtrain atom/input references unknown model context {}",
                          request.model_context);
        iarg += 4;
      }
      for (const auto &mapping : settings.particle_inputs)
        if (mapping.model_context == request.model_context &&
            mapping.name == request.name)
          lmp->error->all(FLERR, iarg,
                          "Duplicate chemtrain atom/input mapping for {}", request.name);
      settings.particle_inputs.push_back(request);

    } else if (strcmp(arg[iarg], "global/input") == 0) {
      saw_model_sensitive_setting = true;
      if (iarg + 2 >= narg)
        lmp->error->all(FLERR, iarg,
                        "chemtrain global/input requires a name and source");
      ChemtrainSettings::InputMapping request;
      if (settings.model_contexts.empty()) {
        request.name = arg[iarg + 1];
        request.source = arg[iarg + 2];
        iarg += 3;
      } else {
        if (iarg + 3 >= narg)
          lmp->error->all(FLERR, iarg,
                          "chemtrain model global/input requires a model context, "
                          "name, and source");
        request.model_context = arg[iarg + 1];
        request.name = arg[iarg + 2];
        request.source = arg[iarg + 3];
        bool known_model_context = false;
        for (const auto &model_context : settings.model_contexts)
          known_model_context |= model_context.name == request.model_context;
        if (!known_model_context)
          lmp->error->all(FLERR, iarg + 1,
                          "chemtrain global/input references unknown model context {}",
                          request.model_context);
        iarg += 4;
      }
      for (const auto &mapping : settings.global_inputs)
        if (mapping.model_context == request.model_context &&
            mapping.name == request.name)
          lmp->error->all(FLERR, iarg,
                          "Duplicate chemtrain global/input mapping for {}", request.name);
      settings.global_inputs.push_back(request);

    } else if (strcmp(arg[iarg], "output") == 0) {
      // Requested auxiliary outputs use the same context syntax as inputs.
      saw_model_sensitive_setting = true;
      if (iarg + 1 >= narg)
        lmp->error->all(FLERR, iarg,
                        "chemtrain output requires an output name");
      ChemtrainSettings::OutputRequest request;
      if (settings.model_contexts.empty()) {
        request.name = arg[iarg + 1];
        iarg += 2;
      } else {
        if (iarg + 2 >= narg)
          lmp->error->all(FLERR, iarg,
                          "chemtrain model output requires a model context and output name");
        request.model_context = arg[iarg + 1];
        request.name = arg[iarg + 2];
        bool known_model_context = false;
        for (const auto &model_context : settings.model_contexts)
          known_model_context |= model_context.name == request.model_context;
        if (!known_model_context)
          lmp->error->all(FLERR, iarg + 1,
                          "chemtrain output references unknown model context {}",
                          request.model_context);
        iarg += 3;
      }
      for (const auto &output : settings.outputs)
        if (output.model_context == request.model_context &&
            output.name == request.name)
          lmp->error->all(FLERR, iarg,
                          "Duplicate chemtrain output request for {}", request.name);
      settings.outputs.push_back(request);
    } else {
      lmp->error->all(FLERR, iarg, "Unknown chemtrain setting {}", arg[iarg]);
    }
  }

  return settings;
}

// ----------------------------------------------------------------------
// Accelerator-process mapping
// ----------------------------------------------------------------------

int LAMMPS_NS::chemtrain_local_rank(LAMMPS *lmp)
{
  const char *names[] = {"SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK",
                         "MV2_COMM_WORLD_LOCAL_RANK", "FLUX_TASK_LOCAL_ID",
                         "PMI_LOCAL_RANK"};
  for (const char *name : names) {
    const char *value = getenv(name);
    if (value != nullptr) return utils::inumeric(FLERR, value, true, lmp);
  }
  return 0;
}
