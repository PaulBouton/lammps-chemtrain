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

#include "compute_chemtrain_deploy_output.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "pair_chemtrain_deploy.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <mpi.h>

using namespace LAMMPS_NS;

ComputeChemtrainDeployOutput::ComputeChemtrainDeployOutput(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), pair(nullptr), scope(JCN_OUTPUT_PARTICLE), extensive(false),
    components(0), nmax(0)
{
  if (narg != 4 && narg != 5) utils::missing_cmd_args(FLERR, "compute chemtrain/output", error);
  if (narg == 4) {
    output_name = arg[3];
  } else {
    model_context_name = arg[3];
    output_name = arg[4];
  }
  pair = dynamic_cast<ChemtrainDeploy *>(force->pair);
  if (pair == nullptr) pair = dynamic_cast<ChemtrainDeploy *>(force->pair_match("chemtrain", 0));
  if (pair == nullptr)
    error->all(FLERR, "Compute chemtrain/output requires active chemtrain pair style");
  // Commands such as compute reduce inspect the output shape during parsing,
  // before Compute::init() is called. The pair_coeff command must therefore
  // expose model metadata before compute chemtrain/output is defined.
  components = pair->chemtrain_output_components(model_context_name, output_name, scope, extensive);
  if (scope == JCN_OUTPUT_PARTICLE) {
    peratom_flag = 1;
    size_peratom_cols = components == 1 ? 0 : components;
  } else {
    if (igroup) error->all(FLERR, 1, "Compute chemtrain/output must use group all");
    if (components == 1) {
      scalar_flag = 1;
      extscalar = extensive ? 1 : 0;
    } else {
      vector_flag = 1;
      size_vector = components;
      extvector = extensive ? 1 : 0;
      values.resize(components);
      vector = values.data();
    }
  }
}

ComputeChemtrainDeployOutput::~ComputeChemtrainDeployOutput()
{
  if (copymode) return;
  memory->destroy(vector_atom);
  memory->destroy(array_atom);
}

void ComputeChemtrainDeployOutput::init()
{
  pair = dynamic_cast<ChemtrainDeploy *>(force->pair);
  if (pair == nullptr) pair = dynamic_cast<ChemtrainDeploy *>(force->pair_match("chemtrain", 0));
  if (pair == nullptr)
    error->all(FLERR, "Compute chemtrain/output requires active chemtrain pair style");

  jcn_output_scope current_scope = JCN_OUTPUT_PARTICLE;
  bool current_extensive = false;
  const int current_components = pair->chemtrain_output_components(
      model_context_name, output_name, current_scope, current_extensive);
  if (current_components != components || current_scope != scope ||
      current_extensive != extensive) {
    error->all(FLERR, "Compute chemtrain/output {} changed shape", output_name);
  }
}

void ComputeChemtrainDeployOutput::compute_peratom()
{
  if (scope != JCN_OUTPUT_PARTICLE)
    error->all(FLERR, "Compute chemtrain/output is not a particle output");
  invoked_peratom = update->ntimestep;
  if (atom->nmax > nmax) {
    if (components == 1) {
      memory->destroy(vector_atom);
      memory->create(vector_atom, atom->nmax, "chemtrain/output:vector_atom");
    } else {
      memory->destroy(array_atom);
      memory->create(array_atom, atom->nmax, components, "chemtrain/output:array_atom");
    }
    nmax = atom->nmax;
  }

  bigint timestep = -1;
  int current_components = 0;
  const double *data =
      pair->chemtrain_output(model_context_name, output_name, current_components, timestep);
  if (data == nullptr || current_components != components)
    error->all(FLERR, "Compute chemtrain/output {} changed shape", output_name);
  if (timestep != update->ntimestep)
    error->all(FLERR,
               "Compute chemtrain/output {} was requested before current "
               "chemtrain force evaluation",
               output_name);

  // Pair output covers every owned atom. LAMMPS compute groups require zero
  // values for atoms outside the selected group.
  int *mask = atom->mask;
  if (components == 1) {
    for (int i = 0; i < atom->nlocal; ++i) vector_atom[i] = mask[i] & groupbit ? data[i] : 0.0;
  } else {
    for (int i = 0; i < atom->nlocal; ++i) {
      if (mask[i] & groupbit) {
        for (int j = 0; j < components; ++j)
          array_atom[i][j] = data[static_cast<size_t>(i) * components + j];
      } else {
        std::fill_n(array_atom[i], components, 0.0);
      }
    }
  }
}

double ComputeChemtrainDeployOutput::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  bigint timestep = -1;
  int current_components = 0;
  const double *data =
      pair->chemtrain_output(model_context_name, output_name, current_components, timestep);
  if (data == nullptr || current_components != 1 || timestep != update->ntimestep)
    error->all(FLERR, "Compute chemtrain/output {} is unavailable for the current timestep",
               output_name);
  // LOCAL contains an additive contribution from the current MPI rank, so the
  // compute applies exactly one MPI sum. GLOBAL already contains the complete
  // configuration value and is copied without another reduction.
  if (scope == JCN_OUTPUT_LOCAL) {
    if (MPI_Allreduce(data, &scalar, 1, MPI_DOUBLE, MPI_SUM, world) != MPI_SUCCESS)
      error->all(FLERR, "Compute chemtrain/output MPI_Allreduce failed");
  } else {
    scalar = data[0];
  }
  return scalar;
}

void ComputeChemtrainDeployOutput::compute_vector()
{
  invoked_vector = update->ntimestep;
  bigint timestep = -1;
  int current_components = 0;
  const double *data =
      pair->chemtrain_output(model_context_name, output_name, current_components, timestep);
  if (data == nullptr || current_components != components || timestep != update->ntimestep)
    error->all(FLERR, "Compute chemtrain/output {} is unavailable for the current timestep",
               output_name);
  // LOCAL contains an additive contribution from the current MPI rank, so the
  // compute applies exactly one MPI sum. GLOBAL already contains the complete
  // configuration value and is copied without another reduction.
  if (scope == JCN_OUTPUT_LOCAL) {
    if (MPI_Allreduce(data, values.data(), components, MPI_DOUBLE, MPI_SUM, world) != MPI_SUCCESS)
      error->all(FLERR, "Compute chemtrain/output MPI_Allreduce failed");
  } else {
    std::copy(data, data + components, values.begin());
  }
}
