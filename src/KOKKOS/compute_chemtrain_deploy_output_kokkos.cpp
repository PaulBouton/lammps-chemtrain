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

#include "compute_chemtrain_deploy_output_kokkos.h"

#if defined(LMP_KOKKOS)

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "pair_chemtrain_deploy_kokkos.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <mpi.h>
#include <type_traits>

using namespace LAMMPS_NS;

template <class DeviceType>
ComputeChemtrainDeployOutputKokkos<DeviceType>::ComputeChemtrainDeployOutputKokkos(LAMMPS *lmp,
                                                                                   int narg,
                                                                                   char **arg) :
    Compute(lmp, narg, arg), pair(nullptr), scope(JCN_OUTPUT_PARTICLE), extensive(false),
    components(0), nmax(0)
{
  if (narg != 4 && narg != 5) utils::missing_cmd_args(FLERR, "compute chemtrain/output/kk", error);
  if (narg == 4) {
    output_name = arg[3];
  } else {
    model_context_name = arg[3];
    output_name = arg[4];
  }
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = MASK_MASK;
  datamask_modify = EMPTY_MASK;
  pair = dynamic_cast<ChemtrainDeployKokkos<LMPDeviceType> *>(force->pair);
  if (pair == nullptr)
    pair =
        dynamic_cast<ChemtrainDeployKokkos<LMPDeviceType> *>(force->pair_match("chemtrain/kk", 0));
  if (pair == nullptr)
    error->all(FLERR, "Compute chemtrain/output/kk requires active chemtrain/kk pair style");
  // Commands such as compute reduce inspect the output shape during parsing,
  // before Compute::init() is called. The pair_coeff command must therefore
  // expose model metadata before compute chemtrain/output is defined.
  components = pair->chemtrain_output_components(model_context_name, output_name, scope, extensive);
  if (scope == JCN_OUTPUT_PARTICLE) {
    peratom_flag = 1;
    size_peratom_cols = components == 1 ? 0 : components;
  } else {
    if (igroup) error->all(FLERR, 1, "Compute chemtrain/output/kk must use group all");
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

template <class DeviceType>
ComputeChemtrainDeployOutputKokkos<DeviceType>::~ComputeChemtrainDeployOutputKokkos()
{
  if (copymode) return;
  memoryKK->destroy_kokkos(k_vector_atom, vector_atom);
  memoryKK->destroy_kokkos(k_array_atom, array_atom);
}

template <class DeviceType> void ComputeChemtrainDeployOutputKokkos<DeviceType>::init()
{
  pair = dynamic_cast<ChemtrainDeployKokkos<LMPDeviceType> *>(force->pair);
  if (pair == nullptr)
    pair =
        dynamic_cast<ChemtrainDeployKokkos<LMPDeviceType> *>(force->pair_match("chemtrain/kk", 0));
  if (pair == nullptr)
    error->all(FLERR, "Compute chemtrain/output/kk requires active chemtrain/kk pair style");

  jcn_output_scope current_scope = JCN_OUTPUT_PARTICLE;
  bool current_extensive = false;
  const int current_components = pair->chemtrain_output_components(
      model_context_name, output_name, current_scope, current_extensive);
  if (current_components != components || current_scope != scope ||
      current_extensive != extensive) {
    error->all(FLERR, "Compute chemtrain/output/kk {} changed shape", output_name);
  }
}

template <class DeviceType> void ComputeChemtrainDeployOutputKokkos<DeviceType>::compute_peratom()
{
  if (scope != JCN_OUTPUT_PARTICLE)
    error->all(FLERR, "Compute chemtrain/output/kk is not a particle output");
  invoked_peratom = update->ntimestep;
  // Group filtering reads the atom mask in the selected execution space.
  atomKK->sync(execution_space, datamask_read);

  if (atom->nmax > nmax) {
    if (components == 1) {
      memoryKK->destroy_kokkos(k_vector_atom, vector_atom);
      memoryKK->create_kokkos(k_vector_atom, vector_atom, atom->nmax,
                              "chemtrain/output/kk:vector_atom");
      d_vector_atom = k_vector_atom.template view<DeviceType>();
    } else {
      memoryKK->destroy_kokkos(k_array_atom, array_atom);
      memoryKK->create_kokkos(k_array_atom, array_atom, atom->nmax, components,
                              "chemtrain/output/kk:array_atom");
      d_array_atom = k_array_atom.template view<DeviceType>();
    }
    nmax = atom->nmax;
  }

  bigint timestep = -1;
  int current_components = 0;

  if constexpr (std::is_same_v<DeviceType, LMPHostType>) {
    const double *data =
        pair->chemtrain_output(model_context_name, output_name, current_components, timestep);
    if (data == nullptr || current_components != components)
      error->all(FLERR, "Compute chemtrain/output/kk {} changed shape", output_name);
    if (timestep != update->ntimestep)
      error->all(FLERR,
                 "Compute chemtrain/output/kk {} was requested before current "
                 "chemtrain force evaluation",
                 output_name);
    // Pair output covers every owned atom. LAMMPS compute groups require zero
    // values for atoms outside the selected group.
    int *mask = atom->mask;
    if (components == 1) {
      for (int i = 0; i < atom->nlocal; ++i) vector_atom[i] = mask[i] & groupbit ? data[i] : 0.0;
      k_vector_atom.modify_host();
    } else {
      for (int i = 0; i < atom->nlocal; ++i) {
        if (mask[i] & groupbit) {
          for (int j = 0; j < components; ++j)
            array_atom[i][j] = data[static_cast<size_t>(i) * components + j];
        } else {
          std::fill_n(array_atom[i], components, 0.0);
        }
      }
      k_array_atom.modify_host();
    }
  } else {
    const auto output = pair->chemtrain_output_device(model_context_name, output_name,
                                                      current_components, timestep);
    if (current_components != components)
      error->all(FLERR, "Compute chemtrain/output/kk {} changed shape", output_name);
    if (timestep != update->ntimestep)
      error->all(FLERR,
                 "Compute chemtrain/output/kk {} was requested before current "
                 "chemtrain force evaluation",
                 output_name);
    const int nlocal = atom->nlocal;
    const int selected_group = groupbit;
    auto mask = atomKK->k_mask.template view<DeviceType>();
    // Apply the same LAMMPS group semantics without copying model output to the
    // host first.
    if (components == 1) {
      auto values = d_vector_atom;
      Kokkos::parallel_for(
          Kokkos::RangePolicy<DeviceType>(0, nlocal),
          KOKKOS_LAMBDA(int i) { values(i) = mask(i) & selected_group ? output(i) : 0.0; });
      k_vector_atom.template modify<DeviceType>();
      k_vector_atom.sync_host();
    } else {
      auto values = d_array_atom;
      const int ncomponents = components;
      Kokkos::parallel_for(
          Kokkos::RangePolicy<DeviceType>(0, nlocal), KOKKOS_LAMBDA(int i) {
            if (mask(i) & selected_group) {
              for (int j = 0; j < ncomponents; ++j) values(i, j) = output(i * ncomponents + j);
            } else {
              for (int j = 0; j < ncomponents; ++j) values(i, j) = 0.0;
            }
          });
      k_array_atom.template modify<DeviceType>();
      k_array_atom.sync_host();
    }
  }
}

template <class DeviceType> double ComputeChemtrainDeployOutputKokkos<DeviceType>::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  bigint timestep = -1;
  int current_components = 0;
  const double *data =
      pair->chemtrain_output(model_context_name, output_name, current_components, timestep);
  if (data == nullptr || current_components != 1 || timestep != update->ntimestep)
    error->all(FLERR, "Compute chemtrain/output/kk {} is unavailable for the current timestep",
               output_name);
  // LOCAL contains an additive contribution from the current MPI rank, so the
  // compute applies exactly one MPI sum. GLOBAL already contains the complete
  // configuration value and is copied without another reduction.
  if (scope == JCN_OUTPUT_LOCAL) {
    if (MPI_Allreduce(data, &scalar, 1, MPI_DOUBLE, MPI_SUM, world) != MPI_SUCCESS)
      error->all(FLERR, "Compute chemtrain/output/kk MPI_Allreduce failed");
  } else {
    scalar = data[0];
  }
  return scalar;
}

template <class DeviceType> void ComputeChemtrainDeployOutputKokkos<DeviceType>::compute_vector()
{
  invoked_vector = update->ntimestep;
  bigint timestep = -1;
  int current_components = 0;
  const double *data =
      pair->chemtrain_output(model_context_name, output_name, current_components, timestep);
  if (data == nullptr || current_components != components || timestep != update->ntimestep)
    error->all(FLERR, "Compute chemtrain/output/kk {} is unavailable for the current timestep",
               output_name);
  // LOCAL contains an additive contribution from the current MPI rank, so the
  // compute applies exactly one MPI sum. GLOBAL already contains the complete
  // configuration value and is copied without another reduction.
  if (scope == JCN_OUTPUT_LOCAL) {
    if (MPI_Allreduce(data, values.data(), components, MPI_DOUBLE, MPI_SUM, world) != MPI_SUCCESS)
      error->all(FLERR, "Compute chemtrain/output/kk MPI_Allreduce failed");
  } else {
    std::copy(data, data + components, values.begin());
  }
}

namespace LAMMPS_NS {
template class ComputeChemtrainDeployOutputKokkos<LMPDeviceType>;
}

#endif
