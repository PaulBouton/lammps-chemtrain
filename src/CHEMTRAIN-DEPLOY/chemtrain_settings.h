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

#ifndef LMP_CHEMTRAIN_SETTINGS_H
#define LMP_CHEMTRAIN_SETTINGS_H

#include <string>
#include <utility>
#include <vector>

namespace LAMMPS_NS {

class LAMMPS;

struct ChemtrainSettings {
  struct ModelContextSpec {
    // A named model context maps one pair_coeff model file to a force scale.
    // Multiple contexts share one PJRT client and one LAMMPS pair style.
    std::string name;
    double scale = 1.0;
  };

  struct OutputRequest {
    std::string model_context;
    std::string name;
  };

  struct InputMapping {
    std::string model_context;
    std::string name;
    std::string source;
  };

  std::string backend;
  double memory_fraction = 0.75;
  double capacity_atom_growth = 1.20;
  double capacity_edge_growth = 1.50;
  double capacity_dense_neighbor_growth = 1.25;
  bool use_communication = false;
  bool install_topology_zero_styles = false;
  std::vector<ModelContextSpec> model_contexts;
  std::vector<InputMapping> particle_inputs;
  std::vector<InputMapping> global_inputs;
  std::vector<OutputRequest> outputs;
};

ChemtrainSettings parse_chemtrain_settings(LAMMPS *, int, char **);
// Returns the node-local MPI rank used to assign accelerators.
int chemtrain_local_rank(LAMMPS *);

}    // namespace LAMMPS_NS

#endif
