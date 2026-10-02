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

#include "chemtrain_model_io.h"
#include "chemtrain_settings.h"
#include "platform.h"

#include "../testing/core.h"
#include "../testing/test_mpi_main.h"
#include "gtest/gtest.h"

#include <cstdio>
#include <mpi.h>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class ChemtrainSettingsTest : public LAMMPSTest {
 protected:
  ChemtrainSettings parse(std::initializer_list<const char *> values)
  {
    std::vector<std::string> argument_storage(values.begin(), values.end());
    std::vector<char *> arguments;
    arguments.reserve(argument_storage.size());
    for (auto &value : argument_storage) arguments.push_back(value.data());
    return parse_chemtrain_settings(lmp, static_cast<int>(arguments.size()),
                                    arguments.data());
  }
};

// ----------------------------------------------------------------------
// Pair-style setting parser
// ----------------------------------------------------------------------

TEST_F(ChemtrainSettingsTest, ParsesCompleteSettings)
{
  const auto settings =
      parse({"cuda", "0.9", "comm", "on", "capacity/atom", "1.3",
             "capacity/edge", "1.6", "capacity/dense-neighbor", "1.4",
             "topology", "atom/input", "residue_id", "i_residue"});

  EXPECT_EQ(settings.backend, "cuda");
  EXPECT_DOUBLE_EQ(settings.memory_fraction, 0.9);
  EXPECT_TRUE(settings.use_communication);
  EXPECT_TRUE(settings.install_topology_zero_styles);
  EXPECT_DOUBLE_EQ(settings.capacity_atom_growth, 1.3);
  EXPECT_DOUBLE_EQ(settings.capacity_edge_growth, 1.6);
  EXPECT_DOUBLE_EQ(settings.capacity_dense_neighbor_growth, 1.4);
  ASSERT_EQ(settings.particle_inputs.size(), 1);
  EXPECT_EQ(settings.particle_inputs[0].name, "residue_id");
  EXPECT_EQ(settings.particle_inputs[0].source, "i_residue");
}

TEST_F(ChemtrainSettingsTest, UsesLAMMPSLogicalValues)
{
  for (const char *value : {"yes", "true", "on", "1"})
    EXPECT_TRUE(parse({"cpu", "comm", value}).use_communication);
  for (const char *value : {"no", "false", "off", "0"})
    EXPECT_FALSE(parse({"cpu", "comm", value}).use_communication);
}

TEST_F(ChemtrainSettingsTest, RejectsMalformedOptions)
{
  TEST_FAILURE(".*missing backend.*", parse_chemtrain_settings(lmp, 0, nullptr););

  TEST_FAILURE(".*Missing value.*comm.*", parse({"cpu", "comm"}););
  TEST_FAILURE(".*Expected floating point parameter instead of '1.2x'.*",
               parse({"cpu", "capacity/atom", "1.2x"}););
  TEST_FAILURE(".*capacity/edge must be greater than 1.0.*",
               parse({"cpu", "capacity/edge", "1.0"}););
  TEST_FAILURE(".*Duplicate.*residue_id.*",
               parse({"cpu", "atom/input", "residue_id", "i_a", "atom/input",
                      "residue_id", "i_b"}););
  TEST_FAILURE(".*Unknown chemtrain setting.*mystery.*",
               parse({"cpu", "mystery"}););
}

TEST_F(ChemtrainSettingsTest, ReplacesSettingsWithoutRetainingOldOptions)
{
  HIDE_OUTPUT([&] {
    command("pair_style chemtrain cpu comm on topology atom/input field i_old");
    command("pair_style chemtrain cpu comm off");
  });
}

// ----------------------------------------------------------------------
// Collective model-file input
// ----------------------------------------------------------------------

TEST_F(ChemtrainSettingsTest, ReadsBinaryModelCollectively)
{
  constexpr const char *FILENAME = "chemtrain-model-io-test.bin";
  const std::vector<char> expected = {'J', 'C', 'N', '\0', 'x', 'y'};
  int me;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  if (me == 0) {
    FILE *file = fopen(FILENAME, "wb");
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(fwrite(expected.data(), sizeof(char), expected.size(), file),
              expected.size());
    fclose(file);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(read_chemtrain_model(lmp, FILENAME), expected);

  MPI_Barrier(MPI_COMM_WORLD);
  if (me == 0) platform::unlink(FILENAME);
}

}    // namespace LAMMPS_NS
