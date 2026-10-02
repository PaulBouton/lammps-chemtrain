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

#include "chemtrain_neighbor.h"
#include "chemtrain_runtime.h"
#include "comm.h"
#include "compute.h"
#include "force.h"
#include "modify.h"
#include "pair_chemtrain_deploy.h"
#include "update.h"
#include "utils.h"

#if defined(LMP_TEST_KOKKOS_OUTPUT)
#include "library.h"
#include "pair_chemtrain_deploy_kokkos.h"
#endif

#include "../testing/core.h"
#include "../testing/test_mpi_main.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace LAMMPS_NS {

template <class PairType> class ChemtrainOutputPairProbe : public PairType {
public:
    explicit ChemtrainOutputPairProbe(LAMMPS *lmp) : PairType(lmp) {}

    void add_output(const std::string &name, int components,
                    jcn_output_scope scope = JCN_OUTPUT_PARTICLE, bool extensive = false)
    {
        ensure_context(true);
        ChemtrainRequestedOutputBinding output;
        output.name       = name;
        output.scope      = scope;
        output.extensive  = extensive;
        output.components = components;
        this->model_contexts.front()->requested_outputs.push_back(std::move(output));
    }

    void add_unloaded_context() { ensure_context(false); }

    void set_output(const std::string &name, std::vector<double> values, bigint timestep = 0)
    {
        for (auto &output : this->model_contexts.front()->requested_outputs) {
            if (output.name != name) continue;
            output.host_values   = std::move(values);
            output.host_timestep = timestep;
            return;
        }
    }

    std::pair<bigint, bigint> mark_current_and_sync_output(const std::string &name, bigint timestep)
    {
        for (auto &output : this->model_contexts.front()->requested_outputs) {
            if (output.name != name) continue;
            output.host_timestep   = timestep;
            output.device_timestep = timestep;
            this->sync_requested_outputs_to_host(*this->model_contexts.front());
            return {output.host_timestep, output.device_timestep};
        }
        return {-1, -1};
    }

private:
    void ensure_context(bool loaded)
    {
        if (this->model_contexts.empty()) {
            auto context     = std::make_unique<ChemtrainModelContext>();
            context->runtime = std::make_unique<ChemtrainModelRuntime>();
            this->model_contexts.push_back(std::move(context));
        }
        if (loaded)
            this->model_contexts.front()->runtime->model = reinterpret_cast<jcn_model *>(this);
    }
};

class ChemtrainOutputTest : public LAMMPSTest {
protected:
    using PairProbe = ChemtrainOutputPairProbe<ChemtrainDeploy>;

    PairProbe *install_pair()
    {
        delete lmp->force->pair;
        delete[] lmp->force->pair_style;
        auto *pair             = new PairProbe(lmp);
        lmp->force->pair       = pair;
        lmp->force->pair_style = utils::strdup("chemtrain");
        return pair;
    }
};

TEST_F(ChemtrainOutputTest, RequiresPairCoeffMetadata)
{
    install_pair()->add_unloaded_context();

    TEST_FAILURE(".*must be defined after pair_coeff.*",
                 command("compute q all chemtrain/output scalar"););
}

TEST_F(ChemtrainOutputTest, ScalarOutputIsPerAtomVectorDuringParsing)
{
    install_pair()->add_output("scalar", 1);

    command("compute q all chemtrain/output scalar");
    auto *compute = lmp->modify->get_compute_by_id("q");
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->peratom_flag, 1);
    EXPECT_EQ(compute->size_peratom_cols, 0);
    EXPECT_NO_THROW(command("compute Q all reduce sum c_q"));
}

TEST_F(ChemtrainOutputTest, MulticomponentOutputIsPerAtomArrayDuringParsing)
{
    install_pair()->add_output("vector", 3);

    command("compute q all chemtrain/output vector");
    auto *compute = lmp->modify->get_compute_by_id("q");
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->peratom_flag, 1);
    EXPECT_EQ(compute->size_peratom_cols, 3);
    EXPECT_NO_THROW(command("compute Q all reduce sum c_q[1]"));
    TEST_FAILURE(".*does not calculate a per-atom vector.*",
                 command("compute invalid all reduce sum c_q"););
}

TEST_F(ChemtrainOutputTest, ScalarParticleOutputZerosAtomsOutsideGroup)
{
    HIDE_OUTPUT([&] {
        command("region box block 0 3 0 1 0 1");
        command("create_box 1 box");
        command("create_atoms 1 single 0.5 0.5 0.5");
        command("create_atoms 1 single 1.5 0.5 0.5");
        command("group subset id 1");
    });
    auto *pair = install_pair();
    pair->add_output("scalar", 1);
    pair->set_output("scalar", {2.0, 3.0});

    command("compute q subset chemtrain/output scalar");
    auto *compute = lmp->modify->get_compute_by_id("q");
    ASSERT_NE(compute, nullptr);
    ASSERT_NO_THROW(compute->init_flags());
    compute->compute_peratom();

    EXPECT_DOUBLE_EQ(compute->vector_atom[0], 2.0);
    EXPECT_DOUBLE_EQ(compute->vector_atom[1], 0.0);
}

TEST_F(ChemtrainOutputTest, VectorParticleOutputZerosAtomsOutsideGroup)
{
    HIDE_OUTPUT([&] {
        command("region box block 0 3 0 1 0 1");
        command("create_box 1 box");
        command("create_atoms 1 single 0.5 0.5 0.5");
        command("create_atoms 1 single 1.5 0.5 0.5");
        command("group subset id 1");
    });
    auto *pair = install_pair();
    pair->add_output("vector", 3);
    pair->set_output("vector", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    command("compute q subset chemtrain/output vector");
    auto *compute = lmp->modify->get_compute_by_id("q");
    ASSERT_NE(compute, nullptr);
    ASSERT_NO_THROW(compute->init_flags());
    compute->compute_peratom();

    EXPECT_DOUBLE_EQ(compute->array_atom[0][0], 1.0);
    EXPECT_DOUBLE_EQ(compute->array_atom[0][1], 2.0);
    EXPECT_DOUBLE_EQ(compute->array_atom[0][2], 3.0);
    EXPECT_DOUBLE_EQ(compute->array_atom[1][0], 0.0);
    EXPECT_DOUBLE_EQ(compute->array_atom[1][1], 0.0);
    EXPECT_DOUBLE_EQ(compute->array_atom[1][2], 0.0);
}

TEST_F(ChemtrainOutputTest, LocalScalarIsSummedAcrossRanks)
{
    auto *pair = install_pair();
    pair->add_output("local_scalar", 1, JCN_OUTPUT_LOCAL);
    pair->set_output("local_scalar", {static_cast<double>(lmp->comm->me + 1)});

    command("compute q all chemtrain/output local_scalar");
    auto *compute = lmp->modify->get_compute_by_id("q");
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->scalar_flag, 1);
    EXPECT_EQ(compute->peratom_flag, 0);
    EXPECT_EQ(compute->extscalar, 0);
    EXPECT_NO_THROW(compute->init_flags());
    const double ranks = static_cast<double>(lmp->comm->nprocs);
    EXPECT_DOUBLE_EQ(compute->compute_scalar(), ranks * (ranks + 1.0) / 2.0);
}

TEST_F(ChemtrainOutputTest, ExtensiveLocalScalarIsMarkedExtensive)
{
    auto *pair = install_pair();
    pair->add_output("local_scalar", 1, JCN_OUTPUT_LOCAL, true);

    command("compute q all chemtrain/output local_scalar");
    auto *compute = lmp->modify->get_compute_by_id("q");
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->scalar_flag, 1);
    EXPECT_EQ(compute->extscalar, 1);
}

TEST_F(ChemtrainOutputTest, ExtensiveGlobalVectorIsMarkedExtensive)
{
    auto *pair = install_pair();
    pair->add_output("global_vector", 3, JCN_OUTPUT_GLOBAL, true);

    command("compute q all chemtrain/output global_vector");
    auto *compute = lmp->modify->get_compute_by_id("q");
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->vector_flag, 1);
    EXPECT_EQ(compute->extvector, 1);
}

TEST_F(ChemtrainOutputTest, GlobalVectorIsNotReduced)
{
    auto *pair = install_pair();
    pair->add_output("global_vector", 3, JCN_OUTPUT_GLOBAL);
    const double rank_value = static_cast<double>(lmp->comm->me + 1);
    pair->set_output("global_vector", {rank_value, 2.0, 3.0});

    command("compute q all chemtrain/output global_vector");
    auto *compute = lmp->modify->get_compute_by_id("q");
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->vector_flag, 1);
    EXPECT_EQ(compute->size_vector, 3);
    EXPECT_EQ(compute->peratom_flag, 0);
    EXPECT_EQ(compute->extvector, 0);
    EXPECT_NO_THROW(compute->init_flags());
    compute->compute_vector();
    EXPECT_DOUBLE_EQ(compute->vector[0], rank_value);
    EXPECT_DOUBLE_EQ(compute->vector[1], 2.0);
    EXPECT_DOUBLE_EQ(compute->vector[2], 3.0);
}

TEST_F(ChemtrainOutputTest, ConfigurationOutputRequiresGroupAll)
{
    install_pair()->add_output("local_scalar", 1, JCN_OUTPUT_LOCAL);
    command("region box block 0 1 0 1 0 1");
    command("create_box 1 box");
    HIDE_OUTPUT([&] {
        command("group subset type 1");
    });
    TEST_FAILURE(".*must use group all.*",
                 command("compute q subset chemtrain/output local_scalar"););
}

#if defined(LMP_TEST_KOKKOS_OUTPUT)
class ChemtrainOutputKokkosTest : public LAMMPSTest {
protected:
    using PairType  = ChemtrainDeployKokkos<LMPDeviceType>;
    using PairProbe = ChemtrainOutputPairProbe<PairType>;

    static void TearDownTestSuite() { lammps_kokkos_finalize(); }

    void SetUp() override
    {
#if defined(LMP_KOKKOS_GPU)
        args.insert(args.end(), {"-k", "on", "g", "1"});
#else
        args.insert(args.end(), {"-k", "on", "t", "1"});
#endif
        args.insert(args.end(), {"-sf", "kk"});
        LAMMPSTest::SetUp();
    }

    PairProbe *install_pair()
    {
        delete lmp->force->pair;
        delete[] lmp->force->pair_style;
        auto *pair             = new PairProbe(lmp);
        lmp->force->pair       = pair;
        lmp->force->pair_style = utils::strdup("chemtrain/kk");
        return pair;
    }
};

TEST_F(ChemtrainOutputKokkosTest, DeviceOutputExposesShapeDuringParsing)
{
    install_pair()->add_output("vector", 3);

    command("compute q all chemtrain/output/kk vector");
    auto *compute = lmp->modify->get_compute_by_id("q");
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->peratom_flag, 1);
    EXPECT_EQ(compute->size_peratom_cols, 3);
    EXPECT_NO_THROW(command("compute Q all reduce sum c_q[1]"));
    TEST_FAILURE(".*does not calculate a per-atom vector.*",
                 command("compute invalid all reduce sum c_q"););
}

TEST_F(ChemtrainOutputKokkosTest, DeviceConfigurationOutputPreservesExtensivity)
{
    install_pair()->add_output("local_scalar", 1, JCN_OUTPUT_LOCAL, true);

    command("compute q all chemtrain/output/kk local_scalar");
    auto *compute = lmp->modify->get_compute_by_id("q");
    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->scalar_flag, 1);
    EXPECT_EQ(compute->extscalar, 1);
}

TEST_F(ChemtrainOutputKokkosTest, DeviceEvaluationInvalidatesSameTimestepHostMirror)
{
    auto *pair = install_pair();
    pair->add_output("scalar", 1);

    const auto timestamps = pair->mark_current_and_sync_output("scalar", lmp->update->ntimestep);

    EXPECT_EQ(timestamps.first, -1);
    EXPECT_EQ(timestamps.second, lmp->update->ntimestep);
}

#endif

} // namespace LAMMPS_NS
