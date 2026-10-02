/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Contributing author: Paul Fuchs (Technical University of Munich), paul.fuchs@tum.de
------------------------------------------------------------------------- */

// unit tests for the chemtrain topology communication fix

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "chemtrain_neighbor_sparse.h"
#include "dihedral.h"
#include "fix_chemtrain_topology.h"
#include "force.h"
#include "improper.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair_chemtrain_deploy.h"
#include "special.h"

#if defined(LMP_TEST_KOKKOS_HOST)
#include "chemtrain_neighbor_sparse_kokkos.h"
#include "fix_chemtrain_topology_kokkos.h"
#include "library.h"
#endif

#include "../testing/core.h"
#include "../testing/test_mpi_main.h"
#include "gtest/gtest.h"

#include <array>
#include <mpi.h>
#include <string>
#include <vector>

namespace LAMMPS_NS {

static constexpr int CHAIN_ATOMS = 5;
static constexpr int CATEGORY_ROW_WIDTH = CHAIN_ATOMS + 1;
static constexpr int TAG_TABLE_SIZE = CHAIN_ATOMS + 1;

struct TopologyTestParam {
    const char *newton_bond;
    bool kokkos_host;
};

class ChemtrainDeployProbe : public ChemtrainDeploy {
public:
    explicit ChemtrainDeployProbe(LAMMPS *lmp) : ChemtrainDeploy(lmp) {}

    void bind_topology(FixChemtrainTopology *fix) { topology_lookup = fix; }
    void unbind_topology() { topology_lookup = nullptr; }
    int classify(int sender, int receiver) const
    {
        return topology_category(sender, receiver);
    }
    void install_topology_styles() { install_missing_topology_styles(); }
    void validate_particle_source(const std::string &source)
    {
        ChemtrainParticleInputBinding binding;
        binding.name = "field";
        binding.source = source;
        validate_custom_particle_input(binding);
    }
};

TEST_F(LAMMPSTest, ChemtrainParticleInputUsesScalarIntegerCustomProperties)
{
    HIDE_OUTPUT([&] {
        command("units lj");
        command("atom_style atomic");
        command("region box block 0 2 0 2 0 2");
        command("create_box 1 box");
        command("create_atoms 1 single 1 1 1");
        command("fix valid all property/atom i_field ghost yes");
        command("fix no_ghost all property/atom i_local");
    });

    ChemtrainDeployProbe probe(lmp);
    EXPECT_NO_THROW(probe.validate_particle_source("i_field"));
    TEST_FAILURE(".*must name an unindexed scalar integer custom property.*",
                 probe.validate_particle_source("i_field[1]"););
    TEST_FAILURE(".*must name an unindexed scalar integer custom property.*",
                 probe.validate_particle_source("i2_field"););
    TEST_FAILURE(".*must name an unindexed scalar integer custom property.*",
                 probe.validate_particle_source("d_field"););
    TEST_FAILURE(".*requires fix property/atom ghost yes.*",
                 probe.validate_particle_source("i_local"););
    TEST_FAILURE(".*does not exist.*", probe.validate_particle_source("i_missing"););
}

#if defined(LMP_TEST_KOKKOS_HOST)
struct MockPositions {
    double values[2][3];
    KOKKOS_INLINE_FUNCTION
    double operator()(int i, int j) const { return values[i][j]; }
};

struct MockOneDimensional {
    int values[6];
    KOKKOS_INLINE_FUNCTION
    int operator()(int i) const { return values[i]; }
};

TEST(ChemtrainKokkosCategory, DetectsOneFiveAndRejectsPeriodicImage)
{
    using Neighbor = ChemtrainNeighborSparseKokkos<LMPHostType>;
    MockPositions positions{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}};
    const MockOneDimensional tags{{1, 5, 0, 0, 0, 0}};
    const MockOneDimensional counts{{1, 2, 3, 1, 2, 3}};
    MockOneDimensional partners{{2, 3, 4, 4, 3, 2}};

    EXPECT_EQ(Neighbor::category(0, 1, 0, 2, positions, tags, counts, partners,
                                 3, 1, 1, 1, 4.0, 4.0, 4.0),
              4);
    EXPECT_EQ(Neighbor::category(0, 1, 2, 2, positions, tags, counts, partners,
                                 3, 1, 1, 1, 4.0, 4.0, 4.0),
              2);
    partners.values[3] = 1;
    EXPECT_EQ(Neighbor::category(1, 0, 0, 1, positions, tags, counts, partners,
                                 3, 1, 1, 1, 4.0, 4.0, 4.0),
              1);
    positions.values[1][0] = 4.1;
    EXPECT_EQ(Neighbor::category(0, 1, 0, 2, positions, tags, counts, partners,
                                 3, 1, 1, 1, 4.0, 4.0, 4.0),
              0);
}
#endif

std::vector<TopologyTestParam> topology_test_parameters()
{
    std::vector<TopologyTestParam> parameters = {{"on", false}, {"off", false}};
#if defined(LMP_TEST_KOKKOS_HOST)
    parameters.push_back({"on", true});
    parameters.push_back({"off", true});
#endif
    return parameters;
}

class ChemtrainTopologyTest : public LAMMPSTest,
                              public ::testing::WithParamInterface<TopologyTestParam> {
protected:
    static void TearDownTestSuite()
    {
#if defined(LMP_TEST_KOKKOS_HOST)
        lammps_kokkos_finalize();
#endif
    }

    void SetUp() override
    {
        if (GetParam().kokkos_host)
            args.insert(args.end(),
                        {"-k", "on", "g", "1", "-pk", "kokkos", "neigh", "half"});
        LAMMPSTest::SetUp();
    }

    void InitSystem() override
    {
        int nprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        ASSERT_TRUE(nprocs == 1 || nprocs == 2);

        HIDE_OUTPUT([&] {
            command("units lj");
            command(GetParam().kokkos_host ? "atom_style molecular/kk" :
                                             "atom_style molecular");
            command("atom_modify map array sort 0 0.0");
            command("boundary p p p");
            command(std::string("newton on ") + GetParam().newton_bond);
            command("special_bonds lj/coul 1.0 1.0 1.0");
            if (nprocs == 2) command("processors 2 1 1");
            command("region box block 0 8 0 2 0 2");
            command("create_box 1 box bond/types 1 angle/types 1 dihedral/types 1 "
                    "improper/types 1 extra/bond/per/atom 2 "
                    "extra/special/per/atom 4");
            command("mass 1 1.0");
            // The chain provides 1-2 through 1-4 special pairs and a 1-5 control.
            command("create_atoms 1 single 3.2 1.0 1.0");
            command("create_atoms 1 single 3.8 1.0 1.0");
            command("create_atoms 1 single 4.4 1.0 1.0");
            command("create_atoms 1 single 5.0 1.0 1.0");
            command("create_atoms 1 single 5.6 1.0 1.0");
            command("bond_style zero");
            command("bond_coeff 1");
            command("angle_style zero");
            command("angle_coeff 1");
            command("dihedral_style zero");
            command("dihedral_coeff 1");
            command("improper_style zero");
            command("improper_coeff 1");
            command("pair_style zero 3.0");
            command("pair_coeff * *");
            command("create_bonds single/bond 1 1 2");
            command("create_bonds single/bond 1 2 3");
            command("create_bonds single/bond 1 3 4");
            command("create_bonds single/bond 1 4 5");

            Special special(lmp);
            special.build(true);
            if (GetParam().kokkos_host) command("run_style verlet/kk");
            command(GetParam().kokkos_host ? "fix topology all CHEMTRAIN/TOPOLOGY/kk/host" :
                                             "fix topology all CHEMTRAIN/TOPOLOGY");
            command("run 0 post no");
        });
    }

    FixChemtrainTopology *topology_fix()
    {
        return dynamic_cast<FixChemtrainTopology *>(lmp->modify->get_fix_by_id("topology"));
    }

    int owner_of(tagint tag)
    {
        int me;
        MPI_Comm_rank(MPI_COMM_WORLD, &me);
        int owner = -1;
        for (int i = 0; i < lmp->atom->nlocal; ++i)
            if (lmp->atom->tag[i] == tag) owner = me;
        int global_owner;
        MPI_Allreduce(&owner, &global_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        return global_owner;
    }

    int native_category(int index, tagint partner) const
    {
        for (int j = 0; j < lmp->atom->nspecial[index][2]; ++j) {
            if (lmp->atom->special[index][j] != partner) continue;
            if (j < lmp->atom->nspecial[index][0]) return 1;
            if (j < lmp->atom->nspecial[index][1]) return 2;
            return 3;
        }
        return 0;
    }

    std::array<std::array<int, CATEGORY_ROW_WIDTH>, TAG_TABLE_SIZE>
    gather_owned_rows() const
    {
        // Collect one special-neighbor category row for each owned atom.
        std::vector<int> local;
        for (int i = 0; i < lmp->atom->nlocal; ++i) {
            local.push_back(static_cast<int>(lmp->atom->tag[i]));
            for (tagint partner = 1; partner <= CHAIN_ATOMS; ++partner)
                local.push_back(native_category(i, partner));
        }

        int nprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        int local_size = static_cast<int>(local.size());
        std::vector<int> sizes(nprocs);
        MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> displacements(nprocs, 0);
        for (int i = 1; i < nprocs; ++i)
            displacements[i] = displacements[i - 1] + sizes[i - 1];
        std::vector<int> gathered(displacements.back() + sizes.back());
        MPI_Allgatherv(local.data(), local_size, MPI_INT, gathered.data(), sizes.data(),
                       displacements.data(), MPI_INT, MPI_COMM_WORLD);

        // Build expected rows from the data gathered from all ranks.
        std::array<std::array<int, CATEGORY_ROW_WIDTH>, TAG_TABLE_SIZE> rows{};
        std::array<int, TAG_TABLE_SIZE> seen{};
        for (std::size_t offset = 0; offset < gathered.size(); offset += CATEGORY_ROW_WIDTH) {
            const int tag = gathered[offset];
            EXPECT_GE(tag, 1);
            EXPECT_LE(tag, CHAIN_ATOMS);
            if (tag < 1 || tag > CHAIN_ATOMS) continue;
            ++seen[tag];
            for (int partner = 1; partner <= CHAIN_ATOMS; ++partner)
                rows[tag][partner] = gathered[offset + partner];
        }
        // Each chain atom has exactly one owning rank.
        for (int tag = 1; tag <= CHAIN_ATOMS; ++tag) EXPECT_EQ(seen[tag], 1);
        return rows;
    }

    void check_local_and_ghost_rows(FixChemtrainTopology *fix) const
    {
        const auto rows = gather_owned_rows();
        int remote_ghosts = 0;
        const int nall = lmp->atom->nlocal + lmp->atom->nghost;
        ASSERT_GE(static_cast<int>(fix->counts().size()), nall);

        for (int i = 0; i < nall; ++i) {
            const int tag = static_cast<int>(lmp->atom->tag[i]);
            if (tag < 1 || tag > CHAIN_ATOMS) continue;
            if (i >= lmp->atom->nlocal) ++remote_ghosts;

            std::array<int, 3> cumulative{};
            for (int partner = 1; partner <= CHAIN_ATOMS; ++partner) {
                EXPECT_EQ(fix->category(i, partner), rows[tag][partner])
                    << "atom " << tag << ", partner " << partner;
                const int category = rows[tag][partner];
                if (category > 0)
                    for (int level = category - 1; level < 3; ++level) ++cumulative[level];
            }
            EXPECT_EQ(fix->counts()[i], cumulative) << "atom " << tag;
        }

        int global_remote_ghosts;
        MPI_Allreduce(&remote_ghosts, &global_remote_ghosts, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        int nprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        if (nprocs == 2) EXPECT_GT(global_remote_ghosts, 0);

#if defined(LMP_TEST_KOKKOS_HOST)
        if (GetParam().kokkos_host) {
            auto *fix_kk = dynamic_cast<FixChemtrainTopologyKokkos<LMPHostType> *>(fix);
            ASSERT_NE(fix_kk, nullptr);
            const auto counts = fix_kk->counts_view();
            const auto partners = fix_kk->partners_view();
            for (int i = 0; i < nall; ++i) {
                for (int k = 0; k < 3; ++k)
                    EXPECT_EQ(counts(3 * i + k), fix->counts()[i][k]);
                for (int k = 0; k < fix->maxspecial(); ++k)
                    EXPECT_EQ(partners(i * fix->maxspecial() + k),
                              fix->partners()[static_cast<std::size_t>(i) *
                                                  fix->maxspecial() + k]);
            }
        }
#endif
    }
};

TEST_P(ChemtrainTopologyTest, SparseEdgesKeepCentralAtomAsSender)
{
    NeighList *list = lmp->force->pair->list;
    ASSERT_NE(list, nullptr);

    int64_t raw_edges = 0;
    for (int ii = 0; ii < list->inum; ++ii) {
        const int central = list->ilist[ii];
        raw_edges += list->numneigh[central];
    }
    ASSERT_GT(raw_edges, 0);
    // Reserve one unused atom index so the test also checks the padded edge.
    const int64_t padding_index = lmp->atom->nlocal + lmp->atom->nghost + 1;

#if defined(LMP_TEST_KOKKOS_HOST)
    if (GetParam().kokkos_host) {
        auto *list_kk = dynamic_cast<NeighListKokkos<LMPHostType> *>(list);
        auto *atom_kk = dynamic_cast<AtomKokkos *>(lmp->atom);
        ASSERT_NE(list_kk, nullptr);
        ASSERT_NE(atom_kk, nullptr);

        ChemtrainNeighborSparseKokkos<LMPHostType> storage(nullptr);
        int64_t observed_edges = 0;
        int max_neighbors = 0;
        storage.analyze(list_kk, list->inum, observed_edges, max_neighbors);
        ASSERT_EQ(observed_edges, raw_edges);
        storage.build(list_kk, list->inum, lmp->atom->nlocal, raw_edges + 1,
                      padding_index, false, nullptr, atom_kk, lmp->domain);

        const auto senders = storage.senders();
        const auto receivers = storage.receivers();
        int64_t edge = 0;
        for (int ii = 0; ii < list->inum; ++ii) {
            const int central = list->ilist[ii];
            for (int jj = 0; jj < list->numneigh[central]; ++jj) {
                const int neighbor = list->firstneigh[central][jj] & NEIGHMASK;
                EXPECT_EQ(senders(edge), central);
                EXPECT_EQ(receivers(edge), neighbor);
                ++edge;
            }
        }
        EXPECT_EQ(senders(raw_edges), padding_index);
        EXPECT_EQ(receivers(raw_edges), padding_index);
        return;
    }
#endif

    ChemtrainNeighborSparse storage;
    storage.build(list, list->inum, raw_edges + 1, padding_index, false,
                  [](int, int, int) { return 0; }, lmp->error);
    int64_t edge = 0;
    for (int ii = 0; ii < list->inum; ++ii) {
        const int central = list->ilist[ii];
        for (int jj = 0; jj < list->numneigh[central]; ++jj) {
            const int neighbor = list->firstneigh[central][jj] & NEIGHMASK;
            EXPECT_EQ(storage.senders()[edge], central);
            EXPECT_EQ(storage.receivers()[edge], neighbor);
            ++edge;
        }
    }
    EXPECT_EQ(storage.senders()[raw_edges], padding_index);
    EXPECT_EQ(storage.receivers()[raw_edges], padding_index);
}

TEST_P(ChemtrainTopologyTest, CommunicatesExplicitMolecularRowsAfterRebuild)
{
    auto *fix = topology_fix();
    ASSERT_NE(fix, nullptr);
    EXPECT_GE(fix->maxspecial(), 3);
    check_local_and_ghost_rows(fix);

#if defined(LMP_TEST_KOKKOS_HOST)
    void *counts_storage = nullptr;
    void *partners_storage = nullptr;
    if (GetParam().kokkos_host) {
        auto *fix_kk = dynamic_cast<FixChemtrainTopologyKokkos<LMPHostType> *>(fix);
        ASSERT_NE(fix_kk, nullptr);
        counts_storage = fix_kk->counts_view().data();
        partners_storage = fix_kk->partners_view().data();
    }
#endif

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const int owner_before = owner_of(2);
    const int generation_before = fix->generation();

    HIDE_OUTPUT([&] {
        command("group crossing id 2");
        command("displace_atoms crossing move 0.3 0.0 0.0 units box");
        command("run 0 post no");
    });

    fix = topology_fix();
    ASSERT_NE(fix, nullptr);
    EXPECT_GT(fix->generation(), generation_before);
    check_local_and_ghost_rows(fix);
#if defined(LMP_TEST_KOKKOS_HOST)
    if (GetParam().kokkos_host) {
        auto *fix_kk = dynamic_cast<FixChemtrainTopologyKokkos<LMPHostType> *>(fix);
        ASSERT_NE(fix_kk, nullptr);
        EXPECT_EQ(fix_kk->counts_view().data(), counts_storage);
        EXPECT_EQ(fix_kk->partners_view().data(), partners_storage);
    }
#endif
    if (nprocs == 2) EXPECT_NE(owner_of(2), owner_before);
}

TEST_P(ChemtrainTopologyTest, DetectsOneFiveFromCommunicatedRows)
{
    auto *fix = topology_fix();
    ASSERT_NE(fix, nullptr);

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int checked = 0;
    int checked_with_ghost = 0;
    const int nall = lmp->atom->nlocal + lmp->atom->nghost;
    for (int i = 0; i < nall; ++i) {
        const tagint itag = lmp->atom->tag[i];
        if (itag != 1 && itag != 5) continue;
        const tagint partner = itag == 1 ? 5 : 1;
        for (int j = 0; j < nall; ++j) {
            if (lmp->atom->tag[j] != partner) continue;
            EXPECT_EQ(fix->category(i, partner), 0);
            EXPECT_EQ(fix->onefive(i, j), 4);
            ++checked;
            if (i >= lmp->atom->nlocal || j >= lmp->atom->nlocal)
                ++checked_with_ghost;
            break;
        }
    }

    int global_checked = 0;
    MPI_Allreduce(&checked, &global_checked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_GE(global_checked, 2);
    if (nprocs == 2) {
        int global_checked_with_ghost = 0;
        MPI_Allreduce(&checked_with_ghost, &global_checked_with_ghost, 1, MPI_INT,
                      MPI_SUM, MPI_COMM_WORLD);
        EXPECT_GT(global_checked_with_ghost, 0);
    }
}

TEST_P(ChemtrainTopologyTest, PreservesShortestCategoryAndRejectsPeriodicImage)
{
    auto *fix = topology_fix();
    ASSERT_NE(fix, nullptr);
    ChemtrainDeployProbe probe(lmp);
    probe.bind_topology(fix);

    const int first = lmp->atom->map(1);
    const int fifth = lmp->atom->map(5);
    ASSERT_GE(first, 0);
    ASSERT_GE(fifth, 0);
    EXPECT_EQ(probe.classify(first, fifth), 4);

    const double saved_x = lmp->atom->x[fifth][0];
    lmp->atom->x[fifth][0] = lmp->atom->x[first][0] + 4.1;
    EXPECT_EQ(probe.classify(first, fifth), 0);
    lmp->atom->x[fifth][0] = saved_x;

    HIDE_OUTPUT([&] {
        command("create_bonds single/bond 1 1 5");
        Special special(lmp);
        special.build(true);
        command("run 0 post no");
    });
    fix = topology_fix();
    ASSERT_NE(fix, nullptr);
    probe.bind_topology(fix);
    EXPECT_EQ(probe.classify(lmp->atom->map(1), lmp->atom->map(5)), 1);
    probe.unbind_topology();
}

TEST_P(ChemtrainTopologyTest, FollowsRetainedRowsAfterDihedralTrimming)
{
    HIDE_OUTPUT([&] {
        command("special_bonds lj/coul 1.0 1.0 1.0 dihedral yes");
        Special special(lmp);
        special.build(true);
        command("run 0 post no");
    });

    auto *fix = topology_fix();
    ASSERT_NE(fix, nullptr);
    const int first = lmp->atom->map(1);
    const int fifth = lmp->atom->map(5);
    ASSERT_GE(first, 0);
    ASSERT_GE(fifth, 0);
    EXPECT_EQ(fix->category(first, 4), 0);
    EXPECT_EQ(fix->onefive(first, fifth), 0);
}

TEST_P(ChemtrainTopologyTest, InstallsPreservesAndPersistsTopologyStyles)
{
    const auto original_nbonds = lmp->atom->nbonds;
    const auto original_nangles = lmp->atom->nangles;
    const auto original_ndihedrals = lmp->atom->ndihedrals;
    const auto original_nimpropers = lmp->atom->nimpropers;
    ChemtrainDeployProbe probe(lmp);
    auto *configured_bond = lmp->force->bond;
    auto *configured_angle = lmp->force->angle;
    auto *configured_dihedral = lmp->force->dihedral;
    auto *configured_improper = lmp->force->improper;
    ASSERT_NE(configured_bond, nullptr);
    ASSERT_NE(configured_angle, nullptr);
    ASSERT_NE(configured_dihedral, nullptr);
    ASSERT_NE(configured_improper, nullptr);
    probe.install_topology_styles();
    EXPECT_EQ(lmp->force->bond, configured_bond);
    EXPECT_EQ(lmp->force->angle, configured_angle);
    EXPECT_EQ(lmp->force->dihedral, configured_dihedral);
    EXPECT_EQ(lmp->force->improper, configured_improper);

    HIDE_OUTPUT([&] {
        command("bond_style none");
        command("angle_style none");
        command("dihedral_style none");
        command("improper_style none");
    });
    ASSERT_EQ(lmp->force->bond, nullptr);
    ASSERT_EQ(lmp->force->angle, nullptr);
    ASSERT_EQ(lmp->force->dihedral, nullptr);
    ASSERT_EQ(lmp->force->improper, nullptr);

    lmp->atom->nbonds = lmp->atom->nangles = 1;
    lmp->atom->ndihedrals = lmp->atom->nimpropers = 1;
    {
        ChemtrainDeployProbe owner(lmp);
        owner.install_topology_styles();
        ASSERT_NE(lmp->force->bond, nullptr);
        ASSERT_NE(lmp->force->angle, nullptr);
        ASSERT_NE(lmp->force->dihedral, nullptr);
        ASSERT_NE(lmp->force->improper, nullptr);
        EXPECT_EQ(lmp->force->bond->datamask_read, 0U);
        EXPECT_EQ(lmp->force->angle->datamask_read, 0U);
        EXPECT_EQ(lmp->force->dihedral->datamask_read, 0U);
        EXPECT_EQ(lmp->force->improper->datamask_read, 0U);
    }
    EXPECT_NE(lmp->force->bond, nullptr);
    EXPECT_NE(lmp->force->angle, nullptr);
    EXPECT_NE(lmp->force->dihedral, nullptr);
    EXPECT_NE(lmp->force->improper, nullptr);
    EXPECT_STREQ(lmp->force->bond_style, "zero");
    EXPECT_STREQ(lmp->force->angle_style, "zero");
    EXPECT_STREQ(lmp->force->dihedral_style, "zero");
    EXPECT_STREQ(lmp->force->improper_style, "zero");
    HIDE_OUTPUT([&] {
        command("bond_style none");
        command("angle_style none");
        command("dihedral_style none");
        command("improper_style none");
    });
    lmp->atom->nbonds = original_nbonds;
    lmp->atom->nangles = original_nangles;
    lmp->atom->ndihedrals = original_ndihedrals;
    lmp->atom->nimpropers = original_nimpropers;
}

INSTANTIATE_TEST_SUITE_P(NewtonBond, ChemtrainTopologyTest,
                         ::testing::ValuesIn(topology_test_parameters()));

}    // namespace LAMMPS_NS
