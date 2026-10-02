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

#include "fix_chemtrain_topology_kokkos.h"

#if defined(LMP_KOKKOS)

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"

#include <type_traits>

using namespace LAMMPS_NS;

template <class DeviceType>
FixChemtrainTopologyKokkos<DeviceType>::FixChemtrainTopologyKokkos(
    LAMMPS *lmp, int narg, char **arg) : FixChemtrainTopology(lmp, narg, arg)
{
  kokkosable = 1;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  forward_comm_device = std::is_same_v<DeviceType, LMPDeviceType>;
}

template <class DeviceType>
void FixChemtrainTopologyKokkos<DeviceType>::refresh()
{
  auto *atomKK = static_cast<AtomKokkos *>(atom);

  const auto ensure_storage = [this]() {
    const int counts_size = std::max(3 * atom->nmax, 1);
    const int partners_size = std::max(maxspecial_ * atom->nmax, 1);
    if (static_cast<int>(k_counts_.extent(0)) < counts_size)
      k_counts_ = DAT::tdual_int_1d("chemtrain:topology_counts",
                                    counts_size);
    if (static_cast<int>(k_partners_.extent(0)) < partners_size)
      k_partners_ = DAT::tdual_tagint_1d(
          "chemtrain:topology_partners", partners_size);
  };

  if constexpr (std::is_same_v<DeviceType, LMPHostType> &&
                !std::is_same_v<DeviceType, LMPDeviceType>) {
    atomKK->sync(Host, SPECIAL_MASK | TAG_MASK);
    FixChemtrainTopology::refresh();

    ensure_storage();
    k_counts_.sync_host();
    k_partners_.sync_host();
    auto h_counts = k_counts_.view_host();
    auto h_partners = k_partners_.view_host();
    const int nall = atom->nlocal + atom->nghost;
    for (int i = 0; i < nall; ++i) {
      for (int k = 0; k < 3; ++k) h_counts(3 * i + k) = counts_[i][k];
      for (int k = 0; k < maxspecial_; ++k)
        h_partners(i * maxspecial_ + k) =
            partners_[static_cast<std::size_t>(i) * maxspecial_ + k];
    }
    k_counts_.modify_host();
    k_partners_.modify_host();
    k_counts_.template sync<DeviceType>();
    k_partners_.template sync<DeviceType>();
    return;
  }

  const bool device_molecular = atom->molecular == Atom::MOLECULAR;
  if (device_molecular) {
    maxspecial_ = atom->maxspecial;
  } else {
    atomKK->sync(Host, TAG_MASK);
    snapshot_owned();
  }

  communication_width_ = 3 + maxspecial_;
  ensure_storage();
  if (device_molecular) {
    atomKK->sync(Device, SPECIAL_MASK);
    auto atom_counts = atomKK->k_nspecial.template view<DeviceType>();
    auto atom_partners = atomKK->k_special.template view<DeviceType>();
    auto topology_counts = k_counts_.template view<DeviceType>();
    auto topology_partners = k_partners_.template view<DeviceType>();
    const int maxspecial = maxspecial_;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType>(0, atom->nlocal), KOKKOS_LAMBDA(int i) {
          for (int k = 0; k < 3; ++k)
            topology_counts(3 * i + k) = atom_counts(i, k);
          const int count = atom_counts(i, 2);
          for (int k = 0; k < maxspecial; ++k)
            topology_partners(i * maxspecial + k) =
                k < count ? atom_partners(i, k) : 0;
        });
    k_counts_.template modify<DeviceType>();
    k_partners_.template modify<DeviceType>();
  } else {
    k_counts_.sync_host();
    k_partners_.sync_host();
    auto h_counts = k_counts_.view_host();
    auto h_partners = k_partners_.view_host();
    for (int i = 0; i < atom->nlocal; ++i) {
      for (int k = 0; k < 3; ++k) h_counts(3 * i + k) = counts_[i][k];
      for (int k = 0; k < maxspecial_; ++k)
        h_partners(i * maxspecial_ + k) =
            partners_[static_cast<std::size_t>(i) * maxspecial_ + k];
    }
    k_counts_.modify_host();
    k_partners_.modify_host();
    k_counts_.template sync<DeviceType>();
    k_partners_.template sync<DeviceType>();
  }
  if (atom->nghost) comm->forward_comm(this, communication_width_);
  ++generation_;
}

template <class DeviceType>
int FixChemtrainTopologyKokkos<DeviceType>::pack_forward_comm(
    int n, int *list, double *buf, int, int *)
{
  k_counts_.sync_host();
  k_partners_.sync_host();
  auto counts = k_counts_.view_host();
  auto partners = k_partners_.view_host();
  for (int ii = 0; ii < n; ++ii) {
    const int i = list[ii];
    for (int k = 0; k < 3; ++k) buf[ii * communication_width_ + k] = ubuf(counts(3 * i + k)).d;
    for (int k = 0; k < maxspecial_; ++k)
      buf[ii * communication_width_ + 3 + k] = ubuf(partners(i * maxspecial_ + k)).d;
  }
  return n * communication_width_;
}

template <class DeviceType>
void FixChemtrainTopologyKokkos<DeviceType>::unpack_forward_comm(
    int n, int first, double *buf)
{
  k_counts_.sync_host();
  k_partners_.sync_host();
  auto counts = k_counts_.view_host();
  auto partners = k_partners_.view_host();
  for (int ii = 0; ii < n; ++ii) {
    const int i = first + ii;
    for (int k = 0; k < 3; ++k) counts(3 * i + k) = ubuf(buf[ii * communication_width_ + k]).i;
    for (int k = 0; k < maxspecial_; ++k)
      partners(i * maxspecial_ + k) = ubuf(buf[ii * communication_width_ + 3 + k]).i;
  }
  k_counts_.modify_host();
  k_partners_.modify_host();
  k_counts_.template sync<DeviceType>();
  k_partners_.template sync<DeviceType>();
}

template <class DeviceType>
int FixChemtrainTopologyKokkos<DeviceType>::pack_forward_comm_kokkos(
    int n, DAT::tdual_int_1d sendlist, DAT::tdual_double_1d &buf, int, int *)
{
  auto list = sendlist.template view<DeviceType>();
  auto output = buf.template view<DeviceType>();
  auto counts = k_counts_.template view<DeviceType>();
  auto partners = k_partners_.template view<DeviceType>();
  const int width = communication_width_;
  const int maxspecial = maxspecial_;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(int ii) {
    const int i = list(ii);
    for (int k = 0; k < 3; ++k) output(ii * width + k) = d_ubuf(counts(3 * i + k)).d;
    for (int k = 0; k < maxspecial; ++k)
      output(ii * width + 3 + k) = d_ubuf(partners(i * maxspecial + k)).d;
  });
  return n * width;
}

template <class DeviceType>
void FixChemtrainTopologyKokkos<DeviceType>::unpack_forward_comm_kokkos(
    int n, int first, DAT::tdual_double_1d &buf)
{
  auto input = buf.template view<DeviceType>();
  auto counts = k_counts_.template view<DeviceType>();
  auto partners = k_partners_.template view<DeviceType>();
  const int width = communication_width_;
  const int maxspecial = maxspecial_;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(int ii) {
    const int i = first + ii;
    for (int k = 0; k < 3; ++k) counts(3 * i + k) = d_ubuf(input(ii * width + k)).i;
    for (int k = 0; k < maxspecial; ++k)
      partners(i * maxspecial + k) = d_ubuf(input(ii * width + 3 + k)).i;
  });
  k_counts_.template modify<DeviceType>();
  k_partners_.template modify<DeviceType>();
}

template class FixChemtrainTopologyKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixChemtrainTopologyKokkos<LMPHostType>;
#endif

#endif
