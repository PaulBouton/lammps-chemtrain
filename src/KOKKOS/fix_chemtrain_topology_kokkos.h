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

#if defined(LMP_KOKKOS)

#ifdef FIX_CLASS
// clang-format off
FixStyle(CHEMTRAIN/TOPOLOGY/kk,FixChemtrainTopologyKokkos<LMPDeviceType>);
FixStyle(CHEMTRAIN/TOPOLOGY/kk/device,FixChemtrainTopologyKokkos<LMPDeviceType>);
FixStyle(CHEMTRAIN/TOPOLOGY/kk/host,FixChemtrainTopologyKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_CHEMTRAIN_TOPOLOGY_KOKKOS_H
#define LMP_FIX_CHEMTRAIN_TOPOLOGY_KOKKOS_H

#include "fix_chemtrain_topology.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template <class DeviceType>
class FixChemtrainTopologyKokkos : public FixChemtrainTopology, public KokkosBase {
 public:
  using AT = ArrayTypes<DeviceType>;

  FixChemtrainTopologyKokkos(class LAMMPS *, int, char **);

  void refresh() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d,
                               DAT::tdual_double_1d &, int, int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d &) override;

  typename AT::t_int_1d counts_view() const
  {
    return k_counts_.template view<DeviceType>();
  }
  typename AT::t_tagint_1d partners_view() const
  {
    return k_partners_.template view<DeviceType>();
  }

 private:
  DAT::tdual_int_1d k_counts_;
  DAT::tdual_tagint_1d k_partners_;
  int communication_width_ = 0;
};

}    // namespace LAMMPS_NS

#endif
#endif
#endif
