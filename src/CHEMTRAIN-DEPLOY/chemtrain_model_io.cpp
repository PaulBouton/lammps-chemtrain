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

#include "comm.h"
#include "error.h"
#include "lammps.h"
#include "lmptype.h"
#include "platform.h"
#include "safe_pointers.h"
#include "utils.h"

#include <algorithm>
#include <climits>
#include <cstdio>

#include <mpi.h>

using namespace LAMMPS_NS;

std::vector<char> LAMMPS_NS::read_chemtrain_model(LAMMPS *lmp,
                                                  const char *filename)
{
  bigint size = 0;
  std::vector<char> bytes;

  if (lmp->comm->me == 0) {
    SafeFilePtr file = fopen(filename, "rb");
    if (file == nullptr)
      lmp->error->one(FLERR, "Cannot open chemtrain model {}: {}", filename,
                      utils::getsyserror());
    if (platform::fseek(file, platform::END_OF_FILE) != 0)
      lmp->error->one(FLERR, "Cannot seek chemtrain model {}", filename);
    size = platform::ftell(file);
    if (size < 0 || platform::fseek(file, 0) != 0)
      lmp->error->one(FLERR, "Cannot determine size of chemtrain model {}", filename);
    bytes.resize(static_cast<size_t>(size));
    if (size > 0)
      utils::sfread(FLERR, bytes.data(), sizeof(char), static_cast<size_t>(size),
                    file, filename, lmp->error);
  }

  MPI_Bcast(&size, 1, MPI_LMP_BIGINT, 0, lmp->world);
  if (size < 0)
    lmp->error->all(FLERR, "Invalid size for chemtrain model {}", filename);
  if (lmp->comm->me != 0) bytes.resize(static_cast<size_t>(size));

  bigint offset = 0;
  while (offset < size) {
    const int chunk = static_cast<int>(std::min<bigint>(size - offset, INT_MAX));
    MPI_Bcast(bytes.data() + offset, chunk, MPI_BYTE, 0, lmp->world);
    offset += chunk;
  }
  return bytes;
}
