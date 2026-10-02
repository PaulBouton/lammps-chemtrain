#!/bin/sh

# The CHEMTRAIN-DEPLOY package only supports the CMake build system.

if test "$1" = 1 || test "$1" = 2; then
  echo "The CHEMTRAIN-DEPLOY package does not support the legacy build system. Please build LAMMPS with CMake instead."
  exit 1
fi

if test "$1" = 0; then
  for file in *.cpp *.h; do
    test -f "$file" && rm -f "../$file"
  done
fi
