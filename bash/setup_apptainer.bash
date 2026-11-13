#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CUR_DIR="$(pwd)"

source "${SCRIPT_DIR}/utils.bash"

#Exit on Error
set -e

help() {
    echo -e "${BOLD}Usage:${GREEN} $(basename "$0") ${RESET}${RED} <cpu/cuda> [optional]<compression_level>${RESET}"
    exit 1
}

if [ "$#" -lt 1 ]; then
    help
fi

TARGET_ARCH="$1"
COMPRESSION_LEVEL="$2"

if [[ -z "$COMPRESSION_LEVEL"  ]]; then
    COMPRESSION_LEVEL=19
fi

echo -e "${GREEN}${BOLD}Targeted Compression Level : $COMPRESSION_LEVEL${BOLD}"

#Change tmp to have enough space
export APPTAINER_TMPDIR=${SCRIPT_DIR}/../tmp
rm -rf ${APPTAINER_TMPDIR}
mkdir ${APPTAINER_TMPDIR}

echo -e "${GREEN}Cleaning Chemtrain directory${RESET}"
rm -rf chemtrain
git submodule update --init --recursive

echo -e "${GREEN}Cleaning previous Apptainer builds${RESET}"
cd ${SCRIPT_DIR}/../apptainer
apptainer cache clean --force
rm -rf ~/.apptainer/cache/*
rm -rf *.sif

if [[ "$TARGET_ARCH" == "cpu" ]]; then
    echo -e "${GREEN}Building CPU apptainer image${RESET}"
    apptainer build --mksquashfs-args "-comp zstd -Xcompression-level ${COMPRESSION_LEVEL}" lammps-chemtrain.sif lammps-chemtrain.def
elif [[ "$TARGET_ARCH" == "cuda" ]]; then
    echo -e "${GREEN}Building CUDA apptainer image${RESET}"
    apptainer build --mksquashfs-args "-comp zstd -Xcompression-level ${COMPRESSION_LEVEL}" lammps-chemtrain-cuda.sif lammps-chemtrain-cuda.def
else
    echo -e "${RED}Invalid target Architecture!${RESET}"
    exit 1
fi

echo -e "${GREEN} Cleaning${RESET}"
rm -rf ${APPTAINER_TMPDIR}

cd ${CUR_DIR}
echo -e "${GREEN} Done${RESET}"
