#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CUR_DIR="$(pwd)"

source "${SCRIPT_DIR}/utils.bash"

#---------------HERE STARTS THE SCRIPT---------------

help() {
    echo -e "${BOLD}Usage:${GREEN} $(basename "$0") ${RESET}${RED}<python_venv_path> <cpu/cuda>${RESET}"
    exit 1
}

if [ "$#" -ne 2 ]; then
    help
fi

VENV_DIR="$1"
TARGET_ARCH="$2"

#Activate python venv
source ${VENV_DIR}/bin/activate

cd ${SCRIPT_DIR}/../chemtrain/chemtrain-deploy

if [[ "$TARGET_ARCH" == "cpu" ]]; then
    python build.py \
      --build_cpu_pjrt_plugin \
      --target_cpu_features=default \
      --output_path="$PWD/out" \
      --install_location="$PWD/lib"
elif [[ "$TARGET_ARCH" == "cuda" ]]; then
    python build.py \
      --enable_cuda \
      --build_gpu_pjrt_plugin \
      --cuda_version=${cuda_version} \
      --cudnn_version=${cudnn_version} \
      --cuda_compute_capabilities=${cuda_compute_version} \
      --target_cpu_features=default \
      --output_path="$PWD/out" \
      --install_location="$PWD/lib"
else
    echo -e "Invalid target Architecture!"
    exit 1
fi

cd ${CUR_DIR}
