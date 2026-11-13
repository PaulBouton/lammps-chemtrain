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

#Reset venv
rm -rf "${VENV_DIR}"
mkdir "${VENV_DIR}"

${python_version} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

#Install needed pip
pip install -e "${SCRIPT_DIR}/../chemtrain"
pip install -e "${SCRIPT_DIR}/../chemtrain[test]"

if [[ "$TARGET_ARCH" == "cuda" ]]; then
    pip install -e "${SCRIPT_DIR}/../chemtrain[cuda${MAIN_CUDA_VERSION}]"
    pip install nvidia-cuda-nvcc-cu${MAIN_CUDA_VERSION}==${pip_cuda_version}
fi

pip install -e "${SCRIPT_DIR}/../tensorial[dev]"
pip install -e "${SCRIPT_DIR}/../tensorial[docs]"

pip install -r "${SCRIPT_DIR}/../requirements.txt"

pip install --upgrade pip
