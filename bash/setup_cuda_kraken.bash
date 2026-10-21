#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CUR_DIR="$(pwd)"

# To modify if CUDA version changes
cuda_version="12.9.41.patched"
cudnn_version="9.10.1.4"

source "${SCRIPT_DIR}/utils.bash"

help() {
    echo -e "${BOLD}Usage:${GREEN} $(basename "$0") ${RESET}${RED}<python_venv_path>${RESET}"
}

if [ "$#" -ne 1 ]; then
    help
else
    VENV_PYTHON_PATH="$1"
    py_dir="$(ls -d ${VENV_PYTHON_PATH}/lib/python3.* | head -1)"
    nvcc_dir="${py_dir}/site-packages/nvidia/cuda_nvcc/"

    # Load CUDA
    module load cuda/${cuda_version}
    export CUDA_HOME=/softs/cuda/${cuda_version}

    # Load CUDNN
    module load cudnn/${cudnn_version}
    export CUDA_HOME=/softs/cudnn/${cudnn_version}

    # Add to path and Linker Path
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

    # XLA flags
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${nvcc_dir}"
    export PATH="${nvcc_dir}/bin:$PATH"
fi


