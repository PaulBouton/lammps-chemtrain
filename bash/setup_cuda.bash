#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CUR_DIR="$(pwd)"

source "${SCRIPT_DIR}/utils.bash"

# Load CUDA
module load cuda/${cuda_version_hpc_cluster}
export CUDA_HOME=/softs/cuda/${cuda_dir_hpc_cluster}

# Load CUDNN
module load cudnn/${cudnn_version_hpc_cluster}
export CUDNN_HOME=/softs/cudnn/${cudnn_version_hpc_cluster}

# Add to path and Linker Path
export PATH=$CUDNN_HOME/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

