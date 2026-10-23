#!/bin/bash

#exit on error
set -e

#Bash color for prints
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

#Core count
n_cores=$(nproc --all)

#Lib Versions

#Python
python_version="python3.13"

#CUDA
MAIN_CUDA_VERSION="12"

cuda_version="${MAIN_CUDA_VERSION}.9.2"
cudnn_version="9.10.1"
cuda_compute_version="9.0"

pip_cuda_version="${MAIN_CUDA_VERSION}.9.41"

cuda_version_hpc_cluster="${MAIN_CUDA_VERSION}.9.41.patched"
cudnn_version_hpc_cluster="9.10.1.4"
