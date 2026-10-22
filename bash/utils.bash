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

#Lib Versions
cuda_version="12.9.2"
cudnn_version="9.10.1"
cuda_compute_version="9.0"

cuda_version_hpc_cluster="12.9.41.patched"
cudnn_version_hpc_cluster="9.10.1.4"

python_version="python3"
