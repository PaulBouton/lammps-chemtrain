#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CUR_DIR="$(pwd)"

source "${SCRIPT_DIR}/utils.bash"

#Exit on Error
set -e

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

#Use a custom build script to use a recent version of bazel
cp ${SCRIPT_DIR}/../python/chemtrain-deploy-bazel-build.py ${SCRIPT_DIR}/../chemtrain/chemtrain-deploy/
mv ${SCRIPT_DIR}/../chemtrain/chemtrain-deploy/chemtrain-deploy-bazel-build.py ${SCRIPT_DIR}/../chemtrain/chemtrain-deploy/build.py 

cd ${SCRIPT_DIR}/../chemtrain/chemtrain-deploy

if [[ "$TARGET_ARCH" == "cpu" ]]; then
    python build.py \
      --build_cpu_pjrt_plugin \
      --target_cpu_features=default \
      --output_path="$PWD/out" \
      --install_location="$PWD/lib"
elif [[ "$TARGET_ARCH" == "cuda" ]]; then
    python build.py \
      --bazel_options=--jobs=$(( n_cores * 2 )) \
      --bazel_options=--local_resources=cpu=$(( n_cores )) \
      --bazel_options=--disk_cache=$PWD \
      --enable_cuda \
      --build_gpu_pjrt_plugin \
      --cuda_version=${cuda_version} \
      --cudnn_version=${cudnn_version} \
      --cuda_compute_capabilities=${cuda_compute_version} \
      --target_cpu_features=default \
      --output_path="$PWD/out" \
      --install_location="$PWD/lib"
else
    git checkout ${SCRIPT_DIR}/../chemtrain
    echo -e "Invalid target Architecture!"
    exit 1
fi

git checkout ${SCRIPT_DIR}/../chemtrain
cd ${CUR_DIR}
