#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CUR_DIR="$(pwd)"

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

#---------------HERE STARTS THE SCRIPT---------------

help() {
    echo -e "${BOLD}Usage:${GREEN} $(basename "$0") ${RESET}${RED}<python venv path>${RESET}"
    exit 1
}

if [ "$#" -ne 1 ]; then
    help
fi

VENV_DIR="$1"

#Reset venv
rm -rf "${VENV_DIR}"
mkdir "${VENV_DIR}"
python3.13 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

#Install needed pip
pip install -e "${SCRIPT_DIR}/chemtrain"
pip install -e "${SCRIPT_DIR}/tensorial[dev]"





