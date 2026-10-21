#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CUR_DIR="$(pwd)"

source "${SCRIPT_DIR}/utils.bash"

#---------------HERE STARTS THE SCRIPT---------------

help() {
    echo -e "${BOLD}Usage:${GREEN} $(basename "$0") ${RESET}${RED}<python_venv_path>${RESET}"
    exit 1
}

if [ "$#" -ne 1 ]; then
    help
fi

VENV_DIR="$1"

#Reset venv
rm -rf "${VENV_DIR}"
mkdir "${VENV_DIR}"

#python version needs to be 3.13
python3.13 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

#Install needed pip
pip install -r requirements.txt
pip install -e "${SCRIPT_DIR}/../chemtrain"
pip install -e "${SCRIPT_DIR}/../chemtrain[test]"
pip install -e "${SCRIPT_DIR}/../tensorial[dev]"
pip install "${SCRIPT_DIR}/tensorial[docs]"
pip install sphinx_autodoc_typehints sphinx-paramlinks #Mising doc dependencies in tensorial





