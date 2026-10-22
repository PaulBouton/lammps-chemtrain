#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CUR_DIR="$(pwd)"

cp ${SCRIPT_DIR}/../hooks/commit-msg ${SCRIPT_DIR}/../.git/hooks
