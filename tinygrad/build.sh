#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

cmake -S . -B build
cmake --build ./build
./build/VampireGrad "${1:-$SCRIPT_DIR/../model_data.h}"