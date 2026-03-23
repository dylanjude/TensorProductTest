#!/bin/bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install triton
pip install numba
echo ""
echo "Done. Activate with:  source .venv/bin/activate"
