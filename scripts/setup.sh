#!/usr/bin/env bash
# Create the virtualenv and install coda in editable mode.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "creating .venv with $PYTHON"
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

echo
echo "done. activate with:  source .venv/bin/activate"
echo "then run:             bash scripts/run_smoke.sh"
