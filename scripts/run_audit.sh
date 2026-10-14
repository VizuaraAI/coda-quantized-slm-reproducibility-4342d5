#!/usr/bin/env bash
# The full matrix at the configured size.
#
# Pass device names to restrict the run, e.g.
#   bash scripts/run_audit.sh ref,cpu_i8mm,npu_pertensor
set -euo pipefail

cd "$(dirname "$0")/.."

BACKENDS="${1:-}"
CONFIG="${CONFIG:-configs/default.yaml}"

if [ -n "$BACKENDS" ]; then
  python -m coda.run_audit --config "$CONFIG" --backends "$BACKENDS"
else
  python -m coda.run_audit --config "$CONFIG"
fi

echo
echo "traces written. Now run:  bash scripts/run_eval.sh"
