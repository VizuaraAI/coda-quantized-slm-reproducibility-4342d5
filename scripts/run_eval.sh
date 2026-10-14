#!/usr/bin/env bash
# Traces to tables to figures.
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="${1:-runs/latest}"
REFERENCE="${REFERENCE:-ref}"

python -m coda.evaluate --run-dir "$RUN_DIR" --reference "$REFERENCE"
python -m coda.plots    --run-dir "$RUN_DIR" --reference "$REFERENCE" --panel all

echo
echo "tables : $RUN_DIR/table_main.csv, $RUN_DIR/table_ablation.csv"
echo "figure : $RUN_DIR/fig_all.png"
echo
echo "Copy the table rows into experiments/log_template.md before you change"
echo "anything. A result you did not write down is a result you will redo."
