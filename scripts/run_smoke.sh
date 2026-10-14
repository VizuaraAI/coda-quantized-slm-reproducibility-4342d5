#!/usr/bin/env bash
# End-to-end check in about a minute. No phone, no model download.
#
# What you should see: cpu_i8mm and npu_pertensor fall well below 100%
# agreement with the FP64 reference, even though every backend loaded
# byte-identical weights. That is the paper's claim, in miniature.
#
# cpu_neon, which differs only in reduction order, will probably sit at
# 100%, and that is the correct answer rather than a bug: see the README for
# why reordering a float sum is too small an effect to flip an int8 argmax.
#
# If EVERY agreement rate is exactly 100%, including npu_pertensor, then the
# presets are not actually differing and the harness is not yet testing
# anything. Check kernels.KERNEL_PRESETS.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m coda.run_audit \
  --config configs/default.yaml \
  --backends ref,cpu_neon,cpu_i8mm,npu_pertensor \
  --n-per-task 2 \
  --max-new-tokens 24 \
  --repeats 2

python -m coda.evaluate --run-dir runs/latest --reference ref
