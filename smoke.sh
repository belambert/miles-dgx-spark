#!/usr/bin/env bash
# Short Qwen3-0.6B GRPO run to check the install end to end: 20 rollouts, no AIME eval.
# Needs the model downloaded and converted (README, Quick Start). Extra args go to the launcher.
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")
PREFIX=${PREFIX:-$HERE/.venv}
SRC=${SRC:-$HERE/miles-src}

cd "$SRC/miles"
"$PREFIX/bin/python" scripts/run_qwen3_dense.py --model-name Qwen3-0.6B --num-gpus-per-node 1 \
  --model-dir "$HERE/models" --data-dir "$HERE/datasets" --output-dir "$HERE/runs" \
  --megatron-path "$SRC/Megatron-LM" \
  --num-rollout 20 --no-enable-eval \
  "$@"
