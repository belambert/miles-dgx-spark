#!/usr/bin/env bash
# smoke-small.sh with Qwen3.5-4B: no thinking, 2k-token responses, math reward. The upstream
# recipe is for 8 GPUs, so this overrides it to TP 1. On the GB10, CPU and GPU share one memory
# pool, so offloading the trainer between steps frees nothing: SGLang drops to 0.3 to leave room
# for the trainer's ~16 B/param of weights, grads and Adam state. Needs Qwen3.5-4B downloaded and
# converted (README, Quick Start).
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")

# full --extra-args: passing it replaces smoke-small.sh's
exec "$HERE/smoke.sh" --model-name Qwen3.5-4B --output-dir "$HERE/runs/smoke-small-qwen35-4b" \
  --extra-args "--tensor-model-parallel-size 1 --rollout-max-response-len 2048 --rollout-batch-size 8 --global-batch-size 64 --sglang-mem-fraction-static 0.3 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --rm-type math" \
  "$@"
