#!/usr/bin/env bash
# smoke-small-qwen35.sh on GSM8K instead of dapo-math: Qwen3.5-0.8B never finished a dapo problem
# within 4k tokens, so every reward was 0. GSM8K's grade-school word problems should be solvable
# in a few hundred tokens. Pass --model-name Qwen3.5-2B (with its own --output-dir) for the 2B.
# Needs GSM8K downloaded (README, Quick Start).
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")

# full --extra-args: passing it replaces smoke-small-qwen35.sh's; the later --prompt-data and keys
# override the recipe's dapo-math ones
exec "$HERE/smoke.sh" --model-name Qwen3.5-0.8B --output-dir "$HERE/runs/smoke-small-qwen35-gsm8k" \
  --extra-args "--prompt-data $HERE/datasets/gsm8k/train.parquet --input-key messages --label-key label \
--rollout-max-response-len 2048 --rollout-batch-size 8 --global-batch-size 64 --sglang-mem-fraction-static 0.7 \
--apply-chat-template-kwargs '{\"enable_thinking\": false}' --rm-type math" \
  "$@"
