#!/usr/bin/env bash
# smoke-thinking-qwen35.sh with a 16k response cap: at 8k, 94% of Qwen3.5-0.8B's answers were
# cut off mid-thought, so every reward was 0.
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")

# full --extra-args: passing it replaces smoke-thinking.sh's
exec "$HERE/smoke.sh" --model-name Qwen3.5-0.8B --output-dir "$HERE/runs/smoke-thinking-qwen35-16k" \
  --extra-args "--rollout-batch-size 8 --global-batch-size 64 --sglang-mem-fraction-static 0.7 --rollout-max-response-len 16384" \
  "$@"
