#!/usr/bin/env bash
# smoke-thinking-qwen35.sh with a 16k response cap: at 8k, 94% of Qwen3.5-0.8B's answers were
# cut off mid-thought, so every reward was 0. Log-probs are computed in 4k-token chunks: unchunked,
# one 16k sequence needs 16384 x 248320 vocab x 4 B = 15 GiB of fp32 logits, which ran out of memory.
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")

# full --extra-args: passing it replaces smoke-thinking.sh's
exec "$HERE/smoke.sh" --model-name Qwen3.5-0.8B --output-dir "$HERE/runs/smoke-thinking-qwen35-16k" \
  --extra-args "--rollout-batch-size 8 --global-batch-size 64 --sglang-mem-fraction-static 0.7 --rollout-max-response-len 16384 --log-probs-chunk-size 4096" \
  "$@"
