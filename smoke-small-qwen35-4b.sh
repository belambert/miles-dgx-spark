#!/usr/bin/env bash
# smoke-small.sh with Qwen3.5-4B: no thinking, math reward. The upstream recipe is for 8 GPUs, so
# this overrides it to TP 1. On the GB10, CPU and GPU share one memory pool, so offloading the
# trainer between steps frees nothing, and the first attempt ran out of memory before training.
# Needs Qwen3.5-4B downloaded and converted (README, Quick Start).
#
# - 4k responses: at 2k, 72% of answers were cut off
# - SGLang at 0.3 and no prefix cache: the cache kept ~3 linear-attention state slots per request,
#   capping concurrency at 19 of 64; prefix hits were only 4% anyway
# - smaller micro-batches, chunked log-probs and bf16 Adam moments to fit training
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")

# full --extra-args: passing it replaces smoke-small.sh's
exec "$HERE/smoke.sh" --model-name Qwen3.5-4B --output-dir "$HERE/runs/smoke-small-qwen35-4b" \
  --extra-args "--tensor-model-parallel-size 1 --rollout-max-response-len 4096 --rollout-batch-size 8 --global-batch-size 64 \
--sglang-mem-fraction-static 0.3 --sglang-disable-radix-cache \
--max-tokens-per-gpu 4096 --log-probs-chunk-size 2048 \
--use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16 \
--apply-chat-template-kwargs '{\"enable_thinking\": false}' --rm-type math" \
  "$@"
