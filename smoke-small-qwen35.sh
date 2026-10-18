#!/usr/bin/env bash
# smoke-small.sh with Qwen3.5-0.8B and a 4k response cap: no thinking, math reward. Even without
# thinking, Qwen3.5 writes long step-by-step solutions; at 2k, 92-98% were cut off and every
# reward was 0. Pass --model-name Qwen3.5-2B to run the 2B instead. Needs the model downloaded and
# converted (README, Quick Start).
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")

# full --extra-args: passing it replaces smoke-small.sh's
exec "$HERE/smoke.sh" --model-name Qwen3.5-0.8B --output-dir "$HERE/runs/smoke-small-qwen35" \
  --extra-args "--rollout-max-response-len 4096 --rollout-batch-size 8 --global-batch-size 64 --sglang-mem-fraction-static 0.7 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --rm-type math" \
  "$@"
