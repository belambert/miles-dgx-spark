#!/usr/bin/env bash
# smoke.sh scaled down so the KV cache never fills: 64 samples per rollout, 2k-token responses.
# The full-size batch needs ~2M tokens of KV cache and takes 40+ min per rollout on one Spark.
# Qwen3's thinking is off: thinking runs past 2k tokens, so every answer was truncated and scored 0.
# That needs the math reward: deepscaler only grades text after </think>, which is now in the prompt.
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")

# own output dir: a run resumes from whatever checkpoint is there, and this setup differs from smoke.sh
exec "$HERE/smoke.sh" --output-dir "$HERE/runs/smoke-small" \
  --extra-args "--rollout-max-response-len 2048 --rollout-batch-size 8 --global-batch-size 64 --sglang-mem-fraction-static 0.7 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --rm-type math" \
  "$@"
