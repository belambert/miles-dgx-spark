#!/usr/bin/env bash
# smoke.sh scaled down so the KV cache never fills: 64 samples per rollout, 2k-token responses.
# The full-size batch needs ~2M tokens of KV cache and takes 40+ min per rollout on one Spark.
set -euo pipefail

# replaces smoke.sh's --extra-args, so the attention backend is repeated here
exec "$(dirname "$(readlink -f "$0")")/smoke.sh" \
  --extra-args "--attention-backend auto --rollout-max-response-len 2048 --rollout-batch-size 8 --global-batch-size 64 --sglang-mem-fraction-static 0.7" \
  "$@"
