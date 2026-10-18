#!/usr/bin/env bash
# smoke.sh with smoke-small.sh's batch but Qwen3's thinking on: 64 samples per rollout, 8k-token
# responses, deepscaler reward. Checks the thinking path (long sequences in training) in minutes per
# rollout instead of the full batch's 40+.
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")

# own output dir: a run resumes from whatever checkpoint is there
exec "$HERE/smoke.sh" --output-dir "$HERE/runs/smoke-thinking" \
  --extra-args "--rollout-batch-size 8 --global-batch-size 64 --sglang-mem-fraction-static 0.7" \
  "$@"
