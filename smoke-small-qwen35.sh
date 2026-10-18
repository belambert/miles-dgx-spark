#!/usr/bin/env bash
# smoke-small.sh with Qwen3.5-0.8B: no thinking, 2k-token responses, math reward. Pass
# --model-name Qwen3.5-2B to run the 2B instead. Needs the model downloaded and converted
# (README, Quick Start).
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")

exec "$HERE/smoke-small.sh" --model-name Qwen3.5-0.8B --output-dir "$HERE/runs/smoke-small-qwen35" "$@"
