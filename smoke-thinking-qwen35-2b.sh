#!/usr/bin/env bash
# smoke-thinking-qwen35-16k.sh with Qwen3.5-2B: same layout and KV cache per token as the 0.8B,
# with twice the hidden size. Needs Qwen3.5-2B downloaded and converted (README, Quick Start).
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")

exec "$HERE/smoke-thinking-qwen35-16k.sh" --model-name Qwen3.5-2B --output-dir "$HERE/runs/smoke-thinking-qwen35-2b" "$@"
