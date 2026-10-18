#!/usr/bin/env bash
# smoke-thinking.sh with Qwen3.5-0.8B: 18 of its 24 layers are linear attention with a fixed-size
# state, so its KV cache is ~9x smaller per token than Qwen3-0.6B's, which should speed up long
# thinking rollouts. Needs Qwen3.5-0.8B downloaded and converted (README, Quick Start).
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")

exec "$HERE/smoke-thinking.sh" --model-name Qwen3.5-0.8B --output-dir "$HERE/runs/smoke-thinking-qwen35" "$@"
