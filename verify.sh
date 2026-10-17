#!/usr/bin/env bash
# Check that the Miles venv built by install.sh imports and can use the GPU.
set -euo pipefail

PREFIX=${PREFIX:-$HOME/.venvs/miles}

# run from a neutral dir so a local package dir can't shadow the installed one
cd "$(mktemp -d)"
"$PREFIX/bin/python" -W ignore::FutureWarning - <<'EOF'
import torch, sglang, megatron.core, transformer_engine.pytorch, torch_memory_saver, miles
from importlib.metadata import version
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| device", torch.cuda.get_device_name(0),
      torch.cuda.get_device_capability())
print("sglang", version("sglang"))  # the package has no top-level __version__
print("transformer_engine", version("transformer_engine"), "| megatron-core", version("megatron-core"))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); print("matmul ok", (x @ x).shape)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn import failed (use --attention-backend auto):", e)
print("Miles import OK")
EOF
