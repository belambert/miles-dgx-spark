#!/usr/bin/env bash
# Non-Docker Miles install for DGX Spark (GB10 / sm_121, aarch64, CUDA 13, Python 3.12).
# Mirrors radixark/miles docker/Dockerfile step by step, minus the base image and
# the model-specific extras that Qwen3 dense doesn't need.
#
# Before running: compare the pins below with `git show <tag>:docker/Dockerfile`
# and the release notes for the tag you target. They move between releases.
set -euo pipefail

# ---------------------------------------------------------------- pins (match radixark/miles:latest as run on this Spark, 2026-10-17)
MILES_REF=${MILES_REF:-e1a791d3e3}            # /root/miles in the image
SGLANG_COMMIT=${SGLANG_COMMIT:-74bd569} # sglang-miles branch on sgl-project/sglang
MEGATRON_COMMIT=${MEGATRON_COMMIT:-9df207fd1} # radixark/Megatron-LM, miles-main
WHEELS_REPO=yueming-yuan/miles-wheels
WHEELS_TAG=cu130-aarch64
TMS_COMMIT=b5588e83de86412a48689a6583a4b567e75f7acc      # torch_memory_saver
MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
BRIDGE_COMMIT=7f0fb3456f8ffe47599b5fd167b454605d85f932   # radixark/Megatron-Bridge
CUDNN_VERSION=9.24.0.43
TVM_FFI_VERSION=0.1.11   # must match sglang's own pin; check python/pyproject.toml
# torch must match what the prebuilt wheels (TE, flash-attn, apex) were built against,
# i.e. the torch in the docker image -- NOT sglang-miles' pyproject pin. Get them with:
#   docker run --rm radixark/miles:<tag> pip freeze | grep -E '^torch(vision|audio)?=='
TORCH_VERSION=${TORCH_VERSION:-2.14.1+cu130}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.29.1+cu130}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.11.0+cu130}

# ---------------------------------------------------------------- locations
PREFIX=${PREFIX:-$HOME/.venvs/miles}   # keep separate from the SGLang/vLLM serving envs
SRC=${SRC:-$HOME/miles-src}
WHEELS=$SRC/wheels

export TORCH_CUDA_ARCH_LIST="12.1"     # only matters for anything built from source
export MAX_JOBS=${MAX_JOBS:-8}         # source builds on 20 ARM cores + unified memory: don't go wild

mkdir -p "$SRC" "$WHEELS"
cd "$SRC"

# ---------------------------------------------------------------- 0. Rust toolchain (sglang builds Rust extensions;
# the docker base image ships one at /root/.cargo/bin)
if ! command -v cargo >/dev/null 2>&1; then
  [ -x "$HOME/.cargo/bin/cargo" ] || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
cargo --version

# ---------------------------------------------------------------- 1. venv
uv venv --seed --allow-existing --python 3.12 "$PREFIX"
# shellcheck disable=SC1091
source "$PREFIX/bin/activate"
python -m pip install -U pip wheel setuptools

# ---------------------------------------------------------------- 2. sources
[ -d miles ] || git clone https://github.com/radixark/miles.git
git -C miles fetch --tags && git -C miles checkout "$MILES_REF"

[ -d sglang ] || git clone https://github.com/sgl-project/sglang.git
git -C sglang fetch origin sglang-miles && git -C sglang checkout -f "$SGLANG_COMMIT"

[ -d Megatron-LM ] || git clone --recursive -b miles-main https://github.com/radixark/Megatron-LM.git
git -C Megatron-LM checkout "$MEGATRON_COMMIT" && git -C Megatron-LM submodule update --init --recursive

# ---------------------------------------------------------------- 3. torch + sglang (replaces the lmsysorg/sglang base image)
# sglang's pyproject pins torch; the cu130 index supplies the CUDA 13 aarch64 build.
pip install -e "sglang/python[all]" --extra-index-url https://download.pytorch.org/whl/cu130
# Replace the torch sglang's pin pulled in with the image's torch (the image installs
# sglang --no-deps on top of the base image's torch).
pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
  --extra-index-url https://download.pytorch.org/whl/cu130
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability())
assert torch.version.cuda.startswith("13"), "need a CUDA 13 torch build"
EOF

# ---------------------------------------------------------------- 4. prebuilt aarch64 wheels (flash-attn, TE, apex, router, ...)
curl -sL "https://api.github.com/repos/${WHEELS_REPO}/releases/tags/${WHEELS_TAG}" \
| python3 -c "import sys,json,subprocess,os; w='$WHEELS'; \
[subprocess.run(['curl','-fSL','-o',os.path.join(w,a['name']),a['browser_download_url']],check=True) \
 for a in json.load(sys.stdin).get('assets',[]) \
 if a['name'].endswith(('.whl','.tar.gz')) and not os.path.exists(os.path.join(w,a['name']))]"
ls -lh "$WHEELS"

# ---------------------------------------------------------------- 5. python deps, in Dockerfile order
pip install "$WHEELS"/flash_attn-*.whl
# FA3 (flash_attn_3) skipped: Hopper-only.
pip install "git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT}" --no-deps
pip install flash-linear-attention==0.4.2
# Skipped (model-specific): tilelang/tile_kernels (DeepSeek-V4), FlashQLA (Qwen GDN),
# fast-hadamard-transform, causal-conv1d/mamba-ssm (Nemotron-H).

# TransformerEngine 2.17, exact wheels, no deps, then the cu13 patches
pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-cu13 transformer-engine-torch || true
pip install --force-reinstall --no-deps \
  "$WHEELS"/transformer_engine-2.17.0-py3-none-any.whl \
  "$WHEELS"/transformer_engine_cu13-2.17.0-py3-none-manylinux_2_28_*.whl \
  "$WHEELS"/transformer_engine_torch-2.17.0-cp312-cp312-linux_*.whl
pip install einops onnx onnxscript pydantic nvdlfw-inspect
TE_DIR=$(python -c 'import importlib.util; print(importlib.util.find_spec("transformer_engine").submodule_search_locations[0])')
for p in miles/docker/patch/cu13/*.patch; do
  [ -e "$p" ] || continue
  echo "Applying $(basename "$p")"
  patch -d "$TE_DIR" -p1 < "$p"
done

pip install "$WHEELS"/apex-*.whl

pip install -e Megatron-LM
pip install "git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.1.0"

# --no-deps added vs the Dockerfile so --force-reinstall can't swap torch underneath us
TMS_CUDA_MAJOR=13 pip install "git+https://github.com/fzyzcjy/torch_memory_saver.git@${TMS_COMMIT}" \
  --no-cache-dir --force-reinstall --no-deps
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
pip install "git+https://github.com/radixark/Megatron-Bridge.git@${BRIDGE_COMMIT}" --no-deps --no-build-isolation
pip install megatron-energon --no-deps
pip install multi-storage-client --no-deps
pip install -r miles/requirements.txt

# cuDNN from pip. The image purges the apt cuDNN because mixing the two breaks TE;
# we don't touch the host's packages, so watch for cuDNN sub-library errors.
pip install "nvidia-cudnn-cu13==${CUDNN_VERSION}"

# Mooncake (P2P / disk-delta weight transfer; harmless if unused)
pip install --force-reinstall --no-deps "$WHEELS"/mooncake_transfer_engine*-*.whl

# ---------------------------------------------------------------- 6. miles itself
pip install -e miles --no-deps
pip install "$WHEELS"/fake_int4_quant_cuda-*.whl

# ---------------------------------------------------------------- 7. router (rollout requests go through it)
pip install --force-reinstall "$WHEELS"/sglang_router-*.whl
tar xzf "$WHEELS"/sgl-model-gateway-linux-*.tar.gz -C "$PREFIX/bin/"
chmod +x "$PREFIX/bin/sgl-model-gateway"

# ---------------------------------------------------------------- 8. re-pin tvm-ffi last (something above drags in an older one)
pip install --force-reinstall --no-deps "apache-tvm-ffi==${TVM_FFI_VERSION}"

# ---------------------------------------------------------------- 9. verify
python - <<'EOF'
import torch, sglang, megatron.core, transformer_engine.pytorch, torch_memory_saver, miles
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| device", torch.cuda.get_device_name(0),
      torch.cuda.get_device_capability())
print("sglang", sglang.__version__)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); print("matmul ok", (x @ x).shape)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn import failed (use --attention-backend auto):", e)
print("Miles import OK")
EOF

echo
echo "Done. Activate with: source $PREFIX/bin/activate"
echo "Launch with: cd $SRC/miles && python scripts/run_qwen3_dense.py --megatron-path $SRC/Megatron-LM --model-dir <models> --data-dir <datasets> ..."
