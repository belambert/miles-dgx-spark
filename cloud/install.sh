#!/usr/bin/env bash
# Non-Docker Miles install for a cloud GPU machine (x86_64 or aarch64 Linux, e.g. one H100), with
# CUDA 13 and Python 3.12; ../install.sh is the DGX Spark version. Mirrors radixark/miles
# docker/Dockerfile step by step, minus the base image and the model-specific extras that Qwen3
# dense doesn't need. TE, apex and flash-attn come from the miles-wheels release the image installs.
#
# Needs an NVIDIA driver >= 580 and a CUDA 13 toolkit (nvcc) in /usr/local/cuda or on PATH:
# torch_memory_saver builds against it, and SGLang compiles some kernels at runtime.
#
# The commit and version pins are in ../pins.sh, shared with the Spark install.sh.
set -euo pipefail

HERE=$(dirname "$(dirname "$(readlink -f "$0")")")  # the repo root, one up from cloud/
source "$HERE/pins.sh"
# the release MILES_IMAGE's Dockerfile installs (WHEELS_TAG_X86 / WHEELS_TAG_ARM64); its TE,
# apex and flash-attn are byte-identical to the image's
WHEELS_REPO=radixark/miles-wheels
WHEELS_TAG=${WHEELS_TAG:-cu130-torch214-$(uname -m)}

# ---------------------------------------------------------------- locations
# everything lives in the repo root, where run.sh looks; paths are absolute because the venv
# and the editable installs record them, so moving the repo means reinstalling
PREFIX=${PREFIX:-$HERE/.venv}
SRC=${SRC:-$HERE/miles-src}
WHEELS=$SRC/wheels/$WHEELS_TAG

# ---------------------------------------------------------------- filesystems that can't chmod
# Some cloud containers mount /workspace as a FUSE volume over object storage (geesefs) that
# can't set permissions, and point UV_CACHE_DIR/PIP_CACHE_DIR at it; uv then fails extracting
# pip into the venv. Move such caches to local disk; the venv itself can't live there either.
can_chmod() {
  local f
  mkdir -p "$1" 2>/dev/null && f=$(mktemp -p "$1" 2>/dev/null) || return 1
  chmod +x "$f" 2>/dev/null; local rc=$?
  rm -f "$f"
  return $rc
}
for var in UV_CACHE_DIR PIP_CACHE_DIR; do
  if [ -n "${!var:-}" ] && ! can_chmod "${!var}"; then
    tool=${var%%_*}; local_dir=$HOME/.cache/${tool,,}
    echo "$var=${!var} can't set file permissions; using $local_dir" >&2
    export "$var=$local_dir"
  fi
done
can_chmod "$PREFIX" || { echo "PREFIX=$PREFIX can't set file permissions, so it can't hold a venv; set PREFIX and SRC to local disk, e.g. PREFIX=/root/miles/.venv SRC=/root/miles/miles-src" >&2; exit 1; }

# ---------------------------------------------------------------- host checks (a cloud image may ship CUDA 12)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
[ "${driver%%.*}" -ge 580 ] || { echo "NVIDIA driver $driver is too old for CUDA 13; need >= 580" >&2; exit 1; }
export PATH=$PATH:/usr/local/cuda/bin
nvcc --version | grep -q "release 13\." || { echo "need a CUDA 13 toolkit (nvcc) in /usr/local/cuda or on PATH" >&2; exit 1; }
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")}

# only matters for anything built from source: 9.0 on an H100
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)}
export MAX_JOBS=${MAX_JOBS:-8}

mkdir -p "$SRC" "$WHEELS"
cd "$SRC"

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }

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
# pinned commits mean detached HEADs; skip git's multi-line advice about it
[ -d miles ] || git clone "$MILES_REPO" miles
git -C miles remote set-url origin "$MILES_REPO"  # older clones point at radixark
git -C miles fetch --tags origin && git -C miles -c advice.detachedHead=false checkout "$MILES_REF"

[ -d sglang ] || git clone https://github.com/sgl-project/sglang.git
git -C sglang fetch origin sglang-miles && git -C sglang -c advice.detachedHead=false checkout -f "$SGLANG_COMMIT"

[ -d Megatron-LM ] || git clone --recursive -b miles-main https://github.com/radixark/Megatron-LM.git
git -C Megatron-LM -c advice.detachedHead=false checkout "$MEGATRON_COMMIT" && git -C Megatron-LM submodule update --init --recursive

# ---------------------------------------------------------------- 3. torch + sglang (replaces the lmsysorg/sglang base image)
# sglang's pyproject pins torch; the cu130 index supplies the CUDA 13 build.
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

# ---------------------------------------------------------------- 4. prebuilt wheels (TE, apex, flash-attn, router, ...)
curl -sL "https://api.github.com/repos/${WHEELS_REPO}/releases/tags/${WHEELS_TAG}" \
| python3 -c "import sys,json,subprocess,os; w='$WHEELS'; \
[subprocess.run(['curl','-fSL','-o',os.path.join(w,a['name']),a['browser_download_url']],check=True) \
 for a in json.load(sys.stdin).get('assets',[]) \
 if a['name'].endswith(('.whl','.tar.gz')) and not os.path.exists(os.path.join(w,a['name']))]"
ls -lh "$WHEELS"

# ---------------------------------------------------------------- 5. python deps, in Dockerfile order
pip install "git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT}" --no-deps
pip install flash-linear-attention==0.5.2
# Skipped (model-specific): tilelang/tile_kernels (DeepSeek-V4), FlashQLA (Qwen GDN),
# fast-hadamard-transform, causal-conv1d/mamba-ssm (Nemotron-H).

# TransformerEngine: the image's 2.17.0+miles build (radixark/TransformerEngine, matching its
# torch). --no-deps, as in the Dockerfile, so pip can't pull another TE core or swap torch.
pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-cu13 transformer-engine-torch || true
pip install --force-reinstall --no-deps "$WHEELS"/transformer_engine{,_cu13,_torch}-2.17.0+miles-*.whl \
  "$WHEELS"/apex-*.whl "$WHEELS"/flash_attn-*.whl
# FA3 is Hopper-only; TE prefers it over FA4 on sm_90
if [[ $TORCH_CUDA_ARCH_LIST == 9.* ]]; then pip install --force-reinstall --no-deps "$WHEELS"/flash_attn_3-*.whl; fi

# TE picks FlashAttention 4 (pulled in by sglang) whenever it's installed, and passes it the
# FA2-style causal window (-1, 0). Since FA4 4.0.0b30 that means an empty local window, so
# attention comes out wrong (NVIDIA/TransformerEngine#3528, fix pending in #3570). Skip FA4 on
# every GPU so TE uses FA3 on Hopper and FA2 elsewhere. Fails loudly if the TE source moves.
python - <<'EOF'
import sysconfig
# by path: importing TE before torch can't find libnccl
path = f"{sysconfig.get_paths()['purelib']}/transformer_engine/pytorch/attention/dot_product_attention/utils.py"
anchor = '            logger.debug("Disabling FlashAttention 4 for compute capability < sm80")\n        use_flash_attention_4 = False\n'
marker = "# miles-dgx-spark: skip FA4"
patch = f"    use_flash_attention_4 = False  {marker}\n"
src = open(path).read()
if marker not in src:
    assert src.count(anchor) == 1, f"FA4 patch anchor not found in {path}"
    open(path, "w").write(src.replace(anchor, anchor + patch))
print("TE: FA4 disabled")
EOF
pip install einops onnx onnxscript pydantic nvdlfw-inspect

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
# Mooncake skipped, as in the Dockerfile: only optional weight-transfer paths import it

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
PREFIX=$PREFIX bash "$HERE/verify.sh"  # bash: the checkout may have lost its exec bits

echo
echo "Done. Activate with: source $PREFIX/bin/activate"
echo "Then follow Quick Start in the README to train Qwen3-0.6B."
