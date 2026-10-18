#!/usr/bin/env bash
# Non-Docker Miles install for DGX Spark (GB10 / sm_121, aarch64, CUDA 13, Python 3.12).
# Mirrors radixark/miles docker/Dockerfile step by step, minus the base image and
# the model-specific extras that Qwen3 dense doesn't need. Needs the image pulled
# locally: TE, apex and flash-attn are copied out of it (nothing runs in a container).
#
# Before running: compare the pins below with `git show <tag>:docker/Dockerfile`
# and the release notes for the tag you target. They move between releases.
set -euo pipefail

# ---------------------------------------------------------------- pins (match radixark/miles:latest as run on this Spark, 2026-10-17)
# Miles has no release matching the image: /root/miles there is e1a791d3e3, an untagged
# origin/main commit (2026-10-16) whose package reports 0.1.2. The v0.1.2 tag (a22fd82419)
# is not its ancestor, so don't swap in a tag. Unlike older refs, this commit has no
# docker/patch/cu13/; its TE patches only exist baked into the image's build.
# We use the dgx-spark branch of our fork: that commit plus single-GPU Qwen3-0.6B,
# Qwen3.5-0.8B and Qwen3.5-2B recipes (and a Qwen3.5-2B model config) and an --attention-backend flag.
MILES_REPO=${MILES_REPO:-https://github.com/belambert/miles.git}
MILES_REF=${MILES_REF:-7cc3b8e2a9}            # dgx-spark, on top of e1a791d3e3 (/root/miles in the image)
SGLANG_COMMIT=${SGLANG_COMMIT:-74bd569} # sglang-miles branch on sgl-project/sglang
MEGATRON_COMMIT=${MEGATRON_COMMIT:-9df207fd1} # radixark/Megatron-LM, miles-main
WHEELS_REPO=yueming-yuan/miles-wheels
WHEELS_TAG=cu130-aarch64
TMS_COMMIT=b5588e83de86412a48689a6583a4b567e75f7acc      # torch_memory_saver
MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
BRIDGE_COMMIT=7f0fb3456f8ffe47599b5fd167b454605d85f932   # radixark/Megatron-Bridge
CUDNN_VERSION=9.24.0.43
TVM_FFI_VERSION=0.1.11   # must match sglang's own pin; check python/pyproject.toml
# compiled extensions are copied from this image, so they match its torch exactly
# pinned by digest: this is the :latest that every pin in this block was read from
MILES_IMAGE=${MILES_IMAGE:-radixark/miles@sha256:ad52af8b83076331d43e65a7a71ecc1daf128512b82829701e098540fcd80478}
IMAGE_DISTS="transformer_engine transformer_engine_cu13 transformer_engine_torch apex flash_attn"
# torch must match what the prebuilt wheels (TE, flash-attn, apex) were built against,
# i.e. the torch in the docker image -- NOT sglang-miles' pyproject pin. Get them with:
#   docker run --rm radixark/miles:<tag> pip freeze | grep -E '^torch(vision|audio)?=='
TORCH_VERSION=${TORCH_VERSION:-2.14.1+cu130}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.29.1+cu130}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.11.0+cu130}

# ---------------------------------------------------------------- locations
# everything lives next to this script; paths are absolute because the venv and
# the editable installs record them, so moving this folder means reinstalling
HERE=$(dirname "$(readlink -f "$0")")
PREFIX=${PREFIX:-$HERE/.venv}          # keep separate from the SGLang/vLLM serving envs
SRC=${SRC:-$HERE/miles-src}
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
# pinned commits mean detached HEADs; skip git's multi-line advice about it
[ -d miles ] || git clone "$MILES_REPO" miles
git -C miles remote set-url origin "$MILES_REPO"  # older clones point at radixark
git -C miles fetch --tags origin && git -C miles -c advice.detachedHead=false checkout "$MILES_REF"

[ -d sglang ] || git clone https://github.com/sgl-project/sglang.git
git -C sglang fetch origin sglang-miles && git -C sglang -c advice.detachedHead=false checkout -f "$SGLANG_COMMIT"

[ -d Megatron-LM ] || git clone --recursive -b miles-main https://github.com/radixark/Megatron-LM.git
git -C Megatron-LM -c advice.detachedHead=false checkout "$MEGATRON_COMMIT" && git -C Megatron-LM submodule update --init --recursive

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

# ---------------------------------------------------------------- 4. prebuilt aarch64 wheels (mooncake, router, ...)
curl -sL "https://api.github.com/repos/${WHEELS_REPO}/releases/tags/${WHEELS_TAG}" \
| python3 -c "import sys,json,subprocess,os; w='$WHEELS'; \
[subprocess.run(['curl','-fSL','-o',os.path.join(w,a['name']),a['browser_download_url']],check=True) \
 for a in json.load(sys.stdin).get('assets',[]) \
 if a['name'].endswith(('.whl','.tar.gz')) and not os.path.exists(os.path.join(w,a['name']))]"
ls -lh "$WHEELS"

# ---------------------------------------------------------------- 5. python deps, in Dockerfile order
# FA3 (flash_attn_3) skipped: Hopper-only.
pip install "git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT}" --no-deps
pip install flash-linear-attention==0.5.2
# Skipped (model-specific): tilelang/tile_kernels (DeepSeek-V4), FlashQLA (Qwen GDN),
# fast-hadamard-transform, causal-conv1d/mamba-ssm (Nemotron-H).

# TransformerEngine, apex, flash-attn: copied out of the image. The release's TE 2.17.0
# wheel was built against an older torch (undefined c10 symbols on import); the image
# has its own patched 2.17.0+miles build matching its torch. Docker is only a file source.
pip uninstall -y $IMAGE_DISTS transformer-engine-cu12 || true
docker run --rm -i --user "$(id -u):$(id -g)" -v "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')":/dst \
  "$MILES_IMAGE" python - $IMAGE_DISTS <<'EOF'
import os, shutil, sys
from importlib.metadata import distribution
for name in sys.argv[1:]:
    d = distribution(name)
    root, n = d.locate_file(""), 0
    for f in map(str, d.files):
        src = os.path.join(root, f)
        if f.startswith("..") or not os.path.isfile(src):  # skip console scripts outside site-packages
            continue
        os.makedirs(os.path.dirname(out := os.path.join("/dst", f)), exist_ok=True)
        shutil.copy2(src, out); n += 1
    print(f"{name} {d.version}: {n} files")
EOF

# TE picks FlashAttention 4 (pulled in by sglang) whenever it's installed, and passes it the
# FA2-style causal window (-1, 0). Since FA4 4.0.0b30 that means an empty local window, so
# attention comes out wrong (NVIDIA/TransformerEngine#3528, fix pending in #3570). Skip FA4 on
# sm_12x so TE uses FA2, which is correct and fast there. Fails loudly if the TE source moves.
python - <<'EOF'
import sysconfig
# by path: importing TE before torch can't find libnccl
path = f"{sysconfig.get_paths()['purelib']}/transformer_engine/pytorch/attention/dot_product_attention/utils.py"
anchor = '            logger.debug("Disabling FlashAttention 4 for compute capability < sm80")\n        use_flash_attention_4 = False\n'
marker = "# miles-dgx-spark: skip FA4"
patch = f"    if device_compute_capability[0] == 12:  {marker}\n        use_flash_attention_4 = False\n"
src = open(path).read()
if marker not in src:
    assert src.count(anchor) == 1, f"FA4 patch anchor not found in {path}"
    open(path, "w").write(src.replace(anchor, anchor + patch))
print("TE: FA4 disabled on sm_12x")
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
PREFIX=$PREFIX "$HERE/verify.sh"

echo
echo "Done. Activate with: source $PREFIX/bin/activate"
echo "Then follow Quick Start in the README to train Qwen3-0.6B."
