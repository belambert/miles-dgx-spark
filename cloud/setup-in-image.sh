#!/usr/bin/env bash
# Set up this repo inside the radixark/miles Docker image itself, e.g. a cloud pod started from it.
# The image already has every package install.sh would build, in the venv at /opt/sglang, so this
# only points the repo's .venv and miles-src at what's there, swaps in our Miles fork and applies
# the TransformerEngine FA4 fix. Takes a minute, against ~15 for cloud/install.sh.
#
#   bash cloud/setup-in-image.sh
#
# .venv becomes a symlink to the image's venv, and miles-src/{sglang,Megatron-LM} symlinks to the
# image's checkouts; miles-src/miles is a clone of our fork, installed editable in place of the
# image's /root/miles (which is left as it is). The image is the snapshot, so cloud/snapshot.sh
# isn't needed.
set -euo pipefail

HERE=$(dirname "$(dirname "$(readlink -f "$0")")")  # the repo root, one up from cloud/
source "$HERE/pins.sh"

# where the image keeps things (see its docker/Dockerfile)
IMAGE_VENV=${IMAGE_VENV:-/opt/sglang}
IMAGE_SGLANG=${IMAGE_SGLANG:-/sgl-workspace/sglang}
IMAGE_MEGATRON=${IMAGE_MEGATRON:-/root/Megatron-LM}

PREFIX=${PREFIX:-$HERE/.venv}
SRC=${SRC:-$HERE/miles-src}

for d in "$IMAGE_VENV/bin/python" "$IMAGE_SGLANG" "$IMAGE_MEGATRON"; do
  [ -e "$d" ] || { echo "$d not found: not the radixark/miles image? Use cloud/install.sh instead" >&2; exit 1; }
done

# ---------------------------------------------------------------- 1. check the image against pins.sh
# a different image tag usually still works, but it isn't what the pins were tested with
mismatch=0
check() {  # name, want, have
  [ "$2" = "$3" ] && return
  echo "warning: image has $1 $3, pins.sh says $2" >&2
  mismatch=1
}
check torch "$TORCH_VERSION" "$("$IMAGE_VENV/bin/python" -c 'import torch; print(torch.__version__)')"
check sglang "$SGLANG_COMMIT" "$(git -C "$IMAGE_SGLANG" rev-parse --short=7 HEAD)"
check Megatron-LM "$MEGATRON_COMMIT" "$(git -C "$IMAGE_MEGATRON" rev-parse --short=9 HEAD)"
check apache-tvm-ffi "$TVM_FFI_VERSION" "$("$IMAGE_VENV/bin/python" -c 'from importlib.metadata import version; print(version("apache-tvm-ffi"))')"
[ $mismatch = 0 ] && echo "image matches pins.sh"

# ---------------------------------------------------------------- 2. venv and sources: link to the image's
link() {  # target, link name; refuses to replace a real directory, such as one install.sh built
  if [ -e "$2" ] && [ ! -L "$2" ]; then
    echo "$2 already exists and isn't a symlink (from install.sh?); move it aside first" >&2
    exit 1
  fi
  ln -sfn "$1" "$2"
}
link "$IMAGE_VENV" "$PREFIX"
mkdir -p "$SRC"
link "$IMAGE_SGLANG" "$SRC/sglang"
link "$IMAGE_MEGATRON" "$SRC/Megatron-LM"

# ---------------------------------------------------------------- 3. miles: our fork instead of the image's
# pinned commits mean detached HEADs; skip git's multi-line advice about it
[ -d "$SRC/miles" ] || git clone "$MILES_REPO" "$SRC/miles"
git -C "$SRC/miles" fetch --tags origin && git -C "$SRC/miles" -c advice.detachedHead=false checkout "$MILES_REF"
"$PREFIX/bin/python" -m pip install -e "$SRC/miles" --no-deps

# ---------------------------------------------------------------- 4. TE: skip FlashAttention 4
# The image has FA4 installed, and TE passes it the FA2-style causal window (-1, 0), which since
# FA4 4.0.0b30 means an empty local window, so attention comes out wrong
# (NVIDIA/TransformerEngine#3528, fix pending in #3570). Skip FA4 on every GPU, as cloud/install.sh
# does, so TE uses FA3 on Hopper and FA2 elsewhere. Fails loudly if the TE source moves.
"$PREFIX/bin/python" - <<'EOF'
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

# ---------------------------------------------------------------- 5. verify
PREFIX=$PREFIX bash "$HERE/verify.sh"  # bash: the checkout may have lost its exec bits
"$PREFIX/bin/python" -c "import miles, os; print('miles from', os.path.dirname(miles.__file__))"

echo
echo "Done. run.sh and verify.sh use the image's venv through $PREFIX."
echo "Then follow Quick Start in the README; set DATA_ROOT to keep data on a volume."
