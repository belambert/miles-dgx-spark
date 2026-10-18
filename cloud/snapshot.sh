#!/usr/bin/env bash
# Save the venv and sources to a persistent volume, and restore them after local disk is wiped
# (a RunPod stop does that; /workspace survives). Restoring takes minutes, against a full
# install.sh rebuild.
#
#   bash cloud/snapshot.sh save       archive PREFIX, SRC and SGLang's kernel cache
#   bash cloud/snapshot.sh restore    reinstall CUDA 13 if it's gone, unpack, run verify.sh
#
# The archive goes in $DATA_ROOT/snapshot, next to the models, datasets and runs that run.sh keeps
# there, so set DATA_ROOT to the volume. A tar archive keeps file permissions even on a volume that
# can't store them. The venv records absolute paths, so it's restored to the same place: clone the
# repo to the same path before restoring.
set -euo pipefail

HERE=$(dirname "$(dirname "$(readlink -f "$0")")")  # the repo root, one up from cloud/
PREFIX=${PREFIX:-$HERE/.venv}
SRC=${SRC:-$HERE/miles-src}
DATA_ROOT=${DATA_ROOT:-$HERE}
ARCHIVE=${ARCHIVE:-$DATA_ROOT/snapshot/miles-env.tar.zst}

case ${1:-} in
  save)
    [ -x "$PREFIX/bin/python" ] || { echo "no venv at PREFIX=$PREFIX" >&2; exit 1; }
    paths=("$PREFIX" "$SRC")
    [ -d "$HOME/.cache/sglang" ] && paths+=("$HOME/.cache/sglang")  # JIT kernels; saves compiling them again
    mkdir -p "$(dirname "$ARCHIVE")"
    # relative to / so restore puts everything back at the same absolute paths; written under a
    # temp name so an interrupted save doesn't replace a good archive
    tar -C / -cf - "${paths[@]#/}" | zstd -T0 -3 -q -o "$ARCHIVE.tmp"
    mv -f "$ARCHIVE.tmp" "$ARCHIVE"
    ls -lh "$ARCHIVE"
    ;;
  restore)
    [ -f "$ARCHIVE" ] || { echo "no snapshot at $ARCHIVE; run save first, or install.sh" >&2; exit 1; }
    /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -q "release 13\." || bash "$HERE/cloud/install-cuda.sh"
    zstd -d -T0 -q -c "$ARCHIVE" | tar -C / -xf -
    PREFIX=$PREFIX bash "$HERE/verify.sh"
    echo "Restored $PREFIX and $SRC"
    ;;
  *)
    sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 1 ;;
esac
