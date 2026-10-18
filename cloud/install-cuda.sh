#!/usr/bin/env bash
# CUDA 13 toolkit for a cloud image that ships only CUDA 12 (Ubuntu, x86_64 or aarch64), which
# install.sh needs. Adds NVIDIA's apt repo and installs cuda-toolkit-13-0 alongside any CUDA 12;
# leaves the driver alone. Points /usr/local/cuda at 13.0. Run as root.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run as root" >&2; exit 1; }

if /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -q "release 13\."; then
  echo "CUDA 13 toolkit already in /usr/local/cuda"; exit 0
fi

# NVIDIA's repo names: ubuntu2404, and x86_64 or sbsa (aarch64 servers)
source /etc/os-release
DISTRO=ubuntu${VERSION_ID//./}
case $(uname -m) in
  x86_64) ARCH=x86_64 ;;
  aarch64) ARCH=sbsa ;;
  *) echo "unsupported arch $(uname -m)" >&2; exit 1 ;;
esac
REPO=https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH

# in a temp dir: the cwd may be a FUSE volume
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/cuda-keyring.deb" "$REPO/cuda-keyring_1.1-1_all.deb"
dpkg -i "$tmp/cuda-keyring.deb"

# Some images already list the same repo without signed-by; with both entries, apt-get update
# fails with "Conflicting values set for option Signed-By". Drop those lists.
for f in $(grep -l developer.download.nvidia.com /etc/apt/sources.list.d/*.list 2>/dev/null || true); do
  grep -q signed-by= "$f" || { echo "removing $f (no signed-by)"; rm -f "$f"; }
done

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-13-0

/usr/local/cuda/bin/nvcc --version | tail -2
