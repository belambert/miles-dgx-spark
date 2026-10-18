# Cloud Install

`install.sh` here installs the same pins as the DGX Spark installer on a cloud GPU machine, such as one H100, without Docker. It takes TransformerEngine, apex and flash-attn from the [miles-wheels](https://github.com/radixark/miles-wheels) release that the image installs, whose aarch64 builds are byte-identical to the image's. On Hopper it adds FlashAttention 3, which TransformerEngine prefers there, and it stops TransformerEngine from picking FlashAttention 4 on any GPU (see Attention in the [main README](../README.md)). Like the Spark installer, it clones the sources into `miles-src` and builds a venv in `.venv`, both at the repo root, where `run.sh` and `verify.sh` look for them.

The host needs an NVIDIA driver of 580 or later and a CUDA 13 toolkit in `/usr/local/cuda`, plus `git`, `curl` and a C++ compiler; the script installs `uv` and Rust if they're missing. From the repo root:

    bash cloud/install.sh 2>&1 | tee install.log

`run.sh` and `verify.sh` work unchanged, and `run.sh` still trains on one GPU. Run them with `bash` too if the checkout has lost its exec bits (see below).

## CUDA 13 Toolkit

Many cloud images ship only CUDA 12. On Ubuntu 24.04 (x86_64), as root, add NVIDIA's apt repo and install the toolkit; it goes alongside any CUDA 12 install and leaves the driver alone:

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
    rm -f /etc/apt/sources.list.d/cuda.list
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-13-0

The `rm` of `cuda.list` is for images that already list the same repo without `signed-by`; with both entries, `apt-get update` fails with `Conflicting values set for option Signed-By`. If the image uses another file name, find it with `grep -rn developer.download.nvidia.com /etc/apt/sources.list.d/` and remove the entry without `signed-by=`. The toolkit takes about 7 GB and points `/usr/local/cuda` at 13.0. For another Ubuntu release or aarch64, change `ubuntu2404/x86_64` in the URL.

## Network Volumes

Some cloud containers mount `/workspace` as a FUSE volume over object storage (`geesefs`, for one) that can't set file permissions or make hard links, and point `UV_CACHE_DIR` and `PIP_CACHE_DIR` at it. Files there can't be made executable, so a venv can't live there, and neither can uv's cache. Check with `mount | grep /workspace`.

The installer moves such caches to `~/.cache`, but it stops if `PREFIX` is on one of these volumes, because the venv would default to one inside a checkout there. Keep the repo on the volume if you like, and put the venv and sources on local disk:

    PREFIX=/root/miles/.venv SRC=/root/miles/miles-src bash cloud/install.sh 2>&1 | tee install.log

Pass the same `PREFIX` and `SRC` to `run.sh` and `verify.sh`. Local disk in a container is often the root overlay, which may be small (30 GB on one A100 machine) and is lost when the container is recreated.
