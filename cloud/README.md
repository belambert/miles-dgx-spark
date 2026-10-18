# Cloud Install

`install.sh` here installs the same pins as the DGX Spark installer on a cloud GPU machine, such as one H100, without Docker. It takes TransformerEngine, apex and flash-attn from the [miles-wheels](https://github.com/radixark/miles-wheels) release that the image installs, whose aarch64 builds are byte-identical to the image's. On Hopper it adds FlashAttention 3, which TransformerEngine prefers there, and it stops TransformerEngine from picking FlashAttention 4 on any GPU (see Attention in the [main README](../README.md)). Like the Spark installer, it clones the sources into `miles-src` and builds a venv in `.venv`, both at the repo root, where `run.sh` and `verify.sh` look for them.

The host needs an NVIDIA driver of 580 or later and a CUDA 13 toolkit in `/usr/local/cuda`, plus `git`, `curl` and a C++ compiler; the script installs `uv` and Rust if they're missing. From the repo root:

    bash cloud/install.sh 2>&1 | tee install.log

`run.sh` and `verify.sh` work unchanged, and `run.sh` still trains on one GPU. If `/workspace` is a network volume that local disk doesn't outlive, see below.

## CUDA 13 Toolkit

Many cloud images ship only CUDA 12. On Ubuntu (x86_64 or aarch64), as root, `install-cuda.sh` adds NVIDIA's apt repo and installs the toolkit; it goes alongside any CUDA 12 install and leaves the driver alone:

    bash cloud/install-cuda.sh

Some images already list the same repo without `signed-by`; with both entries, `apt-get update` fails with `Conflicting values set for option Signed-By`, so the script removes any list in `/etc/apt/sources.list.d/` that names `developer.download.nvidia.com` without `signed-by=`. The toolkit takes about 7 GB and points `/usr/local/cuda` at 13.0. The script does nothing if `/usr/local/cuda` already has CUDA 13.

## Network Volumes

Some cloud containers mount `/workspace` as a FUSE volume over object storage (`geesefs`, for one) that can't set file permissions or make hard links, and point `UV_CACHE_DIR` and `PIP_CACHE_DIR` at it. Files there can't be made executable, so a venv can't live there, and neither can uv's cache. Check with `mount | grep /workspace`.

The installer moves such caches to `~/.cache`, but it stops if `PREFIX` is on one of these volumes. Clone the repo onto local disk instead, so the venv and sources go in it as usual, and keep the data on the volume: `run.sh` puts `models/`, `datasets/` and `runs/` under `DATA_ROOT`, which defaults to the repo root.

    git clone https://github.com/belambert/miles-dgx-spark.git /root/miles-dgx-spark
    cd /root/miles-dgx-spark
    bash cloud/install.sh 2>&1 | tee install.log
    export DATA_ROOT=/workspace/miles-data
    ./run.sh prepare Qwen3.5-2B && ./run.sh qwen35-2b

Local disk in a container is often the root overlay, which may be small (30 GB on one A100 machine) and is lost when the container is recreated; on RunPod, stopping a pod wipes it too. Checkpoints write to the volume fine but slowly: about 140 MB/s on one RunPod volume, so 2.5 minutes for the 2B's 22 GB, against 25 s on local disk.

To skip the rebuild after a stop, archive the venv and sources onto the volume once the install finishes; it's about 8 GB, from 21 GB, and takes 3 minutes:

    DATA_ROOT=/workspace/miles-data bash cloud/snapshot.sh save

After a restart, clone the repo to the same path again; `restore` reinstalls CUDA 13 if it's gone, unpacks the venv and sources to the same paths and runs `verify.sh`:

    DATA_ROOT=/workspace/miles-data bash cloud/snapshot.sh restore

The archive is `$DATA_ROOT/snapshot/miles-env.tar.zst`. Save again after changing the venv.
