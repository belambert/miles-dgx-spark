# Cloud Install

`install.sh` here installs the same pins as the DGX Spark installer on a cloud GPU machine, such as one H100, without Docker. It takes TransformerEngine, apex and flash-attn from the [miles-wheels](https://github.com/radixark/miles-wheels) release that the image installs, whose aarch64 builds are byte-identical to the image's. On Hopper it adds FlashAttention 3, which TransformerEngine prefers there, and it stops TransformerEngine from picking FlashAttention 4 on any GPU (see Attention in the [main README](../README.md)). Like the Spark installer, it clones the sources into `miles-src` and builds a venv in `.venv`, both at the repo root, where `run.sh` and `verify.sh` look for them.

The host needs an NVIDIA driver of 580 or later and a CUDA 13 toolkit in `/usr/local/cuda`, plus `git`, `curl` and a C++ compiler; the script installs `uv` and Rust if they're missing. From the repo root:

    bash cloud/install.sh 2>&1 | tee install.log

`run.sh` and `verify.sh` work unchanged, and `run.sh` still trains on one GPU. If `/workspace` is a network volume that local disk doesn't outlive, see below.

## Inside the Miles Image

On a machine started from the `radixark/miles` image itself, such as a pod using it as its container image, nothing needs building: the image's venv in `/opt/sglang` already has every package. `setup-in-image.sh` instead links `.venv` to that venv and `miles-src/sglang` and `miles-src/Megatron-LM` to the image's checkouts, clones our Miles fork into `miles-src/miles` and installs it in place of the image's `/root/miles`, applies the FlashAttention 4 fix, and runs `verify.sh`. It takes about a minute, and warns if the image's torch, SGLang, Megatron-LM or tvm-ffi differ from `pins.sh`:

    bash cloud/setup-in-image.sh

It won't replace a `.venv` or source directory that `install.sh` built. The image itself plays the part of a snapshot, so `snapshot.sh` isn't needed.

## CUDA 13 Toolkit

Many cloud images ship only CUDA 12. On Ubuntu (x86_64 or aarch64), as root, `install-cuda.sh` adds NVIDIA's apt repo and installs the toolkit; it goes alongside any CUDA 12 install and leaves the driver alone:

    bash cloud/install-cuda.sh

Some images already list the same repo without `signed-by`; with both entries, `apt-get update` fails with `Conflicting values set for option Signed-By`, so the script removes any list in `/etc/apt/sources.list.d/` that names `developer.download.nvidia.com` without `signed-by=`. The toolkit takes about 7 GB and points `/usr/local/cuda` at 13.0. The script does nothing if `/usr/local/cuda` already has CUDA 13.

## Network Volumes

Some cloud containers mount `/workspace` as a FUSE volume over object storage (`geesefs`, for one) that can't set file permissions or make hard links, and point `UV_CACHE_DIR` and `PIP_CACHE_DIR` at it. Files there can't be made executable, so a venv can't live there, and neither can uv's cache. Check with `mount | grep /workspace`.

The installer moves such caches to `~/.cache`, but it stops if `PREFIX` is on one of these volumes. Clone the repo onto local disk instead, so the venv and sources go in it as usual, and keep the data on the volume: `run.sh` puts `models/`, `datasets/` and `runs/` under `DATA_ROOT`, which defaults to the repo root.

Local disk in a container is often the root overlay, which may be small (30 GB on one A100 machine) and is lost when the container is recreated; on RunPod, stopping a pod wipes it too. Checkpoints write to the volume fine but slowly: about 140 MB/s on one RunPod volume, so 2.5 minutes for the 2B's 22 GB, against 25 s on local disk.

## New RunPod Pod

For a pod whose `/workspace` volume is empty. These steps come from one H100 80GB pod with driver 580 and Ubuntu 24.04, on an image that ships CUDA 12.8. Give the pod a container disk of at least 50 GB, for CUDA 13 (5 GB), the venv and sources (21 GB) and caches; and a volume big enough for the data: about 8 GB for the 2B model and datasets, 22 GB per 2B checkpoint, and 8 GB for a snapshot.

Clone the repo onto local disk, not `/workspace`:

    git clone https://github.com/belambert/miles-dgx-spark.git /root/miles-dgx-spark
    cd /root/miles-dgx-spark

Install CUDA 13 (skipped if the image already has it) and build the venv and sources, which takes about 15 minutes:

    bash cloud/install-cuda.sh
    bash cloud/install.sh 2>&1 | tee install.log

Point `DATA_ROOT` at the volume, then download and convert a model and train:

    export DATA_ROOT=/workspace/miles-data
    ./prepare.sh Qwen3.5-2B
    .venv/bin/python prepare-math.py
    ./run.sh qwen35-2b

Runs go to `$DATA_ROOT/runs/<preset>`, and resume from there. To skip the install the next time the pod starts, take a snapshot (below).

## Snapshot and Restore

`snapshot.sh` archives the venv, the sources and SGLang's kernel cache into `$DATA_ROOT/snapshot/miles-env.tar.zst` on the volume. A restarted pod, or a new pod with the same volume attached, then gets them back in a few minutes instead of a rebuild. Models, datasets and runs are already on the volume, so they aren't in it.

Save after the install finishes, and again after anything changes the venv, such as a newer pin. The archive is about 8 GB, from 21 GB, and takes about 3 minutes:

    cd /root/miles-dgx-spark
    DATA_ROOT=/workspace/miles-data bash cloud/snapshot.sh save

To restore on a restarted or new pod, clone the repo to the same path again (the venv records absolute paths), then restore. `restore` reinstalls CUDA 13 if it's gone, unpacks the venv and sources into the checkout and runs `verify.sh`:

    git clone https://github.com/belambert/miles-dgx-spark.git /root/miles-dgx-spark
    cd /root/miles-dgx-spark
    DATA_ROOT=/workspace/miles-data bash cloud/snapshot.sh restore

Then train as before:

    export DATA_ROOT=/workspace/miles-data
    ./run.sh qwen35-2b

Restore only on the same kind of GPU. The install adds FlashAttention 3 only on Hopper and builds some packages for the GPU it ran on, so on another kind of GPU, run `install.sh` again and take a new snapshot.
