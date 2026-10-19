# Pins shared by install.sh (DGX Spark) and cloud/install.sh, sourced by both.
# They match radixark/miles:latest as run on this Spark, 2026-10-17. Before changing them, compare
# with `git show <tag>:docker/Dockerfile` and the release notes for the tag you target.

# Miles has no release matching the image: /root/miles there is e1a791d3e3, an untagged
# origin/main commit (2026-10-16) whose package reports 0.1.2. The v0.1.2 tag (a22fd82419)
# is not its ancestor, so don't swap in a tag. Unlike older refs, this commit has no
# docker/patch/cu13/; its TE patches only exist baked into the image's build (and the
# radixark/miles-wheels release it installs).
# We use the dgx-spark branch of our fork: that commit plus single-GPU Qwen3-0.6B/1.7B,
# Qwen3.5-0.8B and Qwen3.5-2B recipes (and a Qwen3.5-2B model config), --attention-backend and
# --dump-details flags, --save-post-update-log-probs, and the dashboard's ref_drift and step_diff.
MILES_REPO=${MILES_REPO:-https://github.com/belambert/miles.git}
MILES_REF=${MILES_REF:-a86239f3a5}            # dgx-spark, on top of e1a791d3e3 (/root/miles in the image)
SGLANG_COMMIT=${SGLANG_COMMIT:-74bd569} # sglang-miles branch on sgl-project/sglang
MEGATRON_COMMIT=${MEGATRON_COMMIT:-9df207fd1} # radixark/Megatron-LM, miles-main
TMS_COMMIT=b5588e83de86412a48689a6583a4b567e75f7acc      # torch_memory_saver
MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
BRIDGE_COMMIT=7f0fb3456f8ffe47599b5fd167b454605d85f932   # radixark/Megatron-Bridge
CUDNN_VERSION=9.24.0.43
TVM_FFI_VERSION=0.1.11   # must match sglang's own pin; check python/pyproject.toml
# pinned by digest: this is the :latest that every pin in this file was read from
MILES_IMAGE=${MILES_IMAGE:-radixark/miles@sha256:ad52af8b83076331d43e65a7a71ecc1daf128512b82829701e098540fcd80478}
# torch must match what the prebuilt wheels (TE, flash-attn, apex) were built against,
# i.e. the torch in the docker image -- NOT sglang-miles' pyproject pin. Get them with:
#   docker run --rm radixark/miles:<tag> pip freeze | grep -E '^torch(vision|audio)?=='
TORCH_VERSION=${TORCH_VERSION:-2.14.1+cu130}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.29.1+cu130}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.11.0+cu130}
