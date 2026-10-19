#!/usr/bin/env bash
# Download a model and the datasets, and convert the model to Megatron's torch_dist format.
#
#   ./prepare.sh <model>       e.g. Qwen3.5-2B or Qwen3-0.6B; needs the GPU free, so stop any running job
#
# The rollout engine reads the Hugging Face copy and the trainer the torch_dist one, so both are kept.
# MATH, which isn't on the zhuzilin account, has its own script: prepare-math.py.
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")
PREFIX=${PREFIX:-$HERE/.venv}
SRC=${SRC:-$HERE/miles-src}
DATA_ROOT=${DATA_ROOT:-$HERE}  # holds models/, datasets/ and runs/, as in run.sh

model=${1:?usage: ./prepare.sh <model>}
source "$PREFIX/bin/activate"

hf download "Qwen/$model" --local-dir "$DATA_ROOT/models/$model"
for ds in dapo-math-17k aime-2024 gsm8k; do
  hf download --repo-type dataset "zhuzilin/$ds" --local-dir "$DATA_ROOT/datasets/$ds"
done

cd "$SRC/miles"
PYTHONPATH="$SRC/Megatron-LM" python tools/convert_hf_to_torch_dist.py \
  $(python miles/utils/external_utils/model_args_utils.py "${model/Qwen/qwen}") \
  --hf-checkpoint "$DATA_ROOT/models/$model" --save "$DATA_ROOT/models/${model}_torch_dist"
