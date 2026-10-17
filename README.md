# miles-dgx-spark

Non-Docker install of [Miles](https://github.com/radixark/miles) for a DGX Spark (GB10, aarch64, CUDA 13).

## Usage

The installer clones the sources into `./miles-src` and builds a venv in `./.venv`. Miles comes from the `dgx-spark` branch of [our fork](https://github.com/belambert/miles/tree/dgx-spark), which adds a single-GPU Qwen3-0.6B recipe to `scripts/run_qwen3_dense.py`. The installer needs the pinned `radixark/miles` Docker image pulled locally, because a few compiled packages are copied out of it.

    ./install.sh 2>&1 | tee install.log

Check an existing install:

    ./verify.sh

Set `PREFIX` or `SRC` to use other locations. The pins at the top of `install.sh` say which image and commits they match.

## Quick Start

This is the upstream [quick start](https://github.com/radixark/miles/blob/main/docs/getting-started/quick-start.md) on one Spark, with Qwen3-0.6B instead of Qwen3-4B and everything kept under this repo instead of `/root`. Run the steps from the repo root.

Activate the venv and name the directories:

    source .venv/bin/activate
    HERE=$PWD MODELS=$PWD/models DATA=$PWD/datasets

Download the model, the training prompts and the eval benchmark:

    hf download Qwen/Qwen3-0.6B --local-dir $MODELS/Qwen3-0.6B
    hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir $DATA/dapo-math-17k
    hf download --repo-type dataset zhuzilin/aime-2024 --local-dir $DATA/aime-2024

Convert the weights to Megatron's `torch_dist` format. Keep the HuggingFace directory, because the rollout engine reads it:

    cd miles-src/miles
    MODEL_ARGS=$(python miles/utils/external_utils/model_args_utils.py qwen3-0.6B)
    PYTHONPATH=$HERE/miles-src/Megatron-LM python tools/convert_hf_to_torch_dist.py $MODEL_ARGS \
        --hf-checkpoint $MODELS/Qwen3-0.6B --save $MODELS/Qwen3-0.6B_torch_dist

Launch GRPO training. The `auto` attention backend lets Transformer Engine choose, since its flash backend isn't guaranteed on sm_121:

    python scripts/run_qwen3_dense.py --model-name Qwen3-0.6B --num-gpus-per-node 1 \
        --model-dir $MODELS --data-dir $DATA --output-dir $HERE/runs \
        --megatron-path $HERE/miles-src/Megatron-LM \
        --extra-args "--attention-backend auto"

Checkpoints land in `./runs/checkpoints` every 20 rollouts, and rerunning the launch command resumes from the last one. Add `--num-rollout 20 --no-enable-eval` for a short smoke test, and set `WANDB_API_KEY` to log to wandb. Watch `rollout/raw_reward` in the output climb.

The launcher starts a local Ray cluster. Before that it kills any running `sglang`, `ray` and `redis` processes, including ones serving from other envs.

The Ray dashboard listens only on the Spark's localhost, port 8265. To view it from another machine, open a tunnel there and browse to http://localhost:8265:

    ssh -N -L 8265:localhost:8265 bel@10.0.0.245
