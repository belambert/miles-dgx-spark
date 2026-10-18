# miles-dgx-spark

Non-Docker install of [Miles](https://github.com/radixark/miles) for a DGX Spark (GB10, aarch64, CUDA 13).

## Compromises

This runs the upstream [quick start](https://github.com/radixark/miles/blob/main/docs/getting-started/quick-start.md) on one Spark, which takes some changes. "Where" says which launch carries each change: the Quick Start command below (all), `smoke.sh` (smoke), `smoke-thinking.sh` (thinking), or `smoke-small.sh` (small); the other smoke scripts build on `smoke.sh`.

| Setting             | Upstream quick start               | Here                                                  | Where           | Why                                                 |
|---------------------|------------------------------------|-------------------------------------------------------|-----------------|-----------------------------------------------------|
| Install             | Docker image                       | host venv; TE, apex, flash-attn copied from the image | all             | Non-Docker install                                  |
| Model               | Qwen3-4B                           | Qwen3-0.6B                                            | all             | Fits one GPU with TP 1                              |
| GPUs                | 8 (TP 2, 4 SGLang engines)         | 1 (TP 1, 1 engine)                                    | all             | A Spark has one GB10                                |
| Attention           | FlashAttention via TE              | TE patched to use FA2, not FA4                        | all             | TE + FA4 gives wrong attention (TE#3528)            |
| Rollouts            | 3000                               | 20                                                    | smoke           | A quick end-to-end check                            |
| AIME eval           | before training, every 20 rollouts | off                                                   | smoke           | The first eval alone takes hours                    |
| Prompts per rollout | 32 (256 samples)                   | 8 (64 samples)                                        | small, thinking | 256 long samples overflow the KV cache              |
| Global batch        | 256                                | 64                                                    | small, thinking | Matches the smaller rollout                         |
| Max response        | 8192 tokens                        | 2048 tokens                                           | small           | Keeps rollouts under a minute                       |
| Qwen3 thinking      | on                                 | off                                                   | small           | Thinking runs past 2k tokens, so all rewards were 0 |
| Reward              | `deepscaler`                       | `math`                                                | small           | `deepscaler` needs `</think>` in the response       |

So the full Quick Start command differs from upstream only in model size, GPU count and the attention fix, but on one Spark its rollouts take 40+ minutes and its first eval several hours. Only `smoke-small.sh` runs in minutes, and it trains a different setup: short, non-thinking answers graded by a different reward.

## Usage

The installer clones the sources into `./miles-src` and builds a venv in `./.venv`. Miles comes from the `dgx-spark` branch of [our fork](https://github.com/belambert/miles/tree/dgx-spark), which adds single-GPU Qwen3-0.6B and Qwen3.5-0.8B recipes and an `--attention-backend` flag to `scripts/run_qwen3_dense.py`. The installer needs the pinned `radixark/miles` Docker image pulled locally, because a few compiled packages are copied out of it.

    ./install.sh 2>&1 | tee install.log

Check an existing install:

    ./verify.sh

Set `PREFIX` or `SRC` to use other locations. The pins at the top of `install.sh` say which image and commits they match.

## Quick Start

This is the upstream [quick start](https://github.com/radixark/miles/blob/main/docs/getting-started/quick-start.md) on one Spark, with Qwen3-0.6B instead of Qwen3-4B and everything kept under this repo instead of `/root`. Run the steps from the repo root.

| Role     | Hugging Face repo                                                                      | Contents                                         |
|----------|----------------------------------------------------------------------------------------|--------------------------------------------------|
| Model    | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)                              | The policy being trained                         |
| Training | [zhuzilin/dapo-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)       | 17k math problems with checkable answers         |
| Eval     | [zhuzilin/aime-2024](https://huggingface.co/datasets/zhuzilin/aime-2024)               | AIME 2024 problems, evaluated on but not trained |

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

Launch GRPO training. Keep the recipe's `flash` attention backend. Transformer Engine would pick FlashAttention 4 (an SGLang dependency) and pass it a causal window that FA4 4.0.0b30+ reads as empty, giving wrong attention ([TransformerEngine#3528](https://github.com/NVIDIA/TransformerEngine/issues/3528), fix pending in [#3570](https://github.com/NVIDIA/TransformerEngine/pull/3570)). So `install.sh` patches TE to use FlashAttention 2 on the GB10 instead. Don't use `auto`: it picks cuDNN fused attention, which can't run the backward pass on packed sequences:

    python scripts/run_qwen3_dense.py --model-name Qwen3-0.6B --num-gpus-per-node 1 \
        --model-dir $MODELS --data-dir $DATA --output-dir $HERE/runs/quickstart \
        --megatron-path $HERE/miles-src/Megatron-LM

Checkpoints land in `./runs/quickstart/checkpoints` every 20 rollouts, and rerunning the launch command resumes from the last one. Set `WANDB_API_KEY` to log to wandb. Watch `rollout/raw_reward` in the output climb.

The full recipe runs 3000 rollouts and starts with an AIME eval that takes hours on one Spark. For a quick end-to-end check, `smoke.sh` runs the same launch with 20 rollouts and no eval:

    ./smoke.sh

`smoke.sh` keeps the recipe's batch: 256 samples of up to 8k tokens, which overflow SGLang's KV cache and take 40+ minutes per rollout. `smoke-small.sh` runs 64 samples of up to 2k tokens with more memory for the cache and Qwen3's thinking turned off, for about 40 seconds per iteration:

    ./smoke-small.sh

Each launch keeps its checkpoints in its own directory (`runs/quickstart`, `runs/smoke`, `runs/smoke-small`, `runs/smoke-thinking`, `runs/smoke-thinking-qwen35`), because a run resumes from whatever checkpoint its directory holds. A finished smoke run therefore won't train again; delete its directory to start over, for example `rm -rf runs/smoke-small`. Each checkpoint of the 0.6B model is about 8 GB.

`smoke-thinking.sh` sits in between: `smoke-small.sh`'s batch and cache size, but thinking on, 8k-token responses and the recipe's `deepscaler` reward. It checks the thinking path, including training on long sequences, at an estimated 5–10 minutes per rollout:

    ./smoke-thinking.sh

`smoke-thinking-qwen35.sh` runs the same test on [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B). 18 of its 24 layers use linear attention, which keeps a fixed-size state instead of a growing KV cache, so it stores about 12 KB per token against Qwen3-0.6B's 112 KB. Long thinking rollouts are limited by reading that cache, so they should run faster. Download and convert it first; the conversion needs the GPU free, so stop any running job:

    hf download Qwen/Qwen3.5-0.8B --local-dir $MODELS/Qwen3.5-0.8B
    cd miles-src/miles
    MODEL_ARGS=$(python miles/utils/external_utils/model_args_utils.py qwen3.5-0.8B)
    PYTHONPATH=$HERE/miles-src/Megatron-LM python tools/convert_hf_to_torch_dist.py $MODEL_ARGS \
        --hf-checkpoint $MODELS/Qwen3.5-0.8B --save $MODELS/Qwen3.5-0.8B_torch_dist
    cd $HERE
    ./smoke-thinking-qwen35.sh

Thinking is off because Qwen3's thinking runs past 2k tokens: with it on, nearly every response was truncated before reaching an answer, every reward was 0, and GRPO had nothing to learn from. Without thinking, responses average 600–800 tokens and none are truncated.

Turning thinking off also means switching the reward from the recipe's `deepscaler` to `math`. `deepscaler` only grades the text after `</think>` and scores 0 when there isn't one. With thinking off, Qwen3's chat template puts an empty `<think></think>` block in the prompt, so responses never contain `</think>` and every answer scored 0, right or wrong. `math` grades the last `\boxed{}` answer anywhere in the response, which is the format the dataset's prompts ask for.

The launcher starts a local Ray cluster. Before that it kills any running `sglang`, `ray` and `redis` processes, including ones serving from other envs.

The Ray dashboard listens only on the Spark's localhost, port 8265. To view it from another machine, open a tunnel there and browse to http://localhost:8265:

    ssh -N -L 8265:localhost:8265 bel@10.0.0.245

## Managing Jobs

The launcher submits training as a Ray job. With the venv active, list running jobs (`ray job list` prints raw objects, including each job's full `train.py` command line):

    ray list jobs --filter "status=RUNNING" --format json | jq -r '.[] | "\(.submission_id)  \(.status)"'

Inspect, follow or stop a job by its submission ID:

    ray job list                     # every job, including finished and stopped ones
    ray job status <submission_id>   # one job's status
    ray job logs -f <submission_id>  # follow a job's output
    ray job stop <submission_id>
