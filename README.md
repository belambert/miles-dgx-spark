# miles-dgx-spark

Non-Docker install of [Miles](https://github.com/radixark/miles) for a DGX Spark (GB10, aarch64, CUDA 13).

## Compromises

This runs the upstream [quick start](https://github.com/radixark/miles/blob/main/docs/getting-started/quick-start.md) on one Spark, which takes some changes. "Where" says which `run.sh` presets carry each change (see Quick Start).

| Setting             | Upstream quick start               | Here                                                  | Where                        | Why                                                 |
|---------------------|------------------------------------|-------------------------------------------------------|------------------------------|-----------------------------------------------------|
| Install             | Docker image                       | host venv; TE, apex, flash-attn copied from the image | all                          | Non-Docker install                                  |
| Model               | Qwen3-4B                           | Qwen3-0.6B, Qwen3.5-0.8B/2B/4B                        | all                          | Fits one GPU with TP 1                              |
| GPUs                | 8 (TP 2, 4 SGLang engines)         | 1 (TP 1, 1 engine)                                    | all                          | A Spark has one GB10                                |
| Attention           | FlashAttention via TE              | TE patched to use FA2, not FA4                        | all                          | TE + FA4 gives wrong attention (TE#3528)            |
| Rollouts            | 3000                               | 20                                                    | all but `quickstart`         | A quick end-to-end check                            |
| AIME eval           | before training, every 20 rollouts | off                                                   | all but `quickstart`         | The first eval alone takes hours                    |
| Prompts per rollout | 32 (256 samples)                   | 8 (64 samples)                                        | all but `quickstart`, `full` | 256 long samples overflow the KV cache              |
| Global batch        | 256                                | 64                                                    | all but `quickstart`, `full` | Matches the smaller rollout                         |
| Max response        | 8192 tokens                        | 2048–4096 tokens                                      | no-thinking presets          | Keeps rollouts under a minute                       |
| Qwen3 thinking      | on                                 | off                                                   | no-thinking presets          | Thinking runs past 2k tokens, so all rewards were 0 |
| Reward              | `deepscaler`                       | `math`                                                | no-thinking presets          | `deepscaler` needs `</think>` in the response       |

So the `quickstart` preset differs from upstream only in model size, GPU count and the attention fix, but on one Spark its rollouts take 40+ minutes and its first eval several hours. The presets that run in minutes train a different setup: short, non-thinking answers graded by a different reward.

## Usage

The installer clones the sources into `./miles-src` and builds a venv in `./.venv`. Miles comes from the `dgx-spark` branch of [our fork](https://github.com/belambert/miles/tree/dgx-spark), which adds single-GPU Qwen3-0.6B, Qwen3.5-0.8B and Qwen3.5-2B recipes, a Qwen3.5-2B model config, and `--attention-backend` and `--dump-details` flags to `scripts/run_qwen3_dense.py`. The installer needs the pinned `radixark/miles` Docker image pulled locally, because a few compiled packages are copied out of it.

    ./install.sh 2>&1 | tee install.log

Check an existing install:

    ./verify.sh

Set `PREFIX` or `SRC` to use other locations. The pins at the top of `install.sh` say which image and commits they match.

## Quick Start

This is the upstream [quick start](https://github.com/radixark/miles/blob/main/docs/getting-started/quick-start.md) on one Spark, with Qwen3-0.6B instead of Qwen3-4B and everything kept under this repo instead of `/root`. Run the steps from the repo root.

| Role            | Hugging Face repo                                                                | Contents                                          |
|-----------------|----------------------------------------------------------------------------------|---------------------------------------------------|
| Model           | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)                        | The policy being trained                          |
| Training        | [zhuzilin/dapo-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k) | 17k math problems with checkable answers          |
| Eval            | [zhuzilin/aime-2024](https://huggingface.co/datasets/zhuzilin/aime-2024)         | AIME 2024 problems, evaluated on but not trained  |
| Training (easy) | [zhuzilin/gsm8k](https://huggingface.co/datasets/zhuzilin/gsm8k)                 | 7.5k grade-school word problems, for small models |

Download a model and the datasets, and convert the model to Megatron's `torch_dist` format. The rollout engine reads the Hugging Face copy, so both are kept. The conversion needs the GPU, so stop any running job first:

    ./run.sh prepare Qwen3-0.6B

Then train with a preset:

    ./run.sh qwen3-small

`run.sh` launches Miles' `scripts/run_qwen3_dense.py` with the preset's settings. Each preset is one line near the top of the script, and any setting (`MODEL`, `DATASET`, `THINKING`, `MAX_LEN`, `PROMPTS`, `ROLLOUTS`, `EVAL`, `SGLANG_MEM`, `REWARD`, ...) can be overridden from the environment, for example `MAX_LEN=16384 ./run.sh qwen35-thinking`. Arguments after the preset go to the launcher. The first line it prints sums up the run. Every rollout has 8 samples per prompt.

| Preset            | Model        | Data  | Thinking | Max response | Prompts | Notes                                                             |
|-------------------|--------------|-------|----------|--------------|---------|-------------------------------------------------------------------|
| `quickstart`      | Qwen3-0.6B   | dapo  | on       | 8192         | 32      | Upstream recipe: 3000 rollouts, AIME eval; hours per rollout here |
| `full`            | Qwen3-0.6B   | dapo  | on       | 8192         | 32      | 20 rollouts, no eval; overflows the KV cache, 40+ min per rollout |
| `qwen3-small`     | Qwen3-0.6B   | dapo  | off      | 2048         | 8       | The fast end-to-end check: ~40 s per iteration, reward ~0.07      |
| `qwen3-thinking`  | Qwen3-0.6B   | dapo  | on       | 8192         | 8       | 17 min rollouts, reward ~0.09                                     |
| `qwen35-thinking` | Qwen3.5-0.8B | dapo  | on       | 8192         | 8       | 5 min rollouts, but 94% cut off mid-thought, reward 0             |
| `qwen35-dapo`     | Qwen3.5-0.8B | dapo  | off      | 4096         | 8       | Never finishes a dapo problem, reward 0                           |
| `qwen35-gsm8k`    | Qwen3.5-0.8B | gsm8k | off      | 2048         | 8       | ~1 min per iteration, reward 0.36–0.63                            |
| `qwen35-2b`       | Qwen3.5-2B   | dapo  | off      | 4096         | 8       | Untested                                                          |
| `qwen35-4b`       | Qwen3.5-4B   | dapo  | off      | 4096         | 8       | Reward 0.41, but out of memory in its first optimizer step        |

Each preset writes to `runs/<preset>` and resumes from the checkpoint there, so a finished run won't train again; delete its directory to start over, for example `rm -rf runs/qwen3-small`. A checkpoint with optimizer state is about 8 GB for the 0.6B and 50–60 GB for the 4B. Each run also saves every rollout's prompts, responses and rewards in `runs/<preset>/dump_details/trajectory/<rollout>.jsonl`, so you can read what the model wrote. Don't send requests to the running SGLang engine to check outputs instead: it releases its GPU memory during each training step, and a request arriving then crashes the run. Set `WANDB_API_KEY` to log to wandb.

### Why the Presets Look Like This

**Attention.** Keep the recipe's `flash` attention backend. Transformer Engine would pick FlashAttention 4 (an SGLang dependency) and pass it a causal window that FA4 4.0.0b30+ reads as empty, giving wrong attention ([TransformerEngine#3528](https://github.com/NVIDIA/TransformerEngine/issues/3528), fix pending in [#3570](https://github.com/NVIDIA/TransformerEngine/pull/3570)). So `install.sh` patches TE to use FlashAttention 2 on the GB10 instead. Don't use `auto`: it picks cuDNN fused attention, which can't run the backward pass on packed sequences.

**Thinking.** Qwen3's thinking runs past 2k tokens: with it on, nearly every response was truncated before reaching an answer, every reward was 0, and GRPO had nothing to learn from. Without thinking, Qwen3-0.6B's responses average 600–800 tokens and none are truncated.

**Reward.** Turning thinking off means switching the reward from the recipe's `deepscaler` to `math`, which `run.sh` does automatically. `deepscaler` only grades the text after `</think>` and scores 0 when there isn't one. With thinking off, Qwen's chat template puts an empty `<think></think>` block in the prompt, so responses never contain `</think>` and every answer scored 0, right or wrong. `math` grades the last `\boxed{}` answer anywhere in the response, which is the format the prompts ask for.

**Qwen3.5.** 18 of Qwen3.5-0.8B's 24 layers use linear attention, which keeps a fixed-size state instead of a growing KV cache, so it stores about 12 KB per token against Qwen3-0.6B's 112 KB, and long rollouts run about 3× faster. But Qwen3.5 writes much longer answers. With thinking on, 73% were still cut off at 16k. Even without thinking it reasons step by step in the visible answer: offline samples on dapo ran to 4k every time without converging, so the 0.8B gets GSM8K instead. `run.sh` turns off SGLang's prefix cache for Qwen3.5, which kept about 3 linear-attention state slots per request and capped concurrency (19 of 64 for the 4B); prefix hits are only about 4% anyway. Log-probs are computed in chunks, because unchunked, one 16k sequence over Qwen3.5's 248k-token vocabulary needs 15 GiB of FP32 logits.

**Memory.** On the GB10, CPU and GPU share one memory pool, so the trainer's state "offloaded" between steps still takes memory, and its weights, gradients and Adam state (about 16 bytes per parameter) must fit alongside SGLang. `LOW_MEM=1` stores Adam's moments in BF16. The 4B preset also lowers SGLang to 0.3 and halves the training micro-batch, yet still ran out of memory in its first optimizer step, with about 71 GB held outside the trainer.

The launcher starts a local Ray cluster. Before that it kills any running `sglang`, `ray` and `redis` processes, including ones serving from other envs.

The Ray dashboard listens only on the Spark's localhost, port 8265. To view it from another machine, open a tunnel there and browse to http://localhost:8265:

    ssh -N -L 8265:localhost:8265 bel@10.0.0.245

## Measurements

Runs on one Spark (2026-10-17), with the TE patch unless noted. "KL" is `train/train_rollout_kl`, the gap between SGLang's and Megatron's log-probs for the same tokens; near zero means training sees the same model SGLang sampled from. Reward is the fraction of answers graded correct per rollout.

| Preset            | Model        | Samples | Max response | Thinking | Rollout                      | Train step | Mean response | Truncated | Reward         | KL     |
|-------------------|--------------|---------|--------------|----------|------------------------------|------------|---------------|-----------|----------------|--------|
| `full`            | Qwen3-0.6B   | 256     | 8192         | on       | not finished, est. 40–50 min | –          | –             | –         | –              | –      |
| `qwen3-thinking`  | Qwen3-0.6B   | 64      | 8192         | on       | 17 min                       | 99 s       | 7410          | 66%       | 0.094          | 0.0007 |
| `qwen35-thinking` | Qwen3.5-0.8B | 64      | 8192         | on       | 5.3 min                      | 151 s      | 7756–7953     | 94%       | 0.0            | 0.0005 |
| `qwen3-small`     | Qwen3-0.6B   | 64      | 2048         | off      | 21–38 s                      | 10–14 s    | 640–1005      | 0–5%      | 0.067 (0–0.17) | 0.0005 |
| `qwen35-gsm8k`    | Qwen3.5-0.8B | 64      | 2048         | off      | 29–42 s                      | 18–22 s    | 651–1073      | 16–31%    | 0.36–0.63      | 0.0004 |

`full` overflows SGLang's KV cache: throughput fell from about 1,700 to 580 tokens/s as requests were evicted and requeued, and the first rollout never finished. `qwen3-thinking` keeps its 64 answers in the cache (0.7 memory fraction holds about 600k tokens), but throughput still falls from about 1,400 to 300 tokens/s as answers grow, because every decode step reads the whole cache. `qwen3-small` runs 20 rollouts in about 20 minutes; its reward stays flat, as expected for so few steps.

Qwen3.5-0.8B generates the same 64 thinking answers 3.3× faster than Qwen3-0.6B (5.3 vs 17 minutes): with linear attention in 18 of 24 layers, its throughput only falls from about 2,200 to 1,150 tokens/s over a rollout. Its training step is slower (151 s, after a one-time 190 s while Triton kernels compile, against 99 s), but a full iteration still takes 10.5 minutes against about 19. It thinks longer, though: 94% of answers hit 8k before finishing, so every reward was 0 and there was nothing to learn. Training it with thinking on needs a longer cap.

On GSM8K, Qwen3.5-0.8B finally gets a learning signal: 36–63% of answers are correct and `grad_norm` is 1.5–2.4 on every step, against 0 on dapo. The data column of the table is dapo except for this row.

Training-step time by attention backend, on the 64-sample, 2k-token batch with thinking on:

| Attention backend  | Kernel TE runs   | Train step | KL     | Result                             |
|--------------------|------------------|------------|--------|------------------------------------|
| `flash`, unpatched | FlashAttention 4 | 29 s       | 9.7    | wrong attention                    |
| `auto`             | cuDNN fused      | –          | –      | backward fails on packed sequences |
| `unfused`          | plain PyTorch    | 152–163 s  | 0.0008 | correct, slow                      |
| `flash`, patched   | FlashAttention 2 | 24–25 s    | 0.0007 | correct                            |

## Managing Jobs

The launcher submits training as a Ray job. With the venv active, list running jobs (`ray job list` prints raw objects, including each job's full `train.py` command line):

    ray list jobs --filter "status=RUNNING" --format json | jq -r '.[] | "\(.submission_id)  \(.status)"'

Inspect, follow or stop a job by its submission ID:

    ray job list                     # every job, including finished and stopped ones
    ray job status <submission_id>   # one job's status
    ray job logs -f <submission_id>  # follow a job's output
    ray job stop <submission_id>
