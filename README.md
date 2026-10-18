# miles-dgx-spark

Non-Docker install of [Miles](https://github.com/radixark/miles) for a DGX Spark (GB10, aarch64, CUDA 13), plus a variant for cloud GPU machines.

## Compromises

This started as the upstream [quick start](https://github.com/radixark/miles/blob/main/docs/getting-started/quick-start.md) on one Spark. Running it here takes these changes, which are now the defaults of every `run.sh` preset except `quickstart` and `full` (see Quick Start).

| Setting             | Upstream quick start                        | Here                                                  | Why                                                 |
|---------------------|---------------------------------------------|-------------------------------------------------------|-----------------------------------------------------|
| Install             | Docker image                                | host venv; TE, apex, flash-attn copied from the image | Non-Docker install                                  |
| Model               | Qwen3-4B                                    | Qwen3-0.6B, Qwen3.5-0.8B/2B/4B                        | Fits one GPU with TP 1                              |
| GPUs                | 8 (TP 2, 4 SGLang engines)                  | 1 (TP 1, 1 engine)                                    | A Spark has one GB10                                |
| Attention           | FlashAttention via TE                       | TE patched to use FA2, not FA4                        | TE + FA4 gives wrong attention (TE#3528)            |
| Rollouts            | 3000                                        | 20                                                    | A quick end-to-end check                            |
| Eval                | AIME, before training and every 20 rollouts | GSM8K test slice, every 10                            | AIME takes hours per eval; GSM8K tracks learning    |
| Training data       | dapo-math-17k                               | GSM8K                                                 | Models under 4B rarely finish a dapo problem        |
| Prompts per rollout | 32 (256 samples)                            | 8 (64 samples)                                        | 256 long samples overflow the KV cache              |
| Global batch        | 256                                         | 64                                                    | Matches the smaller rollout                         |
| Max response        | 8192 tokens                                 | 2048–4096 tokens                                      | Answers without thinking fit                        |
| Qwen3 thinking      | on                                          | off                                                   | Thinking runs past 2k tokens, so all rewards were 0 |
| Reward              | `deepscaler`                                | `math`                                                | `deepscaler` needs `</think>` in the response       |

The `quickstart` preset is the upstream recipe, differing only in model size, GPU count and the attention fix, but on one Spark its rollouts take 40+ minutes and its first eval several hours.

## Usage

The installer clones the sources into `./miles-src` and builds a venv in `./.venv`. Miles comes from the `dgx-spark` branch of [our fork](https://github.com/belambert/miles/tree/dgx-spark), which adds single-GPU Qwen3-0.6B, Qwen3.5-0.8B and Qwen3.5-2B recipes, a Qwen3.5-2B model config, and `--attention-backend` and `--dump-details` flags to `scripts/run_qwen3_dense.py`. The installer needs the pinned `radixark/miles` Docker image pulled locally, because a few compiled packages are copied out of it.

    ./install.sh 2>&1 | tee install.log

Check an existing install:

    ./verify.sh

Set `PREFIX` or `SRC` to use other locations, and `DATA_ROOT` to keep `run.sh`'s `models/`, `datasets/` and `runs/` somewhere other than the repo root. The pins in `pins.sh` say which image and commits they match.

### Cloud Install

`cloud/install.sh` installs the same pins on a cloud GPU machine without Docker; see [cloud/README.md](cloud/README.md).

## Quick Start

This is the upstream [quick start](https://github.com/radixark/miles/blob/main/docs/getting-started/quick-start.md) on one Spark, with Qwen3-0.6B instead of Qwen3-4B and everything kept under this repo instead of `/root`. Run the steps from the repo root.

| Role            | Hugging Face repo                                                                | Original                                                                                                                                                  | Contents                                          |
|-----------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| Model           | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)                        | –                                                                                                                                                         | The policy being trained                          |
| Training        | [zhuzilin/dapo-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k) | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k), from the [DAPO](https://dapo-sia.github.io/) RL paper | 17k math problems with checkable answers          |
| Eval            | [zhuzilin/aime-2024](https://huggingface.co/datasets/zhuzilin/aime-2024)         | [BytedTsinghua-SIA/AIME-2024](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024); problems from the MAA's AIME I and II                         | AIME 2024 problems, evaluated on but not trained  |
| Training (easy) | [zhuzilin/gsm8k](https://huggingface.co/datasets/zhuzilin/gsm8k)                 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)                                                                                              | 7.5k grade-school word problems, for small models |

The `zhuzilin` copies are what Miles' recipes download: DAPO-Math-17k deduplicated from 1.79M rows (each problem repeated about 100 times) to 17,398, with prompts asking for `Answer: \boxed{...}` instead of `Answer: ...`; AIME-2024 with each of the 30 problems once instead of 32 times; and GSM8K with a chat-format `messages` column and a system prompt asking for `\boxed{}`.

Download a model and the datasets, and convert the model to Megatron's `torch_dist` format. The rollout engine reads the Hugging Face copy, so both are kept. The conversion needs the GPU, so stop any running job first:

    ./run.sh prepare Qwen3-0.6B

Then train with a preset:

    ./run.sh qwen3

`run.sh` launches Miles' `scripts/run_qwen3_dense.py` with the preset's settings. Each preset is one line near the top of the script, and any setting (`MODEL`, `DATASET`, `THINKING`, `MAX_LEN`, `PROMPTS`, `ROLLOUTS`, `EVAL`, `SGLANG_MEM`, `REWARD`, ...) can be overridden from the environment, for example `MAX_LEN=8192 ./run.sh qwen35-2b` or `DATASET=dapo ./run.sh qwen35-4b`. Arguments after the preset go to the launcher. The first line it prints sums up the run. Every rollout has 8 samples per prompt.

By default each GSM8K run (`EVAL=gsm8k`) scores the model on a fixed 256-problem slice of GSM8K's test set, before training and every `EVAL_EVERY` rollouts (default 10), with one sample per problem at the rollout temperature (1.0) and the training response cap. Unlike the rollout reward, which depends on which 8 training problems came up, this measures the same problems each time, so it shows whether training is helping. A longer run to check that:

    ROLLOUTS=200 ./run.sh qwen35-1b

`EVAL=aime` is the recipe's AIME eval, which `quickstart` uses and which takes hours per eval here; `EVAL=off` skips evals.

| Preset       | Model        | Data  | Thinking | Max response | Prompts | Notes                                                                              |
|--------------|--------------|-------|----------|--------------|---------|------------------------------------------------------------------------------------|
| `quickstart` | Qwen3-0.6B   | dapo  | on       | 8192         | 32      | The upstream recipe: 3000 rollouts, AIME eval; hours per rollout here              |
| `full`       | Qwen3-0.6B   | dapo  | on       | 8192         | 32      | `quickstart` for 20 rollouts, no eval; overflows the KV cache, 40+ min per rollout |
| `qwen3`      | Qwen3-0.6B   | GSM8K | off      | 2048         | 8       | The fast end-to-end check                                                          |
| `qwen35-1b`  | Qwen3.5-0.8B | GSM8K | off      | 4096         | 8       | ~1 min per iteration at 2k, reward 0.28–0.66                                       |
| `qwen35-2b`  | Qwen3.5-2B   | GSM8K | off      | 4096         | 8       | Fits in memory, barely; ~2.5 min per iteration, GSM8K test 0.77 → 0.85–0.91        |
| `qwen35-4b`  | Qwen3.5-4B   | GSM8K | off      | 4096         | 8       | Out of memory in its first optimizer step (on dapo)                                |

Each preset writes to `runs/<preset>` and resumes from the checkpoint there, so a finished run won't train again; delete its directory to start over, for example `rm -rf runs/qwen3`. Checkpoints are saved every `SAVE_EVERY` rollouts (default 20) and never deleted, so raise it for long runs, for example `SAVE_EVERY=50 ROLLOUTS=200 ./run.sh qwen35-2b`. A checkpoint with optimizer state is about 8 GB for the 0.6B and 50–60 GB for the 4B. Each run also saves every rollout's samples in `runs/<preset>/dump_details/rollout_data/<rollout>.pt`, so you can read what the model wrote:

    PYTHONPATH=miles-src/miles .venv/bin/python -c "import torch; s = torch.load('runs/qwen3/dump_details/rollout_data/0.pt', weights_only=False)['samples'][0]; print(s['reward'], s['status'], s['response'])"

Ray's job logs, by contrast, are wiped whenever a new run restarts Ray. Don't send requests to the running SGLang engine to check outputs instead: it releases its GPU memory during each training step, and a request arriving then crashes the run. Set `WANDB_API_KEY` to log to wandb.

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

Runs on one Spark (2026-10-17 and 18), with the TE patch unless noted, mostly from before GSM8K and no thinking became the defaults. "KL" is `train/train_rollout_kl`, the gap between SGLang's and Megatron's log-probs for the same tokens; near zero means training sees the same model SGLang sampled from. Reward is the fraction of answers graded correct per rollout.

| Data  | Model        | Samples | Max response | Thinking | Rollout                      | Train step | Mean response | Truncated | Reward         | KL     |
|-------|--------------|---------|--------------|----------|------------------------------|------------|---------------|-----------|----------------|--------|
| dapo  | Qwen3-0.6B   | 256     | 8192         | on       | not finished, est. 40–50 min | –          | –             | –         | –              | –      |
| dapo  | Qwen3-0.6B   | 64      | 8192         | on       | 17 min                       | 99 s       | 7410          | 66%       | 0.094          | 0.0007 |
| dapo  | Qwen3.5-0.8B | 64      | 8192         | on       | 5.3 min                      | 151 s      | 7756–7953     | 94%       | 0.0            | 0.0005 |
| dapo  | Qwen3-0.6B   | 64      | 2048         | off      | 21–38 s                      | 10–14 s    | 640–1005      | 0–5%      | 0.067 (0–0.17) | 0.0005 |
| GSM8K | Qwen3.5-0.8B | 64      | 2048         | off      | 25–42 s                      | 14–23 s    | 536–1105      | 6–39%     | 0.42 → 0.45    | 0.0004 |
| dapo  | Qwen3.5-2B   | 64      | 4096         | off      | 172–178 s                    | 95–105 s   | 3671–4001     | 84–95%    | 0.016–0.125    | 0.0004 |
| GSM8K | Qwen3.5-2B   | 64      | 4096         | off      | 61–102 s                     | 27–59 s    | 504–1357      | 0–19%     | 0.75 → 0.88    | 0.0002 |

The 256-sample thinking batch overflows SGLang's KV cache: throughput fell from about 1,700 to 580 tokens/s as requests were evicted and requeued, and the first rollout never finished. Qwen3-0.6B with thinking and 64 samples keeps its answers in the cache (0.7 memory fraction holds about 600k tokens), but throughput still falls from about 1,400 to 300 tokens/s as answers grow, because every decode step reads the whole cache. Without thinking at 2k it runs 20 rollouts in about 20 minutes; its reward stays flat, as expected for so few steps.

Qwen3.5-0.8B generates the same 64 thinking answers 3.3× faster than Qwen3-0.6B (5.3 vs 17 minutes): with linear attention in 18 of 24 layers, its throughput only falls from about 2,200 to 1,150 tokens/s over a rollout. Its training step is slower (151 s, after a one-time 190 s while Triton kernels compile, against 99 s), but a full iteration still takes 10.5 minutes against about 19. It thinks longer, though: 94% of answers hit 8k before finishing, so every reward was 0 and there was nothing to learn. Training it with thinking on needs a longer cap.

On GSM8K, Qwen3.5-0.8B finally gets a learning signal: 28–66% of answers are correct and `grad_norm` is 1.2–2.4 on every step, against 0 on dapo. Over 20 rollouts the mean reward went from 0.42 (rollouts 0–9) to 0.45 (10–19), within noise for 8 prompts per rollout. Rollouts with fewer answers cut off at 2k score higher, so a 4k cap would likely help.

Qwen3.5-2B is the largest Qwen3.5 model that trains on one Spark with these settings (the 4B runs out of memory), though system memory is nearly exhausted during its run. On dapo it writes long answers like the 0.8B: 84–95% hit 4k, for a weak signal (1 and 8 correct of 64 in its first two rollouts).

On GSM8K the 2B trained for 200 rollouts in 8.7 hours (2026-10-18), including 21 evals of about 2.5 minutes each; ranges in its row are the 10th to 90th percentile per rollout. The GSM8K test score rose from 0.77 before training to 0.86 after 90 rollouts and peaked at 0.91 after 100, then stayed between 0.85 and 0.89 through rollout 200, within the noise of a 256-problem eval. Much of the gain came from finishing more answers: the share of test answers cut off at 4k fell from 9–19% over the first 80 rollouts to 2–7% from 90 on, and their mean length from about 1,050 to 800 tokens. The training reward, averaged over blocks of 20 rollouts, went from 0.75 to 0.86 by rollouts 80–99 and stayed at 0.86–0.88 after that. Checkpoints are 22 GB each, about 25 s to save.

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
