#!/usr/bin/env bash
# GRPO with Miles on one DGX Spark: picks a preset, then launches scripts/run_qwen3_dense.py.
#
#   ./run.sh <preset> [launcher args...]     train; writes to runs/<preset> and resumes from there
#   MAX_LEN=8192 ./run.sh qwen35-2b          any setting below can be overridden from the environment
# Run ./prepare.sh <model> first to download the model and datasets.
#
# Presets train on GSM8K without thinking and score a fixed GSM8K test slice every 10 rollouts, except
# quickstart and full (see README, Measurements, for how each has done).
#   quickstart  the upstream recipe on Qwen3-0.6B: dapo-math, thinking, 8k, 32 prompts, 3000 rollouts,
#               AIME eval; hours per rollout here
#   full        quickstart for 20 rollouts without eval; overflows the KV cache, 40+ min per rollout
#   qwen3       Qwen3-0.6B, 2k                                 the fast end-to-end check
#   qwen35-1b   Qwen3.5-0.8B, 4k
#   qwen35-2b   Qwen3.5-2B, 4k, low-memory optimizer
#   qwen35-4b   Qwen3.5-4B, 4k, low-memory optimizer; runs out of memory in its first step
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")
PREFIX=${PREFIX:-$HERE/.venv}
SRC=${SRC:-$HERE/miles-src}
DATA_ROOT=${DATA_ROOT:-$HERE}  # holds models/, datasets/ and runs/; put it on a volume that outlives the machine

usage() { sed -n '3,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 1; }

preset=${1:-}
[ -n "$preset" ] || usage
shift
[ "$preset" = prepare ] && { echo "prepare moved to its own script: ./prepare.sh <model>" >&2; exit 1; }

# preset settings; values already set in the environment win
case $preset in
  quickstart) : "${DATASET:=dapo}" "${THINKING:=on}" "${MAX_LEN:=8192}" "${PROMPTS:=32}" "${ROLLOUTS:=3000}" "${EVAL:=aime}" "${SGLANG_MEM:=0.5}" ;;
  full)       : "${DATASET:=dapo}" "${THINKING:=on}" "${MAX_LEN:=8192}" "${PROMPTS:=32}" "${EVAL:=off}" "${SGLANG_MEM:=0.5}" ;;
  qwen3)      ;;
  qwen35-1b)  : "${MODEL:=Qwen3.5-0.8B}" "${MAX_LEN:=4096}" ;;
  # 0.5 left too little memory to save a checkpoint with thinking at 16k
  qwen35-2b)  : "${MODEL:=Qwen3.5-2B}" "${MAX_LEN:=4096}" "${SGLANG_MEM:=0.4}" "${LOW_MEM:=1}" ;;
  qwen35-4b)  : "${MODEL:=Qwen3.5-4B}" "${MAX_LEN:=4096}" "${SGLANG_MEM:=0.3}" "${LOW_MEM:=1}" "${MAX_TOKENS_PER_GPU:=4096}" "${LOGPROB_CHUNK:=2048}" ;;
  *)          usage ;;
esac

# defaults
: "${MODEL:=Qwen3-0.6B}"           # a --model-name of run_qwen3_dense.py
: "${DATASET:=gsm8k}"              # gsm8k, math (from prepare-math.py) or dapo (the recipe's; too hard under 4B)
: "${THINKING:=off}"               # Qwen's chat-template thinking; on needs 8k+ and is ~10x slower
: "${MAX_LEN:=2048}"               # response cap in tokens
: "${PROMPTS:=8}"                  # prompts per rollout, 8 samples each
: "${ROLLOUTS:=20}"
: "${SAVE_EVERY:=20}"              # rollouts per checkpoint; Miles keeps them all (~20-25 GB each for the 2B)
: "${EVAL:=$([ "$DATASET" = math ] && echo math || echo gsm8k)}"  # gsm8k, math (MATH-500), aime (hours here) or off
: "${EVAL_EVERY:=10}"              # gsm8k and math eval interval, in rollouts; it also runs before training
: "${SGLANG_MEM:=0.7}"             # SGLang's share of the 128 GB, for weights and KV cache
: "${LOW_MEM:=0}"                  # 1: bf16 Adam moments, for models of 2B and up
: "${MAX_TOKENS_PER_GPU:=9216}"    # training micro-batch budget
: "${LOGPROB_CHUNK:=4096}"         # unchunked, one 16k sequence over a 248k vocab is 15 GiB of fp32 logits
# deepscaler grades only text after </think>, which with thinking off is in the prompt
: "${REWARD:=$([ "$THINKING" = on ] && echo deepscaler || echo math)}"
OUTPUT_DIR=${OUTPUT_DIR:-$DATA_ROOT/runs/$preset}

extra="--tensor-model-parallel-size 1 --rollout-max-response-len $MAX_LEN \
--rollout-batch-size $PROMPTS --global-batch-size $((PROMPTS * 8)) \
--sglang-mem-fraction-static $SGLANG_MEM --max-tokens-per-gpu $MAX_TOKENS_PER_GPU \
--log-probs-chunk-size $LOGPROB_CHUNK --rm-type $REWARD --save-interval $SAVE_EVERY"
# explicit both ways: Qwen3's template thinks unless told not to, Qwen3.5's only when told to
extra+=" --apply-chat-template-kwargs '{\"enable_thinking\": $([ "$THINKING" = on ] && echo true || echo false)}'"
[[ $DATASET == gsm8k || $DATASET == math ]] && extra+=" --prompt-data $DATA_ROOT/datasets/$DATASET/train.parquet --input-key messages --label-key label"
# Qwen3.5's prefix cache keeps ~3 linear-attention state slots per request, capping concurrency; hits are ~4%
[[ $MODEL == Qwen3.5-* ]] && extra+=" --sglang-disable-radix-cache"
[ "$LOW_MEM" = 1 ] && extra+=" --use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
if [ "$EVAL" = gsm8k ]; then
  # a fixed 256-problem slice of the test set: all 1319 take ~13 min per eval on the 0.8B
  eval_data=$DATA_ROOT/datasets/gsm8k/test256.parquet
  [ -f "$eval_data" ] || "$PREFIX/bin/python" -c "import pandas as pd; pd.read_parquet('$DATA_ROOT/datasets/gsm8k/test.parquet').head(256).to_parquet('$eval_data')"
elif [ "$EVAL" = math ]; then
  eval_data=$DATA_ROOT/datasets/math/test500.parquet
fi
if [[ $EVAL == gsm8k || $EVAL == math ]]; then
  extra+=" --eval-interval $EVAL_EVERY --eval-prompt-data $EVAL $eval_data --eval-input-key messages --eval-label-key label \
--n-samples-per-eval-prompt 1 --eval-max-response-len $MAX_LEN"
fi

echo "$preset: $MODEL on $DATASET, thinking $THINKING, ${MAX_LEN} tokens, $PROMPTS prompts x 8, $ROLLOUTS rollouts," \
  "eval $EVAL, reward $REWARD, SGLang $SGLANG_MEM -> $OUTPUT_DIR"

# the launcher shells out to ray, hf and the router by name
export PATH=$PREFIX/bin:$PATH
# Ray kills workers at 95% of RAM, which on the Spark's unified memory includes the GPU's; that killed a
# checkpoint save that swap would have carried
export RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms:-0}
cd "$SRC/miles"
exec "$PREFIX/bin/python" scripts/run_qwen3_dense.py --model-name "$MODEL" --num-gpus-per-node 1 \
  --model-dir "$DATA_ROOT/models" --data-dir "$DATA_ROOT/datasets" --output-dir "$OUTPUT_DIR" \
  --megatron-path "$SRC/Megatron-LM" --num-rollout "$ROLLOUTS" \
  "$([ "$EVAL" = aime ] && echo --enable-eval || echo --no-enable-eval)" --dump-details \
  --extra-args "$extra" "$@"
