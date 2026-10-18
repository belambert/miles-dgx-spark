#!/usr/bin/env bash
# GRPO with Miles on one DGX Spark: picks a preset, then launches scripts/run_qwen3_dense.py.
#
#   ./run.sh <preset> [launcher args...]     train; writes to runs/<preset> and resumes from there
#   ./run.sh prepare <model>                 download a model and the datasets, convert to torch_dist
#   MAX_LEN=8192 ./run.sh qwen35-2b          any setting below can be overridden from the environment
#
# Presets (see README, Measurements, for how each has done):
#   quickstart       upstream quick start on Qwen3-0.6B: thinking, 8k, 32 prompts, 3000 rollouts, AIME eval
#   full             quickstart's batch for 20 rollouts, no eval; overflows the KV cache, 40+ min per rollout
#   qwen3-small      Qwen3-0.6B, no thinking, 2k                  the fast end-to-end check
#   qwen3-thinking   Qwen3-0.6B, thinking, 8k
#   qwen35-thinking  Qwen3.5-0.8B, thinking, 8k                   most answers still thinking at 8k
#   qwen35-dapo      Qwen3.5-0.8B, no thinking, 4k                never finishes a dapo problem
#   qwen35-gsm8k     Qwen3.5-0.8B, no thinking, 2k, GSM8K
#   qwen35-2b        Qwen3.5-2B, no thinking, 4k, low-memory optimizer
#   qwen35-4b        Qwen3.5-4B, no thinking, 4k, low-memory optimizer; runs out of memory in its first step
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")
PREFIX=${PREFIX:-$HERE/.venv}
SRC=${SRC:-$HERE/miles-src}

usage() { sed -n '3,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 1; }

prepare() {
  local model=$1
  source "$PREFIX/bin/activate"
  hf download "Qwen/$model" --local-dir "$HERE/models/$model"
  for ds in dapo-math-17k aime-2024 gsm8k; do
    hf download --repo-type dataset "zhuzilin/$ds" --local-dir "$HERE/datasets/$ds"
  done
  # needs the GPU free: stop any running job first
  cd "$SRC/miles"
  PYTHONPATH="$SRC/Megatron-LM" python tools/convert_hf_to_torch_dist.py \
    $(python miles/utils/external_utils/model_args_utils.py "${model/Qwen/qwen}") \
    --hf-checkpoint "$HERE/models/$model" --save "$HERE/models/${model}_torch_dist"
}

preset=${1:-}
[ -n "$preset" ] || usage
shift
[ "$preset" = prepare ] && { prepare "${1:?usage: ./run.sh prepare <model>}"; exit; }

# preset settings; values already set in the environment win
case $preset in
  quickstart)      : "${THINKING:=on}" "${MAX_LEN:=8192}" "${PROMPTS:=32}" "${ROLLOUTS:=3000}" "${EVAL:=on}" "${SGLANG_MEM:=0.5}" ;;
  full)            : "${THINKING:=on}" "${MAX_LEN:=8192}" "${PROMPTS:=32}" "${SGLANG_MEM:=0.5}" ;;
  qwen3-small)     ;;
  qwen3-thinking)  : "${THINKING:=on}" "${MAX_LEN:=8192}" ;;
  qwen35-thinking) : "${MODEL:=Qwen3.5-0.8B}" "${THINKING:=on}" "${MAX_LEN:=8192}" ;;
  qwen35-dapo)     : "${MODEL:=Qwen3.5-0.8B}" "${MAX_LEN:=4096}" ;;
  qwen35-gsm8k)    : "${MODEL:=Qwen3.5-0.8B}" "${DATASET:=gsm8k}" ;;
  qwen35-2b)       : "${MODEL:=Qwen3.5-2B}" "${MAX_LEN:=4096}" "${SGLANG_MEM:=0.5}" "${LOW_MEM:=1}" ;;
  qwen35-4b)       : "${MODEL:=Qwen3.5-4B}" "${MAX_LEN:=4096}" "${SGLANG_MEM:=0.3}" "${LOW_MEM:=1}" "${MAX_TOKENS_PER_GPU:=4096}" "${LOGPROB_CHUNK:=2048}" ;;
  *)               usage ;;
esac

# defaults
: "${MODEL:=Qwen3-0.6B}"           # a --model-name of run_qwen3_dense.py
: "${DATASET:=dapo}"               # dapo or gsm8k
: "${THINKING:=off}"               # Qwen's chat-template thinking
: "${MAX_LEN:=2048}"               # response cap in tokens
: "${PROMPTS:=8}"                  # prompts per rollout, 8 samples each
: "${ROLLOUTS:=20}"
: "${EVAL:=off}"                   # AIME eval before training and every 20 rollouts
: "${SGLANG_MEM:=0.7}"             # SGLang's share of the 128 GB, for weights and KV cache
: "${LOW_MEM:=0}"                  # 1: bf16 Adam moments, for models of 2B and up
: "${MAX_TOKENS_PER_GPU:=9216}"    # training micro-batch budget
: "${LOGPROB_CHUNK:=4096}"         # unchunked, one 16k sequence over a 248k vocab is 15 GiB of fp32 logits
# deepscaler grades only text after </think>, which with thinking off is in the prompt
: "${REWARD:=$([ "$THINKING" = on ] && echo deepscaler || echo math)}"
OUTPUT_DIR=${OUTPUT_DIR:-$HERE/runs/$preset}

extra="--tensor-model-parallel-size 1 --rollout-max-response-len $MAX_LEN \
--rollout-batch-size $PROMPTS --global-batch-size $((PROMPTS * 8)) \
--sglang-mem-fraction-static $SGLANG_MEM --max-tokens-per-gpu $MAX_TOKENS_PER_GPU \
--log-probs-chunk-size $LOGPROB_CHUNK --rm-type $REWARD"
[ "$THINKING" = off ] && extra+=" --apply-chat-template-kwargs '{\"enable_thinking\": false}'"
[ "$DATASET" = gsm8k ] && extra+=" --prompt-data $HERE/datasets/gsm8k/train.parquet --input-key messages --label-key label"
# Qwen3.5's prefix cache keeps ~3 linear-attention state slots per request, capping concurrency; hits are ~4%
[[ $MODEL == Qwen3.5-* ]] && extra+=" --sglang-disable-radix-cache"
[ "$LOW_MEM" = 1 ] && extra+=" --use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"

echo "$preset: $MODEL on $DATASET, thinking $THINKING, ${MAX_LEN} tokens, $PROMPTS prompts x 8, $ROLLOUTS rollouts," \
  "eval $EVAL, reward $REWARD, SGLang $SGLANG_MEM -> $OUTPUT_DIR"

cd "$SRC/miles"
exec "$PREFIX/bin/python" scripts/run_qwen3_dense.py --model-name "$MODEL" --num-gpus-per-node 1 \
  --model-dir "$HERE/models" --data-dir "$HERE/datasets" --output-dir "$OUTPUT_DIR" \
  --megatron-path "$SRC/Megatron-LM" --num-rollout "$ROLLOUTS" \
  "$([ "$EVAL" = on ] && echo --enable-eval || echo --no-enable-eval)" --dump-details \
  --extra-args "$extra" "$@"
