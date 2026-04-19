#!/bin/bash

# OPSD hidden_think: privileged answer hint hidden inside teacher's <think> block
# Training dataset: open-thoughts/OpenThoughts-114k
#
# Teacher mode: hidden_think
#   - Student: original problem, enable_thinking=False (no <think> in response)
#   - Teacher: identical user message; answer hint prefilled as "<think>The answer
#              to this problem is X.</think>" before the student response tokens
#   - Loss: forward KL (topk) + ref KL
#
# Compared to answer_only: the privileged info is the same (final answer) but it
# is delivered through the model's own thinking channel rather than appended to
# the user message.  This tests whether "silent" knowledge of the answer can
# distil into a student that never uses thinking mode.
#
# Usage: bash examples/on_policy_distillation/run-qwen3-8B-opsd_hidden_think-openthoughts.sh

set -ex
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/slime_siqi/scripts/models/qwen3-1.7B.sh"

OPD_TOKEN_STATS="${OPD_TOKEN_STATS:-1}"
OPD_TOKEN_STATS_TOPK="${OPD_TOKEN_STATS_TOPK:-50}"
OPD_TOKEN_STATS_REPEAT_NGRAM="${OPD_TOKEN_STATS_REPEAT_NGRAM:-3}"
OPD_TOKEN_STATS_EOS_TOKEN_ID="${OPD_TOKEN_STATS_EOS_TOKEN_ID:-151645}"

if [[ "${OPD_TOKEN_STATS}" != "0" && "${OPD_TOKEN_STATS}" != "1" ]]; then
   echo "Invalid OPD_TOKEN_STATS: ${OPD_TOKEN_STATS} (must be 0 or 1)"
   exit 1
fi
if ! [[ "${OPD_TOKEN_STATS_TOPK}" =~ ^[0-9]+$ ]] || [[ "${OPD_TOKEN_STATS_TOPK}" -le 0 ]]; then
   echo "Invalid OPD_TOKEN_STATS_TOPK: ${OPD_TOKEN_STATS_TOPK} (must be positive integer)"
   exit 1
fi
if ! [[ "${OPD_TOKEN_STATS_REPEAT_NGRAM}" =~ ^[0-9]+$ ]] || [[ "${OPD_TOKEN_STATS_REPEAT_NGRAM}" -lt 2 ]]; then
   echo "Invalid OPD_TOKEN_STATS_REPEAT_NGRAM: ${OPD_TOKEN_STATS_REPEAT_NGRAM} (must be integer >= 2)"
   exit 1
fi
if ! [[ "${OPD_TOKEN_STATS_EOS_TOKEN_ID}" =~ ^-?[0-9]+$ ]] || [[ "${OPD_TOKEN_STATS_EOS_TOKEN_ID}" -lt 0 ]]; then
   echo "Invalid OPD_TOKEN_STATS_EOS_TOKEN_ID: ${OPD_TOKEN_STATS_EOS_TOKEN_ID} (must be non-negative integer)"
   exit 1
fi

TOKEN_STATS_ARGS=()
if [[ "${OPD_TOKEN_STATS}" == "1" ]]; then
   TOKEN_STATS_ARGS+=(
      --opd-token-stats
      --opd-token-stats-topk "${OPD_TOKEN_STATS_TOPK}"
      --opd-token-stats-repeat-ngram "${OPD_TOKEN_STATS_REPEAT_NGRAM}"
      --opd-token-stats-eos-token-id "${OPD_TOKEN_STATS_EOS_TOKEN_ID}"
   )
fi

###############################################################################
# Step 0: Preprocess datasets
###############################################################################

PREPROCESS="python3 examples/on_policy_distillation/preprocess_dataset.py"

# ---- Training dataset -------------------------------------------------------

TRAIN_DATASET="${TRAIN_DATASET:-BytedTsinghua-SIA/DAPO-Math-17k}"
TRAIN_CONFIG="${TRAIN_CONFIG:-}"   # DAPO does not require a config subset
TRAIN_OUT="/root/math/data/train_dapo.jsonl"

# EVAL datasets use boxed by default.
TRAIN_ANSWER_FORMAT="${TRAIN_ANSWER_FORMAT:-boxed}"
EVAL_ANSWER_FORMAT="${EVAL_ANSWER_FORMAT:-boxed}"

TRAIN_ARGS=(--dataset "$TRAIN_DATASET" --split train --output "$TRAIN_OUT" --answer-format "$TRAIN_ANSWER_FORMAT")
[ -n "$TRAIN_CONFIG" ] && TRAIN_ARGS+=(--config "$TRAIN_CONFIG")
$PREPROCESS "${TRAIN_ARGS[@]}"


# ---- Eval datasets ----------------------------------------------------------
$PREPROCESS --dataset math-ai/aime24             --split test  --output /root/math/data/eval_aime24.jsonl    --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset math-ai/aime25             --split test  --output /root/math/data/eval_aime25.jsonl    --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset FlagEval/HMMT_2025         --split train --output /root/math/data/eval_hmmt.jsonl      --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset meituan-longcat/AMO-Bench  --split test  --output /root/math/data/eval_amo_bench.jsonl --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset HuggingFaceH4/MATH-500     --split test  --output /root/math/data/eval_math500.jsonl   --answer-format "$EVAL_ANSWER_FORMAT"


###############################################################################
# Training arguments
###############################################################################

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3-8B
   --ref-load "/root/checkpoints_siqi/Qwen3-8B_torch_dist"
   --save /root/slime_siqi/output/Qwen3-8B_opsd_answeronly_dapo/
   --save-interval 2000
)


ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_OUT"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   --num-rollout 1000
   --rollout-batch-size 128
   --n-samples-per-prompt 1
   --rollout-max-response-len 4096
   --rollout-temperature 1.0
   --over-sampling-batch-size 128

   --global-batch-size 64
   --balance-data
)

RM_ARGS=(
   --custom-rm-path examples.on_policy_distillation.on_policy_self_distillation.reward_func
   --custom-reward-post-process-path examples.on_policy_distillation.on_policy_self_distillation.post_process_rewards
   --reward-key math_reward
)

EVAL_ARGS=(
    --eval-interval 100
    --eval-config examples/on_policy_distillation/eval_config.yaml
    --log-passrate
   #  --skip-eval-before-train
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type opsd
   --opd-kl-coef 0.0

   --opsd-teacher-info-mode answer_only

   --opsd-jsd-coef 1.0

   # Top-k forward KL，保留 top 100 个 teacher tokens
   --opsd-loss-type topk_tail_kl
   --opsd-topk 50
   # --opsd-token-clip 0.15
   # --opsd-topk-source student
   # choices=["jsd", "reverse_kl", "forward_kl", "wiener_kl", 
         #   +"topk_forward_kl", "topk_tail_kl", "decoded_kl", "kl_biased_ppo"],
                                                                             
   --opsd-kl-ppo-eta 1.0
   --opsd-kl-ppo-tau 0.0

   --opsd-pure-mode

   --opsd-teacher-think-max-tokens -1 
   #--opsd-teacher-think-max-tokens 的作用是：限制 teacher 侧 <think>...</think> 里可放入的 token 上限。

   --opsd-use-ref-as-teacher
   ${TOKEN_STATS_ARGS[@]}
   --entropy-coef 0.00

   # --use-kl-loss
   # --kl-loss-coef 0.05

   # --use-tis
   # --tis-clip 2.0
   # --tis-clip-low 0.0
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-6
   --lr-decay-style cosine
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-8B-opsd_answeronly_dapo
   --wandb-key 2ed6f8544ac3e30d5c08879166cc10d9c6232448
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.78
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-probs-chunk-size 512
)


echo "Starting Ray job..."

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
unset RAY_ADDRESS
ray stop --force || true
export CUDA_VISIBLE_DEVICES=1,6,7
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 3 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


set +e
echo "Submitting Ray job..."
RAY_JOB_ID="qwen3_8b_opsd_answeronly_dapo_$(date +%s)"
ray job submit --submission-id "${RAY_JOB_ID}" --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": "1,6,7"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

RAY_EXIT_CODE=$?

set -e
echo "Ray job exited with code: ${RAY_EXIT_CODE}"
sleep 10

####clear after training
ray stop --force
# pkill -9 ray
# pkill -9 python
# sleep 3
# pkill -9 ray
# pkill -9 python
