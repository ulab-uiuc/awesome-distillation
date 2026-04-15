#!/bin/bash

# OPSD Weighted GRPO: GRPO policy gradient with OPSD token-level advantage weighting
# Training dataset: open-thoughts/OpenThoughts-114k
#
# Algorithm:
#   - Standard GRPO with n_samples_per_prompt=8 for proper group normalization
#   - OPSD teacher forward computes per-token signal: log p_T(y_t) - log p_S(y_t)
#   - Signal is smoothed (sliding window avg, window=32) and normalized
#     (mean/std or robust median/MAD)
#   - Token weighting on pg_loss:
#       * Positive advantage responses: up-weight teacher-aligned tokens
#       * Negative advantage responses: flip the signal sign, then up-weight tokens
#         aligned with the negative-advantage direction
#   - NO JSD distillation loss (opsd_jsd_coef=0); teacher signal only for weighting
#
# Usage: bash examples/on_policy_distillation/run-qwen3-1.7B-opsd_weighted_grpo-openthoughts.sh

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

###############################################################################
# Step 0: Preprocess datasets
###############################################################################

PREPROCESS="python3 examples/on_policy_distillation/preprocess_dataset.py"

# ---- Training dataset -------------------------------------------------------
TRAIN_OUT="/root/math/data/train_openthoughts_math.jsonl"
TRAIN_ANSWER_FORMAT="${TRAIN_ANSWER_FORMAT:-boxed}"
EVAL_ANSWER_FORMAT="${EVAL_ANSWER_FORMAT:-boxed}"

python3 examples/on_policy_distillation/filter_openthoughts_math.py \
    --output "$TRAIN_OUT" \
    --answer-format "$TRAIN_ANSWER_FORMAT"

# ---- Eval datasets ----------------------------------------------------------
$PREPROCESS --dataset math-ai/aime24             --split test  --output /root/math/data/eval_aime24.jsonl    --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset math-ai/aime25             --split test  --output /root/math/data/eval_aime25.jsonl    --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset FlagEval/HMMT_2025         --split train --output /root/math/data/eval_hmmt.jsonl      --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset meituan-longcat/AMO-Bench  --split test  --output /root/math/data/eval_amo_bench.jsonl --answer-format "$EVAL_ANSWER_FORMAT"
$PREPROCESS --dataset HuggingFaceH4/MATH-500     --split test  --output /root/math/data/eval_math500.jsonl   --answer-format "$EVAL_ANSWER_FORMAT"

OPSD_GRPO_R2_ALPHA="${OPSD_GRPO_R2_ALPHA:-0.0}"


###############################################################################
# Training arguments
###############################################################################

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3-1.7B
   --ref-load "/root/checkpoints_siqi/Qwen3-1.7B_torch_dist"
   --save /root/slime_siqi/output/Qwen3-1.7B_opsd_masked_grpo_openthoughts/
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
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --over-sampling-batch-size 16

   --global-batch-size 16
   --balance-data
)

RM_ARGS=(
   --custom-rm-path examples.on_policy_distillation.on_policy_self_distillation.reward_func
   --custom-reward-post-process-path examples.on_policy_distillation.on_policy_self_distillation.post_process_rewards
   --reward-key math_reward
)

EVAL_ARGS=(
    --eval-interval 10
    --eval-config examples/on_policy_distillation/eval_config.yaml
    --eval-temperature 0.6
    --log-passrate
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
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

   # OPSD setup: teacher forward for signal, but NO JSD loss
   --use-opd
   --opd-type opsd
   --opd-kl-coef 0.0

   --opsd-teacher-info-mode full

   # No JSD distillation loss — signal is only used for pg_loss masking
   --opsd-jsd-coef 0.0

   # OPSD advantage masking: the core masking feature
   # --opsd-advantage-masking
   # OPSD advantage weighting: configurable multiplier on normalized signal.
   # Default is sigmoid -> 2*sigmoid(normalized_signal), can switch to legacy exp.
   # Sign mode default flips z -> -z for samples with mean advantage < 0.
   # Normalization default is robust in this recipe to reduce outlier sensitivity.
   # Override with env vars:
   #   OPSD_ADVANTAGE_WEIGHTING_FN=exp
   #   OPSD_ADV_WEIGHT_SIGN_MODE=none
   #   OPSD_ADVANTAGE_NORMALIZATION=standard
   --opsd-advantage-weighting
   --opsd-advantage-weighting-fn "${OPSD_ADVANTAGE_WEIGHTING_FN:-sigmoid}"
   --opsd-advantage-weighting-sign-mode "${OPSD_ADV_WEIGHT_SIGN_MODE:-flip_on_negative_advantage}"
   --opsd-advantage-normalization "${OPSD_ADVANTAGE_NORMALIZATION:-robust}"
   --opsd-advantage-weighting-epsilon 0.10
   --opsd-signal-window-size 16

   # Scalar reward bonus: task_reward + alpha * R2, where R2 is prompt-group
   # min-max normalization of mean_t[log p_T(y_t) - log p_S_rollout(y_t)].
   # This is orthogonal to --opsd-advantage-weighting above, which only reweights pg_loss per token.

   --opsd-teacher-think-max-tokens -1

   --opsd-use-ref-as-teacher
   --entropy-coef 0.00

   --use-kl-loss
   --kl-loss-coef 0.0
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-1.7B-opsd_weighted_exp_robust_flipnegative_grpo-openthoughts-nothinking
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
   --use-fault-tolerance
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-probs-chunk-size 512
)


echo "Starting Ray job..."

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
unset RAY_ADDRESS
ray stop --force || true
export CUDA_VISIBLE_DEVICES=4,6,8,9
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


set +e
echo "Submitting Ray job..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": "4,6,8,9"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
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
