#!/bin/bash

# Math SFT for Qwen2.5-1.5B-base on pre-generated teacher data from
# Qwen2.5-Math-7B-Instruct over OpenThoughts math problems.
#
# Dataset: /root/math/data/sft_qwen25math_7b_generated_openthoughts_math.parquet
#   columns: messages (list[{role, content}]), label, metadata
#
# Usage:
#   bash examples/on_policy_distillation/run-qwen2.5-1.5Bbase-sft-qwen25math7b_openthoughts_math.sh
#   MAX_SFT_SAMPLES=32 NUM_EPOCH=1 bash examples/on_policy_distillation/run-qwen2.5-1.5Bbase-sft-qwen25math7b_openthoughts_math.sh

set -ex
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-/root/slime_siqi/scripts/models/qwen2.5-math-1.5B.sh}"
source "${MODEL_ARGS_SCRIPT}"

###############################################################################
# Training arguments
###############################################################################

# Pre-generated parquet — no preprocessing step needed.
SFT_DATA_OUT="${SFT_DATA_OUT:-/root/checkpoints_siqi/data/sft_qwen25math_7b_generated_openthoughts_math.parquet}"

SFT_SAVE="${SFT_SAVE:-/root/slime_siqi/output/Qwen2.5-1.5B-base_sft_qwen25math7b_openthoughts_math/}"
SFT_LOAD="${SFT_LOAD:-}"
SFT_REF_LOAD="${SFT_REF_LOAD:-/root/checkpoints_siqi/models--Qwen--Qwen2.5-Math-1.5B_torch_dist}"
SFT_HF_CHECKPOINT="${SFT_HF_CHECKPOINT:-/root/checkpoints_siqi/models--Qwen--Qwen2.5-Math-1.5B}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"

CKPT_ARGS=(
   --hf-checkpoint "${SFT_HF_CHECKPOINT}"
   --ref-load "${SFT_REF_LOAD}"
   --save "${SFT_SAVE}"
   --save-interval "${SAVE_INTERVAL}"
)
if [[ -n "${SFT_LOAD}" ]]; then
   CKPT_ARGS+=(--load "${SFT_LOAD}")
fi

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
NUM_EPOCH="${NUM_EPOCH:-3}"
ROLLOUT_SEED="${ROLLOUT_SEED:-1234}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-1234}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${SFT_DATA_OUT}"
   --input-key messages
   --rollout-seed "${ROLLOUT_SEED}"
   --rollout-shuffle
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --loss-mask-type qwen
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

if [[ -n "${NUM_ROLLOUT:-}" ]]; then
  SFT_ARGS+=(--num-rollout "${NUM_ROLLOUT}")
else
  SFT_ARGS+=(--num-epoch "${NUM_EPOCH}")
fi

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
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-5e-6}"
   --lr-decay-style cosine
   --min-lr "${MIN_LR:-5e-7}"
   --lr-warmup-fraction "${LR_WARMUP_FRACTION:-0.1}"
   --weight-decay "${WEIGHT_DECAY:-0.1}"
   --adam-beta1 "${ADAM_BETA1:-0.9}"
   --adam-beta2 "${ADAM_BETA2:-0.98}"
)

WANDB_ARGS=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-slime-dev}"
      --wandb-group "${WANDB_GROUP:-qwen2.5-1.5B-base-sft-qwen25math7b-openthoughts-math}"
   )
   if [[ -n "${WANDB_KEY:-}" ]]; then
      WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
   fi
fi

MISC_ARGS=(
   --seed "${EXPERIMENT_SEED}"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

###############################################################################
# Launch
###############################################################################
# mv /root/checkpoint_siqi /root/checkpoints_siqi/data/sft_qwen25math_7b_generated_openthoughts_math.parquet


export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
unset RAY_ADDRESS
ray stop --force || true

SFT_CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-4}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"CUDA_VISIBLE_DEVICES\": \"${SFT_CUDA_VISIBLE_DEVICES}\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\"
  }
}"

set +e
echo "Submitting Ray SFT job..."
RAY_JOB_ID="qwen25_1p5b_base_sft_qwen25math7b_openthoughts_math_$(date +%s)"
ray job submit --submission-id "${RAY_JOB_ID}" --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}

RAY_EXIT_CODE=$?
set -e
echo "Ray SFT job exited with code: ${RAY_EXIT_CODE}"
sleep 10
ray stop --force
exit "${RAY_EXIT_CODE}"
