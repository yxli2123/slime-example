#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
pkill -9 python3
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 python3

set -ex


cd /root/slime

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ========== Download HF models ==========
source "/root/slime/scripts/models/qwen3-14B.sh"

HF_REPO_ID=Qwen/Qwen3-14B
HF_MODEL_PATH=/root/model-hf/${HF_REPO_ID}
MEGATRON_MODEL_PATH=/root/model-megatron/${HF_REPO_ID}
hf download ${HF_REPO_ID} --local-dir ${HF_MODEL_PATH}

# Convert HF models to Megatron models.
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint ${HF_MODEL_PATH} \
  --save ${MEGATRON_MODEL_PATH}


# ========== Download SFT data ==========
HF_TRAIN_DATA=yxli2123/mt-2turns-rl-15k-1220-train
HF_EVAL_DATA=yxli2123/mt-20k-turn2-1207-eval
LOCAL_TRAIN_DATA=/root/data/${HF_TRAIN_DATA}.jsonl
LOCAL_EVAL_DATA=/root/data/${HF_EVAL_DATA}.jsonl

python ${SCRIPT_DIR}/convert_to_slime_data.py \
  --hf-train-data ${HF_TRAIN_DATA} \
  --hf-eval-data ${HF_EVAL_DATA} \
  --local-train-data ${LOCAL_TRAIN_DATA} \
  --local-eval-data ${LOCAL_EVAL_DATA}


# ========== Experiment Configs ==========
PROJ_NAME=${PROJ_NAME:-qwen3-14b-mt}
EXPT_NAME=${EXPT_NAME:-experiment4399}

CKPT_ROOT=${CKPT_ROOT:-/root/checkpoint}
CKPT_DIR=${CKPT_ROOT}/${PROJ_NAME}/${EXPT_NAME}
CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --ref-load ${MEGATRON_MODEL_PATH}
   --load ${CKPT_DIR}
   --save ${CKPT_DIR}
   --save-interval 32
)

# Note: set rollout-batch-size * n-samples-per-prompt == global-batch-size to avoid potential checkpointing issues
ROLLOUT_ARGS=(
   --prompt-data ${LOCAL_TRAIN_DATA}
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 3200
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 6144
   --rollout-temperature 0.6

   --global-batch-size 1024
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 32
   --eval-prompt-data ${LOCAL_EVAL_DATA}
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 6144
   --eval-top-p 0.8
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${PROJ_NAME}
   --wandb-group ${EXPT_NAME}
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-tensor-parallel-size 1
   --sglang-mem-fraction-static 0.8
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path multi_if_generate.generate
   --custom-rm-path multi_if_reward.reward_func
   --judge_api_key_path ${API_KEY_PATH}
   --base_url ${BASE_URL}
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
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
   ${CUSTOM_ARGS[@]}

