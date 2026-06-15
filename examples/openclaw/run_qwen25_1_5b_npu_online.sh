#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export USE_OPTIMIZED_MODEL="${USE_OPTIMIZED_MODEL:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-14400}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-HOST}"
export ACL_DEVICE_SYNC_TIMEOUT="${ACL_DEVICE_SYNC_TIMEOUT:-14400}"
export HCCL_EVENT_TIMEOUT="${HCCL_EVENT_TIMEOUT:-14500}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
export ACL_STREAM_TIMEOUT="${ACL_STREAM_TIMEOUT:-14500000}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
ADMIN_API_KEY="${ADMIN_API_KEY:-sk-openclaw-npu-dev}"
PROVIDER_API_KEY="${PROVIDER_API_KEY:-any-placeholder-key}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-openclaw-online-npu}"
TRIAL_NAME="${TRIAL_NAME:-qwen25-1_5b}"
TOTAL_TRAIN_STEPS="${TOTAL_TRAIN_STEPS:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

export AREAL_PRM_JUDGE_BASE_URL="${AREAL_PRM_JUDGE_BASE_URL:-}"
export AREAL_PRM_JUDGE_API_KEY="${AREAL_PRM_JUDGE_API_KEY:-}"
export AREAL_PRM_JUDGE_MODEL="${AREAL_PRM_JUDGE_MODEL:-default}"
export AREAL_PRM_JUDGE_TIMEOUT="${AREAL_PRM_JUDGE_TIMEOUT:-30}"
export AREAL_PROVIDER_IDLE_TIMEOUT="${AREAL_PROVIDER_IDLE_TIMEOUT:-300}"
export AREAL_PROVIDER_IDLE_CHECK_INTERVAL="${AREAL_PROVIDER_IDLE_CHECK_INTERVAL:-30}"

echo "Use the gateway URL printed by AReaL as the provider baseUrl."
echo "For OpenAI-compatible clients, use baseUrl=http://<gateway>/v1 and model=default."
echo "If the client requires an apiKey, any placeholder is fine, e.g. ${PROVIDER_API_KEY}."
echo "Idle sessions auto-end after ${AREAL_PROVIDER_IDLE_TIMEOUT}s."

python3 examples/openclaw/train.py \
  --config examples/openclaw/config_qwen25_1_5b_npu.yaml \
  scheduler.type=local \
  experiment_name="${EXPERIMENT_NAME}" \
  trial_name="${TRIAL_NAME}" \
  total_train_steps="${TOTAL_TRAIN_STEPS}" \
  actor.path="${MODEL_PATH}" \
  ref.path="${MODEL_PATH}" \
  tokenizer_path="${MODEL_PATH}" \
  sglang.model_path="${MODEL_PATH}" \
  vllm.model="${MODEL_PATH}" \
  train_dataset.batch_size="${TRAIN_BATCH_SIZE}" \
  rollout.consumer_batch_size="${TRAIN_BATCH_SIZE}" \
  gconfig.max_new_tokens="${MAX_NEW_TOKENS}" \
  actor.max_new_tokens="${MAX_NEW_TOKENS}" \
  rollout.agent.admin_api_key="${ADMIN_API_KEY}" \
  rollout.admin_api_key="${ADMIN_API_KEY}"
