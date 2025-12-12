#!/bin/bash
#
# Run training with skyrl-tx backend (uses vLLM for inference)
#
# PREREQUISITES:
#   Terminal 1: ./scripts/start_vllm.sh
#   Terminal 2: ./scripts/run_skyrl_tx.sh
#
# This script automatically starts skyrl-tx in the background.
#

set -e

# Ensure ~/.local/bin is in PATH (where uv is typically installed)
export PATH="$HOME/.local/bin:$PATH"

# Increase file descriptor limit to prevent "Too many open files" errors
ulimit -n 65536

# The tinker SDK requires TINKER_API_KEY to be set, but skyrl-tx ignores it
export TINKER_API_KEY="local"

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
TP_SIZE="${TP_SIZE:-1}"
VLLM_URL="${VLLM_URL:-http://localhost:8001}"
SKYRL_TX_URL="${SKYRL_TX_URL:-http://localhost:8000}"
LORA_DIR="${LORA_DIR:-/tmp/lora_models}"
TRAIN_GPUS="${TRAIN_GPUS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_TX_DIR="${SCRIPT_DIR}/../SkyRL/skyrl-tx"

echo "=============================================="
echo "Training with skyrl-tx + vLLM"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "vLLM URL: ${VLLM_URL}"
echo "Training CUDA_VISIBLE_DEVICES: ${TRAIN_GPUS}"
echo "=============================================="
echo ""

# Check vLLM is running
if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server not responding at ${VLLM_URL}"
    echo ""
    echo "Start vLLM first in another terminal:"
    echo "  ./scripts/start_vllm.sh"
    exit 1
fi
echo "✓ vLLM server is running"

# Pin training to specific GPU(s) to avoid contention with vLLM
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"

# Start skyrl-tx in background if not already running
if curl -s "${SKYRL_TX_URL}/api/v1/healthz" > /dev/null 2>&1; then
    echo "✓ skyrl-tx server already running"
else
    echo "Starting skyrl-tx server in background..."
    
    # JAX memory allocation
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
    
    cd "$SKYRL_TX_DIR"
    uv run --extra tinker --extra gpu python -m tx.tinker.api \
        --base-model "${MODEL_NAME}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --max-lora-rank 32 \
        --port 8000 \
        --gradient-checkpointing \
        --train-micro-batch-size 2 \
        --external-inference-url "${VLLM_URL}" \
        --external-inference-lora-base "${LORA_DIR}" \
        > /tmp/skyrl_tx.log 2>&1 &
    
    SKYRL_TX_PID=$!
    cd - > /dev/null
    
    # Wait for it to start
    echo "Waiting for skyrl-tx to start (PID: $SKYRL_TX_PID)..."
    for i in {1..30}; do
        if curl -s "${SKYRL_TX_URL}/api/v1/healthz" > /dev/null 2>&1; then
            echo "✓ skyrl-tx server is running"
            break
        fi
        if ! kill -0 $SKYRL_TX_PID 2>/dev/null; then
            echo "ERROR: skyrl-tx failed to start. Check /tmp/skyrl_tx.log"
            cat /tmp/skyrl_tx.log
            exit 1
        fi
        sleep 1
    done
    
    if ! curl -s "${SKYRL_TX_URL}/api/v1/healthz" > /dev/null 2>&1; then
        echo "ERROR: skyrl-tx failed to start within 30 seconds"
        echo "Log output:"
        cat /tmp/skyrl_tx.log
        exit 1
    fi
    
    # Cleanup on exit
    trap "echo 'Stopping skyrl-tx...'; kill $SKYRL_TX_PID 2>/dev/null" EXIT
fi

echo ""

# Run training
cd "$SCRIPT_DIR/.."

python -m src.train \
  backend=skyrl-tx \
  skyrl_tx_url=${SKYRL_TX_URL} \
  model_name=${MODEL_NAME} \
  tasks_dir=./datasets/extracted_tasks \
  learning_rate=2e-4 \
  batch_size=1 \
  group_size=4 \
  eval_split=0.2 \
  eval_group_size=4 \
  n_parallel_envs=8 \
  max_tokens=2048 \
  temperature=0.7 \
  context_limit=32000 \
  proactive_summarization_threshold=4000 \
  enable_summarize=true \
  n_epochs=1 \
  num_substeps=4 \
  remove_constant_reward_groups=true \
  normalize_advantages_by_std=true \
  loss_fn=ppo \
  environment_type=docker \
  wandb_project=harbor-training \
  wandb_name=qwen-3-4b-skyrl-tx
