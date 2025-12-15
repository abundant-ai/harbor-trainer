#!/bin/bash
#
# Run training with skyrl-tx backend (uses vLLM for inference)
#
# USAGE (2-GPU setup):
#   Terminal 1: ./scripts/start_vllm.sh       # Starts vLLM on GPU 1
#   Terminal 2: ./scripts/run_skyrl_tx.sh     # Starts training on GPU 0
#
# This script automatically starts skyrl-tx in the background on GPU 0.
# GPU assignments are baked into the scripts for optimal 2-GPU setup.
#

set -e

# Ensure ~/.local/bin is in PATH (where uv is typically installed)
export PATH="$HOME/.local/bin:$PATH"

# Increase file descriptor limit to prevent "Too many open files" errors
ulimit -n 65536

# The tinker SDK requires TINKER_API_KEY to be set, but skyrl-tx ignores it
export TINKER_API_KEY="local"

# Configuration - GPU assignments for 2-GPU setup
SKYRL_TX_GPUS="0"  # GPU 0 for training (skyrl-tx + training script)
TRAIN_GPUS="0"     # GPU 0 for training script (must match skyrl-tx)
# NOTE: vLLM should run on GPU 1 (see start_vllm.sh)

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
TP_SIZE="${TP_SIZE:-1}"
VLLM_URL="${VLLM_URL:-http://localhost:8001}"
SKYRL_TX_URL="${SKYRL_TX_URL:-http://localhost:8000}"
LORA_DIR="${LORA_DIR:-/tmp/lora_models}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_TX_DIR="${SCRIPT_DIR}/../SkyRL/skyrl-tx"

echo "=============================================="
echo "Training with skyrl-tx + vLLM (2-GPU Setup)"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "vLLM URL: ${VLLM_URL} (expected on GPU 1)"
echo "Training GPU: ${SKYRL_TX_GPUS} (skyrl-tx + training)"
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

# Start skyrl-tx in background if not already running
# NOTE: Do NOT set CUDA_VISIBLE_DEVICES globally here - we set it per-process below
if curl -s "${SKYRL_TX_URL}/api/v1/healthz" > /dev/null 2>&1; then
    echo "✓ skyrl-tx server already running"
else
    echo "Starting skyrl-tx server in background..."
    
    # JAX memory allocation - 85% to leave headroom for peak memory usage
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
    
    cd "$SKYRL_TX_DIR"
    # Pin skyrl-tx to specific GPU(s) - MUST be different from vLLM GPU(s)!
    CUDA_VISIBLE_DEVICES="${SKYRL_TX_GPUS}" uv run --extra tinker --extra gpu python -m tx.tinker.api \
        --base-model "${MODEL_NAME}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --max-lora-rank 32 \
        --port 8000 \
        --gradient-checkpointing \
        --train-micro-batch-size 1 \
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

# Pin training script to specific GPU(s) - usually same as skyrl-tx
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" uv run python -m src.train \
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
