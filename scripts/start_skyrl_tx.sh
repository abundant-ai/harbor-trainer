#!/bin/bash
#
# Start skyrl-tx server for training
#
# Usage:
#   ./scripts/start_skyrl_tx.sh                    # Qwen3-4B on 1 GPU
#   ./scripts/start_skyrl_tx.sh Qwen/Qwen3-8B 2    # Qwen3-8B on 2 GPUs
#   ./scripts/start_skyrl_tx.sh Qwen/Qwen3-30B-A3B 4  # MoE on 4 GPUs
#

set -e

MODEL_NAME="${1:-Qwen/Qwen3-4B}"
TP_SIZE="${2:-1}"
PORT="${PORT:-8000}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_TX_DIR="${SCRIPT_DIR}/../SkyRL/skyrl-tx"

if [ ! -d "$SKYRL_TX_DIR" ]; then
    echo "ERROR: SkyRL/skyrl-tx not found at $SKYRL_TX_DIR"
    echo "Make sure you have SkyRL cloned in the project root."
    exit 1
fi

echo "=============================================="
echo "Starting skyrl-tx server"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Tensor Parallel Size: ${TP_SIZE}"
echo "Port: ${PORT}"
echo "Max LoRA Rank: ${MAX_LORA_RANK}"
echo "=============================================="
echo ""

cd "$SKYRL_TX_DIR"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' not found. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Start the server
echo "Starting server... (Ctrl+C to stop)"
echo ""

# JAX GPU memory pre-allocation fraction
# JAX pre-allocates a fixed memory pool - model weights + LoRA + optimizer state + KV cache must all fit
# Default 0.7 (~57GB on 80GB GPU) works for Qwen3-4B with LoRA rank 32
# Increase to 0.8+ for larger models or higher LoRA ranks
# Note: KV cache for inference grows with (batch_size * context_length), so use --sample-max-num-sequences
# to limit how many sequences are processed in parallel during generation
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}"
echo "JAX memory fraction: ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo ""

# Max sequences to process in parallel during sampling (inference)
# Lower this if you hit OOM during generation
SAMPLE_MAX_SEQ="${SAMPLE_MAX_SEQ:-2}"

# Micro-batch size for training (gradient accumulation)
# Lower this if you hit OOM during training backward pass
# 0 means process all sequences at once (will OOM with many rollouts)
TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-2}"

echo "Sample max sequences: ${SAMPLE_MAX_SEQ}"
echo "Train micro batch size: ${TRAIN_MICRO_BATCH}"
echo ""

uv run --extra tinker --extra gpu python -m tx.tinker.api \
    --base-model "${MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-lora-rank "${MAX_LORA_RANK}" \
    --port "${PORT}" \
    --gradient-checkpointing \
    --sample-max-num-sequences "${SAMPLE_MAX_SEQ}" \
    --train-micro-batch-size "${TRAIN_MICRO_BATCH}"

