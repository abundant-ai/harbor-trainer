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
# JAX pre-allocates a fixed memory pool - model weights + LoRA + optimizer state must all fit within it
# Default 0.7 (~57GB on 80GB GPU) works for Qwen3-4B with LoRA rank 32
# Increase to 0.8+ for larger models or higher LoRA ranks
# Note: This is memory WITHIN the JAX pool, not total GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}"
echo "JAX memory fraction: ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo ""

uv run --extra tinker --extra gpu python -m tx.tinker.api \
    --base-model "${MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-lora-rank "${MAX_LORA_RANK}" \
    --port "${PORT}" \
    --gradient-checkpointing

