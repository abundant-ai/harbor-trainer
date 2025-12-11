#!/bin/bash
#
# Start skyrl-tx server for training (uses vLLM for inference)
#
# Usage:
#   # First start vLLM in another terminal:
#   ./scripts/start_vllm.sh Qwen/Qwen3-4B 1
#
#   # Then start skyrl-tx:
#   ./scripts/start_skyrl_tx.sh Qwen/Qwen3-4B 1
#

set -e

# Ensure ~/.local/bin is in PATH (where uv is typically installed)
export PATH="$HOME/.local/bin:$PATH"

MODEL_NAME="${1:-Qwen/Qwen3-4B}"
TP_SIZE="${2:-1}"
PORT="${PORT:-8000}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"

# vLLM inference settings
VLLM_URL="${VLLM_URL:-http://localhost:8001}"
LORA_DIR="${LORA_DIR:-/tmp/lora_models}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_TX_DIR="${SCRIPT_DIR}/../SkyRL/skyrl-tx"

if [ ! -d "$SKYRL_TX_DIR" ]; then
    echo "ERROR: SkyRL/skyrl-tx not found at $SKYRL_TX_DIR"
    echo "Make sure you have SkyRL cloned in the project root."
    exit 1
fi

echo "=============================================="
echo "Starting skyrl-tx server (training)"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Tensor Parallel Size: ${TP_SIZE}"
echo "Port: ${PORT}"
echo "Max LoRA Rank: ${MAX_LORA_RANK}"
echo "vLLM URL: ${VLLM_URL}"
echo "LoRA Directory: ${LORA_DIR}"
echo "=============================================="
echo ""

cd "$SKYRL_TX_DIR"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' not found. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Verify vLLM is running
echo "Checking vLLM server at ${VLLM_URL}..."
if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server not responding at ${VLLM_URL}"
    echo ""
    echo "Start vLLM first in another terminal:"
    echo "  ./scripts/start_vllm.sh ${MODEL_NAME} ${TP_SIZE}"
    exit 1
fi
echo "✓ vLLM server is running"
echo ""

# JAX memory - needs space for training (model + optimizer + gradients)
# When sharing GPU with vLLM, use ~0.45 to avoid conflicts
# vLLM should also use ~0.45
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
echo "JAX memory fraction: ${XLA_PYTHON_CLIENT_MEM_FRACTION}"

# Micro-batch size for training (gradient accumulation)
TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-2}"
echo "Train micro batch size: ${TRAIN_MICRO_BATCH}"

echo ""
echo "Starting server... (Ctrl+C to stop)"
echo ""

uv run --extra tinker --extra gpu python -m tx.tinker.api \
    --base-model "${MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-lora-rank "${MAX_LORA_RANK}" \
    --port "${PORT}" \
    --gradient-checkpointing \
    --train-micro-batch-size "${TRAIN_MICRO_BATCH}" \
    --external-inference-url "${VLLM_URL}" \
    --external-inference-lora-base "${LORA_DIR}"
