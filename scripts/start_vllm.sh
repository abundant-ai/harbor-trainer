#!/bin/bash
#
# Start vLLM server for inference (used with skyrl-tx training)
#
# Usage:
#   ./scripts/start_vllm.sh                         # Qwen3-4B on GPU 0
#   ./scripts/start_vllm.sh Qwen/Qwen3-8B 2 "0,1"   # Qwen3-8B tensor-parallel=2 on GPUs 0,1
#   ./scripts/start_vllm.sh Qwen/Qwen3-30B-A3B 4 "0,1,2,3"  # MoE on 4 GPUs
#

set -e

# Ensure ~/.local/bin is in PATH (where uv is typically installed)
export PATH="$HOME/.local/bin:$PATH"

MODEL_NAME="${1:-Qwen/Qwen3-4B}"
TP_SIZE="${2:-1}"
GPU_IDS="${3:-${GPU_IDS:-0}}"
PORT="${VLLM_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"

# Directory where LoRA adapters will be stored for dynamic loading
LORA_DIR="${LORA_DIR:-/tmp/lora_models}"
mkdir -p "$LORA_DIR"

echo "=============================================="
echo "Starting vLLM inference server"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Tensor Parallel Size: ${TP_SIZE}"
echo "CUDA_VISIBLE_DEVICES: ${GPU_IDS}"
echo "Port: ${PORT}"
echo "Max Model Length: ${MAX_MODEL_LEN}"
echo "Max LoRA Rank: ${MAX_LORA_RANK}"
echo "LoRA Directory: ${LORA_DIR}"
echo "=============================================="
echo ""

# Enable dynamic LoRA loading via filesystem resolver
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_PLUGINS=lora_filesystem_resolver
export VLLM_LORA_RESOLVER_CACHE_DIR="${LORA_DIR}"

# Use vLLM V1 engine (better performance)
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Allow insecure serialization for weight updates (needed for LoRA)
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

echo "Environment configured for dynamic LoRA loading"
echo ""

# GPU memory utilization for vLLM
# When sharing GPU with JAX training, lower (e.g., 0.45)
# When vLLM has dedicated GPU(s), keep high (default: 0.9)
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"

echo "Starting server... (Ctrl+C to stop)"
echo ""

cd "$(dirname "$0")/../SkyRL/skyrl-train"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

uv run --isolated --extra vllm python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype bfloat16 \
    --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
    --enable-lora \
    --max-lora-rank "${MAX_LORA_RANK}" \
    --max-loras 8 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code \
    --disable-log-requests
