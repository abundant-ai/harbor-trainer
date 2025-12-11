#!/bin/bash
#
# Run training with skyrl-tx backend (self-hosted)
#
# PREREQUISITES:
# 1. Start skyrl-tx server in a separate terminal:
#    cd SkyRL/skyrl-tx
#    uv run --extra tinker --extra gpu python -m tx.tinker.api \
#      --base-model Qwen/Qwen3-4B \
#      --tensor-parallel-size 1 \
#      --max-lora-rank 32
#
# 2. Verify server is running:
#    curl http://localhost:8000/api/v1/healthz
#    # Should return: {"status":"ok"}
#

# Increase file descriptor limit to prevent "Too many open files" errors
ulimit -n 65536

# Load .env and ensure TINKER_API_KEY is set (required by tinker SDK even for local server)
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a
fi

# For skyrl-tx, we need TINKER_API_KEY set but it can be a dummy value
if [ -z "$TINKER_API_KEY" ] || [ "$TINKER_API_KEY" = "your-api-key-here" ]; then
    export TINKER_API_KEY="dummy-key-for-skyrl-tx"
fi

# Configuration
SKYRL_TX_URL="${SKYRL_TX_URL:-http://localhost:8000}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"

echo "=============================================="
echo "Training with skyrl-tx backend"
echo "Server URL: ${SKYRL_TX_URL}"
echo "Model: ${MODEL_NAME}"
echo "=============================================="

# Verify skyrl-tx server is running
if ! curl -s "${SKYRL_TX_URL}/api/v1/healthz" > /dev/null 2>&1; then
    echo "ERROR: skyrl-tx server not responding at ${SKYRL_TX_URL}"
    echo ""
    echo "Start the server first:"
    echo "  cd SkyRL/skyrl-tx"
    echo "  uv run --extra tinker --extra gpu python -m tx.tinker.api \\"
    echo "    --base-model ${MODEL_NAME} \\"
    echo "    --tensor-parallel-size 1 \\"
    echo "    --max-lora-rank 32"
    exit 1
fi

echo "✓ skyrl-tx server is running"
echo ""

# Note: n_parallel_envs controls how many rollouts run concurrently
# The server's --sample-max-num-sequences controls how many are batched for inference
# If you hit OOM, either:
#   1. Reduce SAMPLE_MAX_SEQ when starting the server (default: 2)
#   2. Reduce n_parallel_envs below
#   3. Reduce context_limit

python -m src.train \
  backend=skyrl-tx \
  skyrl_tx_url=${SKYRL_TX_URL} \
  model_name=${MODEL_NAME} \
  tasks_dir=./datasets/terminal-bench-2 \
  learning_rate=2e-4 \
  batch_size=1 \
  group_size=8 \
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

