#!/usr/bin/env bash
# Harbor Training with Prime-RL
#
# This script launches RL training on Harbor tasks using:
# - Terminus-2 agent (multi-turn terminal agent)
# - Harbor's task format and verification system
# - Prime-RL's async training infrastructure
# - Verifiers library for environment abstraction
#
# Usage:
#   ./run_training.sh                              # Use default config (harbor_8b.toml)
#   ./run_training.sh configs/harbor_8b_modal.toml # Use custom config
#
# Requirements:
#   - Docker running (for docker environment) or Modal/E2B configured (for cloud)
#   - 2+ GPUs (1 for inference vLLM server, 1+ for training)
#   - Harbor tasks in the configured tasks_dir
#   - PYTHONPATH must include this repo root for src.harbor_env to be importable

set -e

# Default config
CONFIG="${1:-configs/harbor_8b.toml}"

# Navigate to harbortrainer root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -a  # automatically export all variables
    source .env
    set +a
else
    echo "Warning: .env file not found. Create one with API keys if needed."
fi

# Set dummy OPENAI_API_KEY for LiteLLM if not already set
# LiteLLM requires this even for local vLLM servers (vLLM doesn't validate it)
if [ -z "$OPENAI_API_KEY" ]; then
    export OPENAI_API_KEY="dummy-key-for-local-vllm"
    echo "Set dummy OPENAI_API_KEY for local vLLM server"
fi

echo "=========================================="
echo "Harbor Training with Prime-RL"
echo "=========================================="
echo "Config: $CONFIG"
echo "Root:   $SCRIPT_DIR"
echo ""

# Check config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    echo ""
    echo "Available configs:"
    ls -la configs/*.toml 2>/dev/null || echo "  No configs found in configs/"
    exit 1
fi

# Check tasks directory
TASKS_DIR=$(grep -E '^\s*tasks_dir' "$CONFIG" | head -1 | sed 's/.*=\s*"\([^"]*\)".*/\1/' || echo "harbor_tasks/extracted_tasks")
if [ ! -d "$TASKS_DIR" ]; then
    echo "Warning: Tasks directory not found: $TASKS_DIR"
    echo "Training will start but may fail if no tasks are available."
    echo ""
fi

# Display task count if directory exists
if [ -d "$TASKS_DIR" ]; then
    TASK_COUNT=$(find "$TASKS_DIR" -name "task.toml" 2>/dev/null | wc -l || echo "0")
    echo "Found $TASK_COUNT Harbor tasks in $TASKS_DIR"
    echo ""
fi

# Check for Docker if using docker environment
ENV_TYPE=$(grep -E '^\s*environment_type' "$CONFIG" | head -1 | sed 's/.*=\s*"\([^"]*\)".*/\1/' || echo "docker")
if [ "$ENV_TYPE" = "docker" ]; then
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed but environment_type is 'docker'"
        exit 1
    fi
    if ! docker info &> /dev/null; then
        echo "Error: Docker daemon is not running"
        exit 1
    fi
    echo "Docker environment: OK"
fi

# Ensure PYTHONPATH includes this repo root
# This is required for prime-rl to import src.harbor_env
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

echo "Starting training..."
echo "Command: uv run rl @ $CONFIG"
echo ""

# Run prime-rl orchestrator
# The 'rl' command launches all components (inference server, orchestrator, trainer)
# in a tmux session for easy monitoring
exec uv run rl @ "$CONFIG"

