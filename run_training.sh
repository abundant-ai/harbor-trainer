#!/usr/bin/env bash
# Harbor Training with Prime-RL
#
# Usage:
#   ./run_training.sh                           # Use default config (harbor_8b.toml)
#   ./run_training.sh configs/harbor_8b_modal.toml  # Use custom config
#
# Requirements:
#   - Docker running (for docker environment) or Modal configured (for modal)
#   - 2 GPUs minimum (1 for inference, 1 for training)
#   - Harbor tasks in datasets/terminal-bench-2/

set -e

# Default config
CONFIG="${1:-configs/harbor_8b.toml}"

# Navigate to harbortrainer root
cd "$(dirname "$0")"

echo "=========================================="
echo "Harbor Training with Prime-RL"
echo "=========================================="
echo "Config: $CONFIG"
echo ""

# Check config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    echo ""
    echo "Available configs:"
    ls -la configs/*.toml 2>/dev/null || echo "  No configs found"
    exit 1
fi

# Check tasks directory
TASKS_DIR=$(grep -E '^\s*tasks_dir' "$CONFIG" | head -1 | sed 's/.*=\s*"\([^"]*\)".*/\1/' || echo "datasets/terminal-bench-2")
if [ ! -d "$TASKS_DIR" ]; then
    echo "Warning: Tasks directory not found: $TASKS_DIR"
    echo "Training will start but may fail if no tasks are available."
fi

# Run prime-rl with PYTHONPATH set to include src/
echo "Starting training..."
echo "Command: PYTHONPATH=. uv run --directory prime-rl rl @ ../$CONFIG"
echo ""

PYTHONPATH="." uv run --directory prime-rl rl @ "../$CONFIG"

