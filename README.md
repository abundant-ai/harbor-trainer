# Harbor Trainer

RL training for LLM agents on Harbor tasks using Prime-RL.

## Overview

This repository implements RL training for Harbor tasks using:
- **Prime-RL** for distributed LLM training (vLLM sampling + LoRA optimization)
- **Harbor Framework** for trial infrastructure (Terminus-2 agent, environments, verification)

The trainer uses Harbor's `Trial` to run episodes with the Terminus-2 agent, collecting token-level rollout data (token IDs, logprobs) for GRPO-style policy optimization via Prime-RL.

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/yourusername/harbortrainer
cd harbortrainer

# Install dependencies
uv pip install -e .
docker login  # optional but recommended for pulling/pushing images
```

## Quick Start

### 1. Set up credentials in `.env`

```bash
WANDB_API_KEY="your-api-key"
TOKENIZERS_PARALLELISM=false
```

### 2. Add tasks

Tasks follow the [Harbor task format](https://harborframework.com/docs/task-format):

```
datasets/harbor_tasks/
├── task_name/
│   ├── task.toml           # Task configuration
│   ├── instruction.md      # Task description for the agent
│   ├── environment/
│   │   └── Dockerfile      # Container setup
│   ├── solution/
│   │   └── solve.sh        # Reference solution
│   └── tests/
│       └── test.sh         # Verification script
```

### 3. Run training

Using the provided wrapper:

```bash
python -m src.train --config configs/harbor_8b.toml
```

Or using Prime-RL directly:

```bash
PYTHONPATH=. uv run --directory prime-rl rl @ configs/harbor_8b.toml
```

## Project Structure

```
.
├── src/
│   ├── harbor_env.py       # HarborEnvironment - verifiers Environment wrapper
│   ├── train.py            # CLI entry point
│   └── __init__.py
├── configs/
│   ├── harbor_8b.toml      # 8B model training config (local Docker)
│   └── harbor_8b_modal.toml # 8B model training config (Modal cloud)
├── prime-rl/               # Prime-RL submodule
└── datasets/
    └── harbor_tasks/       # Your Harbor tasks
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HarborEnvironment                        │
│              (verifiers.Environment wrapper)                │
├─────────────────────────────────────────────────────────────┤
│  For each rollout:                                          │
│    1. Run Harbor Trial with Terminus-2 agent                │
│    2. Collect rollout_details (token IDs, logprobs)         │
│    3. Get rewards from Harbor verifier                      │
│    4. Return trajectory in verifiers State format           │
└─────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────────┐
│  Harbor Trial   │          │   Prime-RL          │
│  - Terminus-2   │          │   - vLLM server     │
│  - Environments │          │   - LoRA training   │
│  - Verification │          │   - Weight updates  │
└─────────────────┘          └─────────────────────┘
```

## Configuration

Edit `configs/harbor_8b.toml` to customize training:

```toml
[[orchestrator.env]]
id = "harbor_env"
name = "harbor"
args = { 
    tasks_dir = "datasets/harbor_tasks",
    environment_type = "docker",  # or "modal", "e2b", etc.
    vllm_base_url = "http://localhost:8000/v1",
    model_name = "Qwen/Qwen3-8B-Instruct",
    n_parallel_envs = 8,
    max_turns = 20,
}
```

## References

- [Prime-RL](https://github.com/allenai/prime-rl)
- [Harbor Framework](https://harborframework.com/)
- [Terminus-2 Agent](https://harborframework.com/docs/agents/terminus-2)

## License

MIT
