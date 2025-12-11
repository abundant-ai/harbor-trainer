# SkyRL-TX Migration Plan

> **Status: ✅ IMPLEMENTED** - The codebase now supports both Tinker and skyrl-tx backends.

## Quick Start

```bash
# Terminal 1: Start skyrl-tx server
./scripts/start_skyrl_tx.sh Qwen/Qwen3-4B 1

# Terminal 2: Test the connection
python scripts/test_skyrl_tx.py

# Terminal 3: Run training
./run_skyrl_tx.sh
```

## Overview

This document details how to migrate the `harbortrainer` codebase to support **both** Tinker (cloud API) and skyrl-tx (self-hosted) backends. The goal is a unified codebase where the backend can be selected via configuration.

**Key Insight**: Both Tinker and skyrl-tx expose the same REST API, and your code uses the `tinker` Python SDK. This means **95%+ of your code works unchanged** - you just point at a different URL.

---

## Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Your Training Code                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐    │
│  │    train.py     │  │terminus2_trainer │  │    tinker_llm.py    │    │
│  │   (CLI/Config)  │  │  (Harbor Trial)  │  │   (LLM Backend)     │    │
│  └────────┬────────┘  └────────┬─────────┘  └──────────┬──────────┘    │
│           │                    │                       │               │
│           └────────────────────┴───────────────────────┘               │
│                                │                                        │
│                    ┌───────────┴───────────┐                           │
│                    │   tinker Python SDK   │                           │
│                    │  (ServiceClient, etc) │                           │
│                    └───────────┬───────────┘                           │
└────────────────────────────────┼────────────────────────────────────────┘
                                 │ HTTP REST API
                    ┌────────────┴────────────┐
                    │                         │
           ┌────────▼────────┐      ┌─────────▼─────────┐
           │  Tinker Cloud   │      │    skyrl-tx       │
           │  (Managed API)  │      │  (Self-Hosted)    │
           │                 │      │                   │
           │ • Limited models│      │ • Any model       │
           │ • 32K context   │      │ • No context limit│
           │ • LoRA only     │      │ • LoRA (full soon)│
           └─────────────────┘      └───────────────────┘
```

---

## Code Reuse Analysis

### Files That Work Unchanged (100% Reuse)

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `terminus2_trainer.py` | 761 | ✅ **No changes** | All GRPO logic, Harbor Trial orchestration, advantage computation, datum building |
| `tinker_llm.py` | 167 | ✅ **No changes** | Uses tinker SDK which works with both backends |
| `train.py` | 227 | ⚠️ **Minor changes** | Add backend selection config |

### What Actually Changes

Only **configuration** changes are needed:

```python
# Current: Implicit Tinker cloud
tinker_base_url: str | None = None  # Defaults to Tinker cloud

# New: Explicit backend selection
backend: Literal["tinker", "skyrl-tx"] = "tinker"
tinker_base_url: str | None = None  # For Tinker cloud
skyrl_tx_url: str | None = None     # For self-hosted skyrl-tx
```

---

## Detailed Migration Steps

### Phase 1: Configuration Updates (30 minutes)

#### 1.1 Update `TrainerConfig` in `terminus2_trainer.py`

Add backend selection:

```python
@dataclass
class TrainerConfig:
    """Configuration for Terminus2RLTrainer."""
    
    # ... existing fields ...
    
    # Backend selection (NEW)
    backend: str = "tinker"  # "tinker" or "skyrl-tx"
    skyrl_tx_url: str | None = None  # e.g., "http://localhost:8000"
    
    # Tinker configuration (existing)
    tinker_base_url: str | None = None
    lora_rank: int = 32
```

#### 1.2 Update `CLIConfig` in `train.py`

```python
@chz.chz
class CLIConfig:
    # ... existing fields ...
    
    # Backend selection (NEW)
    backend: Literal["tinker", "skyrl-tx"] = chz.field(
        default="tinker",
        doc="Training backend: 'tinker' (cloud) or 'skyrl-tx' (self-hosted)",
    )
    skyrl_tx_url: str | None = chz.field(
        default=None,
        doc="skyrl-tx server URL (e.g., http://localhost:8000)",
    )
```

#### 1.3 Update `setup()` in `terminus2_trainer.py`

```python
async def setup(self) -> None:
    """Initialize Tinker clients."""
    # Determine base URL based on backend
    if self.config.backend == "skyrl-tx":
        if not self.config.skyrl_tx_url:
            raise ValueError("skyrl_tx_url required when backend='skyrl-tx'")
        base_url = self.config.skyrl_tx_url
    else:
        base_url = self.config.tinker_base_url  # None = Tinker cloud default
    
    self._service_client = tinker.ServiceClient(base_url=base_url)
    # ... rest unchanged ...
```

### Phase 2: Create Run Scripts (15 minutes)

#### 2.1 Create `run_skyrl_tx.sh`

```bash
#!/bin/bash

# Start skyrl-tx server (run this first in a separate terminal)
# cd ../SkyRL/skyrl-tx
# uv run --extra tinker --extra gpu python -m tx.tinker.api \
#   --base-model Qwen/Qwen3-4B \
#   --tensor-parallel-size 1 \
#   --max-lora-rank 32

ulimit -n 65536

python -m src.train \
  backend=skyrl-tx \
  skyrl_tx_url=http://localhost:8000 \
  model_name=Qwen/Qwen3-4B \
  tasks_dir=./datasets/terminal-bench-2 \
  learning_rate=2e-4 \
  batch_size=2 \
  group_size=16 \
  n_parallel_envs=16 \
  max_tokens=2048 \
  temperature=0.7 \
  context_limit=128000 \
  proactive_summarization_threshold=8000 \
  enable_summarize=true \
  n_epochs=2 \
  num_substeps=4 \
  remove_constant_reward_groups=true \
  normalize_advantages_by_std=true \
  loss_fn=ppo \
  environment_type=docker \
  wandb_project=harbor-training \
  wandb_name=qwen3-skyrl-tx
```

#### 2.2 Keep `run.sh` for Tinker (unchanged)

Your existing `run.sh` continues to work for Tinker cloud.

### Phase 3: Environment Variables (Optional)

Create `.env.skyrl-tx`:

```bash
# skyrl-tx backend configuration
TINKER_API_KEY=""  # Not needed for skyrl-tx
WANDB_API_KEY="your-api-key"
TOKENIZERS_PARALLELISM=false
SKYRL_TX_URL="http://localhost:8000"
```

---

## API Compatibility Matrix

| Tinker API Endpoint | Tinker Cloud | skyrl-tx | Notes |
|---------------------|--------------|----------|-------|
| `POST /api/v1/create_model` | ✅ | ✅ | Identical |
| `POST /api/v1/forward_backward` | ✅ | ✅ | Identical |
| `POST /api/v1/optim_step` | ✅ | ✅ | Identical |
| `POST /api/v1/save_weights` | ✅ | ✅ | Identical |
| `POST /api/v1/save_weights_for_sampler` | ✅ | ✅ | Identical |
| `POST /api/v1/asample` | ✅ | ✅ | Identical |
| `POST /api/v1/retrieve_future` | ✅ | ✅ | Identical |
| `GET /api/v1/healthz` | ✅ | ✅ | Identical |

**Loss Functions Supported:**
| Loss Function | Tinker | skyrl-tx |
|---------------|--------|----------|
| `cross_entropy` | ✅ | ✅ |
| `importance_sampling` | ✅ | ✅ |
| `ppo` | ✅ | ✅ |

---

## Model Support Comparison

| Model | Tinker Cloud | skyrl-tx | Notes |
|-------|--------------|----------|-------|
| DeepSeek-V3.1 | ✅ | ❌ | skyrl-tx needs DeepSeek model impl |
| Qwen/Qwen3-4B | ✅ | ✅ | Works today |
| Qwen/Qwen3-8B | ✅ | ✅ | Works today |
| Qwen/Qwen3-30B-A3B | ? | ✅ | MoE model |
| Qwen/Qwen3-235B-A22B | ? | ✅ | Large MoE |

**Recommendation**: Start with Qwen3-4B on skyrl-tx to validate, then add DeepSeek support.

---

## Configuration Differences

| Parameter | Tinker Cloud | skyrl-tx | Notes |
|-----------|--------------|----------|-------|
| `context_limit` | **32,000 max** | **Unlimited** | Major advantage |
| `lora_rank` | Up to 32 | Up to 32 (configurable) | Same |
| `max_tokens` | Limited | Model-dependent | More flexible |
| `temperature` | Standard | Standard | Same |

---

## Code Changes (Detailed)

### Change 1: `train.py` - Add Backend Config

```python
# Location: src/train.py, around line 34

# ADD these fields to CLIConfig:
backend: Literal["tinker", "skyrl-tx"] = chz.field(
    default="tinker",
    doc="Training backend: 'tinker' (cloud API) or 'skyrl-tx' (self-hosted)",
)
skyrl_tx_url: str | None = chz.field(
    default=None,
    doc="skyrl-tx server URL when backend='skyrl-tx'",
)
```

```python
# Location: src/train.py, in run_training() around line 157

# UPDATE TrainerConfig initialization:
trainer_config = TrainerConfig(
    # ... existing fields ...
    backend=config.backend,                    # ADD
    skyrl_tx_url=config.skyrl_tx_url,         # ADD
    tinker_base_url=config.tinker_base_url,
    # ... rest unchanged ...
)
```

### Change 2: `terminus2_trainer.py` - Backend Selection

```python
# Location: src/terminus2_trainer.py, around line 46

@dataclass
class TrainerConfig:
    # ... existing fields ...
    
    # ADD after tinker_base_url:
    backend: str = "tinker"  # "tinker" or "skyrl-tx"
    skyrl_tx_url: str | None = None
```

```python
# Location: src/terminus2_trainer.py, in setup() around line 318

async def setup(self) -> None:
    """Initialize Tinker clients."""
    logging.getLogger("harbor").setLevel(logging.WARNING)
    logging.getLogger("harbor.utils.logger").setLevel(logging.WARNING)
    
    # REPLACE the ServiceClient initialization:
    # OLD:
    # self._service_client = tinker.ServiceClient(base_url=self.config.tinker_base_url)
    
    # NEW:
    if self.config.backend == "skyrl-tx":
        if not self.config.skyrl_tx_url:
            raise ValueError("skyrl_tx_url required when backend='skyrl-tx'")
        base_url = self.config.skyrl_tx_url
        logger.info(f"Using skyrl-tx backend at {base_url}")
    else:
        base_url = self.config.tinker_base_url
        logger.info(f"Using Tinker backend" + (f" at {base_url}" if base_url else " (default)"))
    
    self._service_client = tinker.ServiceClient(base_url=base_url)
    
    # ... rest unchanged ...
```

### Change 3: Add Backend Validation (Optional)

```python
# Location: src/terminus2_trainer.py, add new method around line 350

def _validate_backend_compatibility(self) -> None:
    """Validate that the model is supported by the selected backend."""
    if self.config.backend == "skyrl-tx":
        # skyrl-tx currently only supports Qwen3 models
        supported_prefixes = ["Qwen/Qwen3", "qwen/qwen3"]
        if not any(self.config.model_name.startswith(p) for p in supported_prefixes):
            logger.warning(
                f"Model {self.config.model_name} may not be supported by skyrl-tx. "
                f"Currently supported: Qwen3 family. "
                f"Consider using backend='tinker' or contributing model support to skyrl-tx."
            )
        
        # skyrl-tx has no context limit
        if self.config.context_limit > 128000:
            logger.info(
                f"skyrl-tx supports large context windows. "
                f"Using context_limit={self.config.context_limit}"
            )
```

---

## Running Both Backends

### Scenario 1: Quick Experiments (Tinker Cloud)

```bash
# Use Tinker for:
# - Supported models (DeepSeek, etc.)
# - No GPU setup needed
# - Pay-per-use

./run.sh  # Uses Tinker cloud (existing behavior)
```

### Scenario 2: Large Context / Custom Models (skyrl-tx)

```bash
# Terminal 1: Start skyrl-tx server
cd ../SkyRL/skyrl-tx
uv run --extra tinker --extra gpu python -m tx.tinker.api \
  --base-model Qwen/Qwen3-4B \
  --tensor-parallel-size 1 \
  --max-lora-rank 32 \
  --gradient-checkpointing  # For large context

# Terminal 2: Run training
cd ../harbortrainer
./run_skyrl_tx.sh
```

### Scenario 3: Multi-GPU Training (skyrl-tx)

```bash
# For larger models or faster training
cd ../SkyRL/skyrl-tx
uv run --extra tinker --extra gpu python -m tx.tinker.api \
  --base-model Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 8 \
  --max-lora-rank 32 \
  --train-micro-batch-size 2 \
  --gradient-checkpointing
```

---

## Testing the Migration

### Test 1: Verify API Compatibility

```python
# test_skyrl_tx_api.py
import asyncio
import tinker

async def test_connection():
    client = tinker.ServiceClient(base_url="http://localhost:8000")
    
    # Test model creation
    training_client = await client.create_lora_training_client_async(
        base_model="Qwen/Qwen3-4B",
        rank=32,
    )
    print(f"✅ Created training client")
    
    # Test tokenizer
    tokenizer = training_client.get_tokenizer()
    tokens = tokenizer.encode("Hello, world!")
    print(f"✅ Tokenizer works: {tokens}")
    
    # Test sampling
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        name="test"
    )
    print(f"✅ Sampling client created")

if __name__ == "__main__":
    asyncio.run(test_connection())
```

### Test 2: Run Single Task

```bash
# Test with a single simple task
python -m src.train \
  backend=skyrl-tx \
  skyrl_tx_url=http://localhost:8000 \
  model_name=Qwen/Qwen3-4B \
  tasks_dir=./datasets/terminal-bench-2 \
  batch_size=1 \
  group_size=2 \
  n_parallel_envs=2 \
  n_epochs=1 \
  environment_type=docker
```

---

## Troubleshooting

### Issue: "Model not supported"

**skyrl-tx Error:**
```
ValueError: Requested base_model 'deepseek-ai/DeepSeek-V3.1' does not match engine's base_model 'Qwen/Qwen3-4B'
```

**Solution:** skyrl-tx must be started with the same model you're training:
```bash
# Match the model in your training config
uv run python -m tx.tinker.api --base-model deepseek-ai/DeepSeek-V3.1
```

If the model isn't supported yet, use Tinker or contribute model support.

### Issue: Context Length

**Tinker Error:**
```
ContextLengthExceededError: Context length 45000 exceeds limit 32000
```

**Solution:** Switch to skyrl-tx which has no hardcoded limit:
```bash
./run_skyrl_tx.sh  # With context_limit=128000 or higher
```

### Issue: Connection Refused

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution:** Ensure skyrl-tx server is running:
```bash
# Check if server is up
curl http://localhost:8000/api/v1/healthz
# Should return: {"status":"ok"}
```

---

## Migration Checklist

- [ ] **Phase 1: Config Updates**
  - [ ] Add `backend` field to `TrainerConfig`
  - [ ] Add `skyrl_tx_url` field to `TrainerConfig`
  - [ ] Add `backend` field to `CLIConfig`
  - [ ] Add `skyrl_tx_url` field to `CLIConfig`
  - [ ] Update `setup()` to select backend URL

- [ ] **Phase 2: Scripts**
  - [ ] Create `run_skyrl_tx.sh`
  - [ ] Create `.env.skyrl-tx` (optional)
  - [ ] Test with Qwen3-4B

- [ ] **Phase 3: Validation**
  - [ ] Run `test_skyrl_tx_api.py`
  - [ ] Complete single-task training run
  - [ ] Compare metrics between Tinker and skyrl-tx

- [ ] **Phase 4: Production**
  - [ ] Multi-GPU setup
  - [ ] Long-context training test
  - [ ] Document any skyrl-tx specific issues

---

## Future: Adding DeepSeek Support to skyrl-tx

Once migration is validated with Qwen3, you can contribute DeepSeek support:

```
SkyRL/skyrl-tx/tx/models/
├── qwen3.py          # Existing
├── deepseek.py       # You would add this
└── configs.py        # Add DeepSeekConfig
```

The pattern is identical to `qwen3.py`:
1. Create config class
2. Implement attention, MLP, decoder layers
3. Use existing `LoRALinear`, `LoRAExpert` for LoRA support
4. Register in model loader

---

## Summary

| Aspect | Effort | Impact |
|--------|--------|--------|
| Config changes | ~30 min | Enables dual-backend |
| New run scripts | ~15 min | Easy switching |
| Testing | ~1 hour | Validates migration |
| **Total** | **~2 hours** | **Full flexibility** |

**Result:** A single codebase that works with both Tinker (for supported models, no setup) and skyrl-tx (for unlimited context, custom models, self-hosted).

