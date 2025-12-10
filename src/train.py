from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Literal

import chz
from dotenv import load_dotenv

from src.terminus2_trainer import Terminus2RLTrainer, TrainerConfig

load_dotenv()


@chz.chz
class CLIConfig:
    """CLI configuration for Terminus2 RL training."""

    # Required
    model_name: str = chz.field(doc="Model name (e.g., 'meta-llama/Llama-3.1-8B')")
    tasks_dir: str = chz.field(doc="Directory containing Harbor tasks")

    # Directories
    logs_dir: str = chz.field(
        default="/tmp/terminus2-training",
        doc="Logs directory",
        munger=lambda _, s: os.path.expanduser(s),
    )

    # Tinker configuration
    tinker_base_url: str | None = chz.field(default=None, doc="Tinker API base URL")
    lora_rank: int = chz.field(default=32, doc="LoRA rank")

    # Training hyperparameters
    learning_rate: float = chz.field(default=5e-5, doc="Learning rate")
    batch_size: int = chz.field(default=8, doc="Tasks per batch")
    group_size: int = chz.field(default=4, doc="Rollouts per task (GRPO group size)")
    n_epochs: int = chz.field(default=1, doc="Number of epochs")

    # RL hyperparameters
    loss_fn: Literal["importance_sampling", "ppo"] = chz.field(
        default="importance_sampling",
        doc="Loss function: 'importance_sampling' (REINFORCE) or 'ppo'",
    )
    num_substeps: int = chz.field(
        default=1,
        doc="Optimizer substeps per batch (PPO-style multi-epoch)",
    )
    remove_constant_reward_groups: bool = chz.field(
        default=False,
        doc="Remove groups where all rollouts have the same reward",
    )

    # Agent configuration
    max_turns: int | None = chz.field(default=None, doc="Max agent turns (None = unlimited)")
    temperature: float = chz.field(default=0.7, doc="Sampling temperature")
    max_tokens: int = chz.field(default=4096, doc="Max tokens per generation")
    context_limit: int = chz.field(default=128000, doc="Model context limit")

    enable_summarize: bool = chz.field(
        default=True,
        doc="Enable context summarization when context is full (matches Terminus 2 eval)",
    )
    proactive_summarization_threshold: int = chz.field(
        default=8000,
        doc="Trigger proactive summarization when free tokens fall below this threshold",
    )

    # Environment configuration
    environment_type: Literal["docker", "daytona", "modal", "e2b", "runloop"] = chz.field(
        default="docker",
        doc="Environment backend: docker (local), daytona/modal/e2b/runloop (cloud)",
    )
    environment_kwargs: list[str] | None = chz.field(
        default=None,
        doc=(
            "Environment kwargs in key=value form (can be repeated). "
            "Common: override_cpus, override_memory_mb, override_storage_mb. "
            "Modal-only: add_python_version=3.11 to auto-install Python. "
            "Examples: timeout=300, override_cpus=4"
        ),
    )
    n_parallel_envs: int = chz.field(
        default=1,
        doc="Parallel environments. Keep low (1-2) for docker, higher for cloud",
    )
    trial_timeout_sec: float | None = chz.field(
        default=None,
        doc="Optional timeout (seconds) per trial; None uses task defaults",
    )

    # Logging and checkpoints
    wandb_project: str | None = chz.field(default=None, doc="Weights & Biases project")
    wandb_name: str | None = chz.field(default=None, doc="Weights & Biases run name")
    save_every: int = chz.field(default=20, doc="Save checkpoint every N batches")


def _parse_kwargs(kwargs_list: list[str] | None) -> dict[str, object]:
    """Parse key=value strings; values are JSON if possible, else strings."""
    if not kwargs_list:
        return {}
    parsed: dict[str, object] = {}
    for item in kwargs_list:
        if "=" not in item:
            raise ValueError(f"Invalid environment kwarg: {item}. Expected key=value")
        key, value = item.split("=", 1)
        try:
            parsed[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key.strip()] = value.strip()
    return parsed


async def run_training(config: CLIConfig) -> None:
    """Run Terminus2 RL training."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Starting Terminus2 RL Training")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Tasks: {config.tasks_dir}")
    logger.info(f"Environment: {config.environment_type} (n_parallel={config.n_parallel_envs})")
    logger.info(f"Batch size: {config.batch_size}, Group size: {config.group_size}")
    logger.info(f"Loss function: {config.loss_fn}, Substeps: {config.num_substeps}")
    logger.info(f"Context summarization: {config.enable_summarize} (threshold: {config.proactive_summarization_threshold})")
    logger.info("=" * 60)

    trainer_config = TrainerConfig(
        model_name=config.model_name,
        tasks_dir=Path(config.tasks_dir),
        logs_dir=Path(config.logs_dir),
        tinker_base_url=config.tinker_base_url,
        lora_rank=config.lora_rank,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        group_size=config.group_size,
        n_epochs=config.n_epochs,
        loss_fn=config.loss_fn,
        num_substeps=config.num_substeps,
        remove_constant_reward_groups=config.remove_constant_reward_groups,
        max_turns=config.max_turns,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        context_limit=config.context_limit,
        enable_summarize=config.enable_summarize,
        proactive_summarization_threshold=config.proactive_summarization_threshold,
        environment_type=config.environment_type,
        environment_kwargs=_parse_kwargs(config.environment_kwargs),
        n_parallel_envs=config.n_parallel_envs,
        trial_timeout_sec=config.trial_timeout_sec,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        save_every=config.save_every,
    )

    trainer = Terminus2RLTrainer(trainer_config)
    await trainer.train()

    logger.info("Training complete!")


def main() -> None:
    """Entry point for training."""
    # Set root logger to WARNING to suppress verbose library logs
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Enable INFO logs only for our own modules
    # logging.getLogger("__main__").setLevel(logging.INFO)
    # logging.getLogger("src.terminus2_trainer").setLevel(logging.INFO)
    # logging.getLogger("src.tinker_llm").setLevel(logging.INFO)
    
    # # Keep tinker-cookbook at INFO for training metrics
    # logging.getLogger("tinker_cookbook.utils.ml_log").setLevel(logging.INFO)
    
    # # Suppress verbose Harbor logs (Docker/environment operations)
    # logging.getLogger("harbor").setLevel(logging.WARNING)

    config = chz.entrypoint(CLIConfig)
    asyncio.run(run_training(config))


if __name__ == "__main__":
    main()