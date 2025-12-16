from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Dataset as HFDataset
from harbor.llms.lite_llm import LiteLLM
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


class HarborEnvironment:
    """
    Verifiers Environment that wraps Harbor Task execution.
    
    This allows using Harbor's:
    - Task format (task.toml, instruction.md, etc.)
    - Sandboxes (Docker, Modal, E2B, etc.)
    - Terminus-2 agent harness with LiteLLM (vLLM backend)
    - Verification system
    
    While integrating with prime-rl's training loop.
    """
    
    def __init__(
        self,
        tasks_dir: str = "datasets/extracted_tasks",
        environment_type: str = "docker",
        environment_kwargs: dict[str, Any] | None = None,
        max_turns: int | None = 20,
        enable_summarize: bool = True,
        proactive_summarization_threshold: int = 8000,
        trial_timeout_sec: float | None = None,
        n_parallel_envs: int = 8,
        # vLLM config - these come from prime-rl's inference server
        vllm_base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen3-8B-Instruct",
        temperature: float = 0.7,
        trials_dir: str = "/tmp/harbor-prime-rl",
    ):
        """
        Initialize HarborEnvironment.
        
        Args:
            tasks_dir: Path to directory containing Harbor tasks
            environment_type: Type of sandbox ("docker", "modal", "e2b", etc.)
            environment_kwargs: Additional kwargs for environment
            max_turns: Maximum agent turns per episode
            enable_summarize: Enable context summarization
            proactive_summarization_threshold: Token threshold for summarization
            trial_timeout_sec: Optional timeout per trial
            n_parallel_envs: Max concurrent trials
            vllm_base_url: URL of vLLM server (started by prime-rl)
            model_name: Model name in vLLM
            temperature: Sampling temperature
            trials_dir: Directory for trial logs
        """
        self.tasks_dir = Path(tasks_dir)
        self.environment_type = environment_type
        self.environment_kwargs = environment_kwargs or {}
        self.max_turns = max_turns
        self.enable_summarize = enable_summarize
        self.proactive_summarization_threshold = proactive_summarization_threshold
        self.trial_timeout_sec = trial_timeout_sec
        self.n_parallel_envs = n_parallel_envs
        self.vllm_base_url = vllm_base_url
        self.model_name = model_name
        self.temperature = temperature
        self.trials_dir = Path(trials_dir)
        
        # State
        self.tasks: list = []
        self._semaphore: asyncio.Semaphore | None = None
        
        # Verifiers Environment interface requirements
        self.env_id = "harbor"
        self.env_args: dict[str, Any] = {}
        self.max_seq_len: int | None = None
        self.dataset: Dataset | None = None
        self.eval_dataset: Dataset | None = None
        
        # LoRA tracking for weight broadcast
        self._current_lora: str | None = None
        
        # Initialize on construction
        self._setup_sync()
    
    def _setup_sync(self):
        """Synchronous setup called during construction."""
        # Initialize semaphore
        self._semaphore = asyncio.Semaphore(self.n_parallel_envs)
        
        # Load Harbor tasks
        self.tasks = []
        if self.tasks_dir.exists():
            for task_dir in sorted(self.tasks_dir.iterdir()):
                if task_dir.is_dir() and (task_dir / "task.toml").exists():
                    try:
                        self.tasks.append(Task(task_dir=task_dir))
                    except Exception as e:
                        logger.warning(f"Failed to load task {task_dir}: {e}")
        
        logger.info(f"Loaded {len(self.tasks)} Harbor tasks from {self.tasks_dir}")
        
        # Create dataset in verifiers format
        self._create_dataset()
    
    def _create_dataset(self):
        """Convert Harbor tasks to verifiers dataset format."""
        data = []
        for idx, task in enumerate(self.tasks):
            # Load task instruction as prompt
            instruction_file = task.task_dir / "instruction.md"
            if instruction_file.exists():
                prompt = instruction_file.read_text()
            else:
                prompt = task.config.get("instruction", "Complete the task.")
            
            data.append({
                "prompt": [{"role": "user", "content": prompt}],
                "example_id": idx,
                "task": "harbor",
                "answer": "",  # Harbor uses verifier, not ground truth
                "info": {
                    "task_dir": str(task.task_dir),
                    "task_name": task.name,
                },
            })
        
        if data:
            self.dataset = HFDataset.from_list(data)
            self.eval_dataset = self.dataset
        else:
            # Empty dataset
            self.dataset = HFDataset.from_list([{
                "prompt": [{"role": "user", "content": "placeholder"}],
                "example_id": 0,
                "task": "harbor",
                "answer": "",
                "info": {},
            }])
            self.eval_dataset = self.dataset
    
    def _create_llm(self):
        """Create a LiteLLM instance pointing to vLLM server."""
        # For LoRA, vLLM expects model name in format "base:adapter"
        model_name = self.model_name
        if self._current_lora:
            model_name = f"{self.model_name}:{self._current_lora}"
        
        return LiteLLM(
            model_name=model_name,
            api_base=self.vllm_base_url,
            collect_rollout_details=True,  # Enable token IDs + logprobs
            temperature=self.temperature,
        )
    
    # =========================================================================
    # Verifiers Environment Interface
    # =========================================================================
    
    def get_dataset(self, n: int = -1, seed: int | None = None):
        """Get training dataset (verifiers interface)."""
        if self.dataset is None:
            return None
        if n > 0 and n < len(self.dataset):
            return self.dataset.select(range(n))
        return self.dataset
    
    def get_eval_dataset(self, n: int = -1, seed: int | None = None):
        """Get evaluation dataset (verifiers interface)."""
        if self.eval_dataset is None:
            return None
        if n > 0 and n < len(self.eval_dataset):
            return self.eval_dataset.select(range(n))
        return self.eval_dataset
    
    def set_max_seq_len(self, max_seq_len: int | None):
        """Set max sequence length (verifiers interface)."""
        self.max_seq_len = max_seq_len
    
    def update_lora(self, lora_name: str | None):
        """Update LoRA adapter name for weight broadcast.
        
        Called by prime-rl after weight broadcast to point to new adapter.
        """
        self._current_lora = lora_name
        logger.info(f"Updated LoRA adapter: {lora_name}")
    
    async def _run_harbor_trial(self, task, example_id: int) -> dict:
        """
        Execute a Harbor trial and return trajectory with tokens.
        
        Returns dict compatible with verifiers State format.
        """
        # Create fresh LLM instance for this trial (with current LoRA)
        llm = self._create_llm()
        
        # Create trial config
        env_type = EnvironmentType(self.environment_type)
        trial_config = TrialConfig(
            task=TaskConfig(path=task.task_dir),
            trials_dir=self.trials_dir,
            agent=AgentConfig(
                name="terminus-2",
                model_name=self.model_name,
                kwargs={
                    "llm": llm,
                    "collect_rollout_details": True,
                    "enable_summarize": self.enable_summarize,
                    "proactive_summarization_threshold": self.proactive_summarization_threshold,
                    "max_turns": self.max_turns,
                },
            ),
            environment=EnvironmentConfig(
                type=env_type,
                delete=True,
                kwargs=self.environment_kwargs,
            ),
        )
        
        # Run trial
        trial = Trial(trial_config)
        if self.trial_timeout_sec:
            result = await asyncio.wait_for(
                trial.run(),
                timeout=self.trial_timeout_sec,
            )
        else:
            result = await trial.run()
        
        # Convert to verifiers trajectory format
        trajectory = []
        if result.agent_result and result.agent_result.rollout_details:
            # Only use [0] - main agent rollout, excluding subagent (e.g., summarization)
            rd = result.agent_result.rollout_details[0]
            
            prompt_tokens = rd.get("prompt_token_ids", [])
            completion_tokens = rd.get("completion_token_ids", [])
            logprobs = rd.get("logprobs", [])
            
            n_turns = len(completion_tokens)
            for turn_idx in range(n_turns):
                if turn_idx >= len(prompt_tokens) or turn_idx >= len(logprobs):
                    continue
                
                turn_prompt = prompt_tokens[turn_idx]
                turn_completion = completion_tokens[turn_idx]
                turn_logprobs = logprobs[turn_idx]
                
                # Skip empty turns
                if not turn_completion:
                    continue
                
                trajectory.append({
                    "tokens": {
                        "prompt_ids": turn_prompt,
                        "prompt_mask": [0] * len(turn_prompt),
                        "completion_ids": turn_completion,
                        "completion_mask": [1] * len(turn_completion),
                        "completion_logprobs": turn_logprobs if turn_logprobs else [0.0] * len(turn_completion),
                        "overlong_prompt": False,
                        "is_truncated": False,
                    },
                    "prompt": [],  # Messages - not needed for training
                    "completion": [],
                    "response": None,
                    "reward": None,
                    "advantage": None,
                    "extras": {},
                })
        
        # Extract reward from verifier
        reward = 0.0
        if result.verifier_result and result.verifier_result.rewards:
            reward_value = result.verifier_result.rewards.get("reward", 0.0)
            if isinstance(reward_value, (int, float)):
                reward = float(reward_value)
            else:
                reward = 1.0 if result.verifier_result.rewards.get("pass", False) else 0.0
        
        return {
            "example_id": example_id,
            "task": "harbor",
            "trajectory": trajectory,
            "reward": reward,
            "is_completed": True,
        }
    
    async def rollout(
        self,
        input: dict,  # RolloutInput
        client,  # AsyncOpenAI (unused - we use Harbor's LLM)
        model: str,
        sampling_args: dict | None = None,
    ) -> dict:  # State
        """
        Run a single rollout (verifiers interface).
        
        This executes a Harbor Trial and returns a State dict.
        """
        example_id = input.get("example_id", 0)
        
        if example_id >= len(self.tasks):
            logger.error(f"Invalid example_id {example_id}, only {len(self.tasks)} tasks")
            return self._error_state(input, "Invalid example_id")
        
        task = self.tasks[example_id]
        
        try:
            state = await self._run_harbor_trial(task, example_id)
            state["input"] = input
            state["error"] = None
            return state
        except asyncio.TimeoutError:
            logger.error(f"Trial timed out for {task.name}")
            return self._error_state(input, "Trial timed out")
        except Exception as e:
            logger.error(f"Trial failed for {task.name}: {e}")
            return self._error_state(input, str(e))
    
    def _error_state(self, input: dict, error: str) -> dict:
        """Create an error state."""
        return {
            "input": input,
            "example_id": input.get("example_id", 0),
            "task": "harbor",
            "trajectory": [],
            "reward": 0.0,
            "is_completed": True,
            "error": error,
        }
    
    async def run_rollout(
        self,
        sem,  # AsyncContextManager (semaphore)
        input: dict,
        client,
        model: str,
        sampling_args: dict | None = None,
    ) -> dict:
        """Run rollout with semaphore (verifiers interface)."""
        async with sem:
            return await self.rollout(input, client, model, sampling_args)
    
    async def run_group(
        self,
        group_inputs: list[dict],
        client,
        model: str,
        gen_sampling_args: dict,
        gen_sem,
        score_sem,
        **kwargs,
    ) -> list[dict]:
        """Run rollouts for a group and score them (verifiers interface)."""
        rollout_tasks = [
            self.run_rollout(gen_sem, input, client, model, gen_sampling_args)
            for input in group_inputs
        ]
        states = await asyncio.gather(*rollout_tasks)
        
        # Scoring is done by Harbor verifier during trial
        # Compute group-relative advantage (GRPO-style)
        rewards = [s["reward"] for s in states]
        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            for state in states:
                state["advantage"] = state["reward"] - avg_reward
                # Propagate advantage to trajectory steps
                for t in state.get("trajectory", []):
                    if t.get("advantage") is None:
                        t["advantage"] = state["advantage"]
        
        return list(states)


# =============================================================================
# Entry Point for Prime-RL
# =============================================================================

def load_environment(
    tasks_dir: str = "datasets/extracted_tasks",
    environment_type: str = "docker",
    environment_kwargs: dict | None = None,
    max_turns: int | None = 20,
    enable_summarize: bool = True,
    proactive_summarization_threshold: int = 8000,
    trial_timeout_sec: float | None = None,
    n_parallel_envs: int = 8,
    vllm_base_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen3-8B-Instruct",
    temperature: float = 0.7,
    trials_dir: str = "/tmp/harbor-prime-rl",
    **kwargs,  # Accept additional args for compatibility
) -> HarborEnvironment:
    """
    Load Harbor environment (verifiers entry point).
    
    This function is called by prime-rl when loading the environment.
    Prime-RL dynamically imports this module and calls load_environment().
    
    Usage in prime-rl config (rl.toml):
    
        [[orchestrator.env]]
        id = "harbor_env"
        name = "harbor"
        args = { 
            tasks_dir = "datasets/extracted_tasks",
            environment_type = "docker",
            vllm_base_url = "http://localhost:8000/v1",
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
        }
    
    Args:
        tasks_dir: Path to Harbor tasks directory
        environment_type: "docker", "modal", "e2b", etc.
        environment_kwargs: Additional environment kwargs
        max_turns: Max agent turns per episode
        enable_summarize: Enable context summarization
        proactive_summarization_threshold: Token threshold for summarization
        trial_timeout_sec: Optional timeout per trial
        n_parallel_envs: Max concurrent trials
        vllm_base_url: URL of vLLM server (started by prime-rl)
        model_name: Model name in vLLM
        temperature: Sampling temperature
        trials_dir: Directory for trial logs
        **kwargs: Additional args (ignored)
    
    Returns:
        HarborEnvironment instance
    """
    return HarborEnvironment(
        tasks_dir=tasks_dir,
        environment_type=environment_type,
        environment_kwargs=environment_kwargs,
        max_turns=max_turns,
        enable_summarize=enable_summarize,
        proactive_summarization_threshold=proactive_summarization_threshold,
        trial_timeout_sec=trial_timeout_sec,
        n_parallel_envs=n_parallel_envs,
        vllm_base_url=vllm_base_url,
        model_name=model_name,
        temperature=temperature,
        trials_dir=trials_dir,
    )

