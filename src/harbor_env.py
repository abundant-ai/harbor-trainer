from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from datasets import Dataset as HFDataset
import verifiers as vf
from verifiers.types import (
    Messages,
    RolloutInput,
    RolloutTiming,
    SamplingArgs,
    State,
    TrajectoryStep,
    TrajectoryStepTokens,
)

from harbor.models.environment_type import EnvironmentType
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

if TYPE_CHECKING:
    from datasets import Dataset
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class MockUsage:
    """Mock OpenAI usage object for token counting."""
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockResponse:
    """Mock OpenAI response object with usage information."""
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.usage = MockUsage(prompt_tokens, completion_tokens)


def harbor_verification_reward(state: State, **kwargs) -> float:
    """
    Reward function that extracts Harbor's verification result from state.
    
    Harbor Trial stores verification results in state['harbor_result'].
    
    Returns:
        1.0 if task passed verification, 0.0 otherwise
    """
    harbor_result = state.get('harbor_result', {})
    passed = harbor_result.get('passed', False)
    return 1.0 if passed else 0.0


class HarborEnv(vf.MultiTurnEnv):
    """
    Verifiers Environment that wraps Harbor Task execution.
    
    This environment runs Harbor Trials in the rollout method and converts
    the resulting token IDs and logprobs to verifiers' trajectory format
    for RL training with prime-rl.
    """
    
    def __init__(
        self,
        # Harbor task configuration
        tasks_dir: str = "harbor_tasks/extracted_tasks",
        environment_type: str = "docker",
        environment_kwargs: dict[str, Any] | None = None,
        max_turns: int | None = None,
        enable_summarize: bool = True,
        proactive_summarization_threshold: int = 8000,
        trial_timeout_sec: float | None = None,
        n_parallel_envs: int = 8,
        # vLLM config - these come from prime-rl's inference server
        vllm_base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen3-8B",
        temperature: float = 0.7,
        trials_dir: str = "/tmp/harbor-prime-rl",
        # Optional model info for custom/local models
        model_info: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize HarborEnv.
        
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
            model_info: Optional dict for custom model info (context length, etc.)
            **kwargs: Additional args passed to MultiTurnEnv
        """
        self.tasks_dir = Path(tasks_dir)
        self.environment_type = environment_type
        self.environment_kwargs = environment_kwargs or {}
        
        # Handle max_turns: support None (unlimited), int, or string "none"
        if isinstance(max_turns, str) and max_turns.lower() in ("none", "null", "unlimited"):
            self.harbor_max_turns = None
        else:
            self.harbor_max_turns = max_turns
        
        self.enable_summarize = enable_summarize
        self.proactive_summarization_threshold = proactive_summarization_threshold
        self.trial_timeout_sec = trial_timeout_sec
        self.n_parallel_envs = n_parallel_envs
        self.vllm_base_url = vllm_base_url
        self.model_name = model_name
        self.temperature = temperature
        self.trials_dir = Path(trials_dir)
        self.model_info = model_info
        
        # Load Harbor tasks
        self.tasks: list[Task] = []
        if self.tasks_dir.exists():
            for task_dir in sorted(self.tasks_dir.iterdir()):
                if task_dir.is_dir() and (task_dir / "task.toml").exists():
                    try:
                        self.tasks.append(Task(task_dir=task_dir))
                    except Exception as e:
                        logger.warning(f"Failed to load task {task_dir}: {e}")
        
        logger.info(f"Loaded {len(self.tasks)} Harbor tasks from {self.tasks_dir}")
        
        # Create dataset in verifiers format
        dataset = self._create_dataset()
        
        # Create rubric with Harbor verification reward
        rubric = vf.Rubric(
            funcs=[harbor_verification_reward],
            weights=[1.0]
        )
        
        # Initialize parent class
        # Note: We set max_turns=1 for MultiTurnEnv since Harbor handles multi-turn internally
        # Each "turn" from verifiers perspective is a complete Harbor trial
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            max_turns=1,  # Harbor trial is one "turn" from verifiers perspective
            env_id="harbor",
            **kwargs,
        )
        
        # Semaphore for parallel trial execution
        self._semaphore = asyncio.Semaphore(n_parallel_envs)
        
        # LoRA tracking for weight broadcast
        self._current_lora: str | None = None
    
    def _create_dataset(self) -> HFDataset:
        """Convert Harbor tasks to verifiers dataset format."""
        data = []
        for idx, task in enumerate(self.tasks):
            # Load task instruction as prompt
            instruction_file = task.task_dir / "instruction.md"
            if instruction_file.exists():
                prompt_content = instruction_file.read_text()
            else:
                prompt_content = task.config.get("instruction", "Complete the task.")
            
            data.append({
                "prompt": [{"role": "user", "content": prompt_content}],
                "example_id": idx,
                "task": "harbor",
                "answer": "",  # Harbor uses verifier, not ground truth
                "info": {
                    "task_dir": str(task.task_dir),
                    "task_name": task.name,
                },
            })
        
        if data:
            return HFDataset.from_list(data)
        else:
            # Return minimal dataset to avoid errors
            return HFDataset.from_list([{
                "prompt": [{"role": "user", "content": "placeholder"}],
                "example_id": 0,
                "task": "harbor",
                "answer": "",
                "info": {},
            }])
    
    def _get_llm_model_name(self) -> str:
        """Get the model name for LLM, including LoRA adapter if set.
        
        For vLLM (OpenAI-compatible API), LiteLLM requires the `openai/` prefix.
        """
        base_name = self.model_name
        
        # Add openai/ prefix if not already present (required for LiteLLM with vLLM)
        if not base_name.startswith("openai/"):
            base_name = f"openai/{base_name}"
        
        # For LoRA, vLLM expects model name in format "base:adapter"
        if self._current_lora:
            return f"{base_name}:{self._current_lora}"
        return base_name
    
    def update_lora(self, lora_name: str | None):
        """Update LoRA adapter name for weight broadcast.
        
        Called by prime-rl after weight broadcast to point to new adapter.
        """
        self._current_lora = lora_name
        logger.info(f"Updated LoRA adapter: {lora_name}")
    
    # =========================================================================
    # Verifiers MultiTurnEnv Interface Implementation
    # =========================================================================
    
    async def setup_state(self, state: State) -> State:
        """
        Setup state before rollout.
        
        For Harbor, we prepare the task info needed for trial execution.
        """
        # Get task from example_id
        example_id = state.get("example_id", 0)
        if example_id < len(self.tasks):
            task = self.tasks[example_id]
            state["harbor_task"] = task
            state["harbor_task_dir"] = str(task.task_dir)
        else:
            logger.error(f"Invalid example_id {example_id}, only {len(self.tasks)} tasks")
            state["error"] = ValueError(f"Invalid example_id: {example_id}")
        
        return state
    
    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        """
        Generate environment response.
        
        For Harbor, the entire trial runs as a single "turn" from verifiers perspective.
        The env_response triggers the Harbor trial and returns the terminal output.
        """
        # Harbor trial is executed in rollout(), not here
        # Return empty - the trajectory is built from Harbor's rollout details
        return []
    
    @vf.stop
    async def harbor_trial_completed(self, state: State) -> bool:
        """Stop condition: Harbor trial has completed."""
        return state.get("harbor_trial_completed", False)
    
    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """
        Run a complete Harbor trial and convert results to verifiers State.
        
        This overrides the parent rollout to run Harbor's trial execution
        instead of the standard multi-turn loop.
        """
        # Initialize state
        state = await self.init_state(input, client, model, sampling_args)
        
        try:
            state = await self.setup_state(state)
        except vf.Error as e:
            state["error"] = e
            return state
        
        if state.get("error"):
            return state
        
        # Run Harbor trial
        async with self._semaphore:
            try:
                await self._run_harbor_trial(state)
            except asyncio.TimeoutError:
                logger.error(f"Trial timed out for task {state.get('harbor_task_dir')}")
                state["error"] = TimeoutError("Trial timed out")
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                state["error"] = e
        
        # Mark trial as completed for is_completed check
        state["harbor_trial_completed"] = True
        
        # Ensure is_completed runs cleanup
        await self.is_completed(state)
        
        return state
    
    async def _run_harbor_trial(self, state: State) -> None:
        """
        Execute a Harbor trial and populate state with trajectory.
        
        This runs the Terminus-2 agent on the task and converts the
        rollout details to verifiers trajectory format.
        """
        task = state.get("harbor_task")
        if task is None:
            raise ValueError("No harbor_task in state")
        
        # Create trial config
        # Note: We pass LLM parameters instead of an LLM instance to avoid serialization issues
        # Terminus2 will create its own LiteLLM from these parameters
        env_type = EnvironmentType(self.environment_type)
        trial_config = TrialConfig(
            task=TaskConfig(path=task.task_dir),
            trials_dir=self.trials_dir,
            agent=AgentConfig(
                name="terminus-2",
                model_name=self._get_llm_model_name(),
                kwargs={
                    # LLM parameters - Terminus2 will create LiteLLM internally
                    "api_base": self.vllm_base_url,
                    "temperature": self.temperature,
                    "model_info": self.model_info,
                    # Rollout collection for RL training
                    "collect_rollout_details": True,
                    # Agent behavior
                    "enable_summarize": self.enable_summarize,
                    "proactive_summarization_threshold": self.proactive_summarization_threshold,
                    "max_turns": self.harbor_max_turns,
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
        
        # Extract verification result for reward
        passed = False
        if result.verifier_result and result.verifier_result.rewards:
            reward_val = result.verifier_result.rewards.get("reward", 0.0)
            if isinstance(reward_val, bool):
                passed = reward_val
            elif isinstance(reward_val, (int, float)):
                passed = reward_val > 0.5
            else:
                passed = result.verifier_result.rewards.get("pass", False)
        
        state["harbor_result"] = {
            "passed": passed,
            "trial_name": trial_config.trial_name,
        }
        
        # Convert Harbor rollout details to verifiers trajectory
        self._convert_rollout_to_trajectory(state, result)
    
    def _convert_rollout_to_trajectory(self, state: State, result) -> None:
        """
        Convert Harbor's rollout details to verifiers trajectory format.
        
        Harbor's RolloutDetail structure:
        - prompt_token_ids: list[list[int]] - Full prompt token IDs per turn
        - completion_token_ids: list[list[int]] - Response token IDs per turn  
        - logprobs: list[list[float]] - Logprobs per turn
        
        Verifiers TrajectoryStep expects:
        - tokens: TrajectoryStepTokens with prompt_ids, completion_ids, masks, logprobs
        - prompt: Messages (the input messages)
        - completion: Messages (the output messages)
        - reward: float | None
        - advantage: float | None
        """
        trajectory: list[TrajectoryStep] = []
        
        # Get rollout details from agent context
        rollout_details = []
        if result.agent_result and result.agent_result.rollout_details:
            rollout_details = result.agent_result.rollout_details
        
        # Process main agent rollout (first element)
        if rollout_details:
            main_rollout = rollout_details[0]
            
            prompt_ids_per_turn = main_rollout.get("prompt_token_ids", [])
            completion_ids_per_turn = main_rollout.get("completion_token_ids", [])
            logprobs_per_turn = main_rollout.get("logprobs", [])
            
            n_turns = len(completion_ids_per_turn)
            
            for turn_idx in range(n_turns):
                # Get data for this turn
                prompt_ids = prompt_ids_per_turn[turn_idx] if turn_idx < len(prompt_ids_per_turn) else []
                completion_ids = completion_ids_per_turn[turn_idx] if turn_idx < len(completion_ids_per_turn) else []
                turn_logprobs = logprobs_per_turn[turn_idx] if turn_idx < len(logprobs_per_turn) else []
                
                # Skip empty turns
                if not completion_ids:
                    continue
                
                # Ensure logprobs length matches completion_ids
                if len(turn_logprobs) != len(completion_ids):
                    # Pad or truncate logprobs to match completion length
                    if len(turn_logprobs) < len(completion_ids):
                        turn_logprobs = turn_logprobs + [0.0] * (len(completion_ids) - len(turn_logprobs))
                    else:
                        turn_logprobs = turn_logprobs[:len(completion_ids)]
                
                # Create tokens structure
                tokens = TrajectoryStepTokens(
                    prompt_ids=prompt_ids,
                    prompt_mask=[0] * len(prompt_ids),  # Prompt tokens not trained
                    completion_ids=completion_ids,
                    completion_mask=[1] * len(completion_ids),  # Train on completions
                    completion_logprobs=turn_logprobs,
                    overlong_prompt=False,
                    is_truncated=False,
                )
                
                # Create mock response with usage information for prime-rl
                mock_response = MockResponse(
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=len(completion_ids)
                )
                
                # Create trajectory step
                step = TrajectoryStep(
                    prompt=state["prompt"],  # Original prompt
                    completion=[],  # Message format - not needed for token training
                    response=mock_response,  # Mock response with usage info
                    tokens=tokens,
                    reward=None,  # Set at group level
                    advantage=None,  # Set at group level
                    extras={
                        "turn_idx": turn_idx,
                        "task_name": state.get("info", {}).get("task_name", "unknown"),
                    },
                )
                trajectory.append(step)
        
        # Store trajectory - ensure at least one step to avoid IndexError in verifiers
        if not trajectory:
            # Create a minimal trajectory step for failed/empty trials
            # This prevents IndexError when verifiers tries to access state["trajectory"][-1]
            # Even for failed trials, we need to provide token structure for prime-rl
            mock_response = MockResponse(prompt_tokens=1, completion_tokens=1)
            
            # Create minimal tokens structure (prime-rl requires this)
            minimal_tokens = TrajectoryStepTokens(
                prompt_ids=[0],  # Minimal prompt token
                prompt_mask=[0],
                completion_ids=[0],  # Minimal completion token
                completion_mask=[0],  # Don't train on failed trials
                completion_logprobs=[0.0],
                overlong_prompt=False,
                is_truncated=False,
            )
            
            trajectory.append(TrajectoryStep(
                prompt=state.get("prompt", []),
                completion=[{"role": "assistant", "content": "[Trial failed - no completion]"}],
                response=mock_response,  # Mock response even for failed trials
                tokens=minimal_tokens,  # Provide minimal tokens instead of None
                reward=None,
                advantage=None,
                extras={"error": "Harbor trial produced no turns"},
            ))
        
        state["trajectory"] = trajectory
        
        # Set completion from trajectory (required by verifiers)
        state["completion"] = []
        
        # Store timing info
        state["timing"] = RolloutTiming(
            generation_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
            start_time=time.time(),
        )


# =============================================================================
# Entry Point for Prime-RL / Verifiers
# =============================================================================

def load_environment(
    tasks_dir: str = "harbor_tasks/extracted_tasks",
    environment_type: str = "docker",
    environment_kwargs: dict[str, Any] | None = None,
    max_turns: int | None = None,
    enable_summarize: bool = True,
    proactive_summarization_threshold: int = 8000,
    trial_timeout_sec: float | None = None,
    n_parallel_envs: int = 8,
    vllm_base_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen3-8B",
    temperature: float = 0.7,
    trials_dir: str = "/tmp/harbor-prime-rl",
    model_info: dict[str, Any] | None = None,
    **kwargs,
) -> HarborEnv:
    """
    Load Harbor environment (verifiers entry point).
    
    This function is called by prime-rl when loading the environment.
    Prime-RL dynamically imports this module and calls load_environment().
    
    Usage in prime-rl config (*.toml):
    
        [[orchestrator.env]]
        id = "src.harbor_env"
        name = "harbor"
        
        [orchestrator.env.args]
        tasks_dir = "harbor_tasks/extracted_tasks"
        environment_type = "docker"
        max_turns = 20
    
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
        model_info: Optional dict for custom model info
        **kwargs: Additional args (ignored)
    
    Returns:
        HarborEnv instance
    """
    return HarborEnv(
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
        model_info=model_info,
    )
