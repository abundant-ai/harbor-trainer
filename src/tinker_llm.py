from __future__ import annotations

from typing import Any

import tinker
from harbor.llms.base import (
    BaseLLM,
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.models.metric import UsageInfo
from pydantic import BaseModel, ConfigDict, PrivateAttr
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer


class LogprobsMissingError(RuntimeError):
    """Raised when Tinker does not return logprobs for sampled tokens."""


class TinkerLLM(BaseLLM, BaseModel):
    """
    LLM backend using Tinker SamplingClient for RL training.

    This allows Terminus2 (with llm=TinkerLLM) to use the same agent harness and
    conversation management as the evaluation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Serializable fields (type declarations for Pydantic)
    model_name: str
    max_tokens: int
    temperature: float
    context_limit: int

    # Private attributes excluded from serialization
    _client: tinker.SamplingClient = PrivateAttr()
    _tokenizer: Tokenizer = PrivateAttr()
    _renderer: Renderer = PrivateAttr()

    def __init__(
            self,
            sampling_client: tinker.SamplingClient,
            tokenizer: Tokenizer,
            renderer: Renderer,
            model_name: str,
            max_tokens: int,
            temperature: float,
            context_limit: int,
    ):
        """
        Initialize TinkerLLM.

        Args:
            sampling_client: Tinker SamplingClient for sampling
            tokenizer: tinker_cookbook Tokenizer for encoding/decoding text
            renderer: tinker_cookbook Renderer for formatting conversations
            model_name: Name of the model (for metadata)
            max_tokens: Maximum tokens to generate per response
            temperature: Sampling temperature
            context_limit: Maximum context length for the model
        """
        # Pydantic's __init__ sets self.model_name, self.temperature, etc.
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            context_limit=context_limit,
        )
        self._client = sampling_client
        self._tokenizer = tokenizer
        self._renderer = renderer

    @property
    def tokenizer(self) -> Tokenizer:
        """Get the tinker_cookbook tokenizer."""
        return self._tokenizer

    @property
    def renderer(self) -> Renderer:
        """Get the tinker_cookbook renderer."""
        return self._renderer

    def get_model_context_limit(self) -> int:
        """Get the context limit (max input tokens) for the current model."""
        return self.context_limit

    async def call(
            self,
            prompt: str,
            message_history: list[dict[str, Any]] | None = None,
            **kwargs,
    ) -> LLMResponse:
        """
        Sample from Tinker and return an LLMResponse.

        Args:
            prompt: The user prompt to add
            message_history: Previous conversation messages
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            LLMResponse with content, token IDs, and logprobs

        Raises:
            ContextLengthExceededError: If the input exceeds context limit
            OutputLengthExceededError: If the output was truncated
        """
        # Build conversation messages
        if message_history is None:
            message_history = []
        messages = message_history + [{"role": "user", "content": prompt}]

        # Use tinker_cookbook renderer for message formatting
        model_input = self._renderer.build_generation_prompt(messages)

        # Get stop sequences from renderer
        stop_sequences = self._renderer.get_stop_sequences()

        # Check context length
        input_length = model_input.length
        if input_length > self.context_limit - self.max_tokens:
            raise ContextLengthExceededError(
                f"Context length {input_length} exceeds limit "
                f"{self.context_limit - self.max_tokens}"
            )

        # Sample from Tinker
        result = await self._client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=stop_sequences,
            ),
        )

        # Extract from response
        sampled_seq = result.sequences[0]
        completion_tokens = list(sampled_seq.tokens)
        completion_logprobs = (
            list(sampled_seq.logprobs) if sampled_seq.logprobs else None
        )

        if completion_logprobs is None:
            raise LogprobsMissingError(
                "Tinker sample response did not include logprobs; enable logprob "
                "return from the Tinker API so RL training can compute advantages."
            )

        # Decode using tinker_cookbook tokenizer
        content = self._tokenizer.decode(completion_tokens)

        # Check for truncation
        if sampled_seq.stop_reason == "length":
            raise OutputLengthExceededError(
                f"Output truncated at {self.max_tokens} tokens",
                truncated_response=content,
            )

        # Get prompt token IDs
        prompt_token_ids = model_input.to_ints()

        return LLMResponse(
            content=content,
            reasoning_content=None,
            usage=UsageInfo(
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(completion_tokens),
                cache_tokens=0,
                cost_usd=0.0,
            ),
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_tokens,
            logprobs=completion_logprobs,
        )
