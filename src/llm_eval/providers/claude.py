from __future__ import annotations

import time

from anthropic import AsyncAnthropic

from llm_eval.providers.base import Completion, Provider


class ClaudeProvider(Provider):
    """Anthropic Claude via the official SDK. Reads ANTHROPIC_API_KEY from env."""

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self.client = AsyncAnthropic()
        self.name = f"claude:{model}"

    async def complete(self, prompt: str, system: str = "") -> Completion:
        kwargs = {}
        if system:
            kwargs["system"] = system
        start = time.perf_counter()
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return Completion(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=time.perf_counter() - start,
        )
