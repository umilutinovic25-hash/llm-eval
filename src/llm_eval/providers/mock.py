from __future__ import annotations

import asyncio
import random
import time

from llm_eval.providers.base import Completion, Provider


class MockProvider(Provider):
    """Deterministic-ish fake provider for developing the framework offline.

    Echoes a canned answer when the prompt contains a recognizable marker,
    otherwise returns the last line of the prompt. Simulates latency.
    """

    name = "mock"

    async def complete(self, prompt: str, system: str = "") -> Completion:
        start = time.perf_counter()
        await asyncio.sleep(random.uniform(0.05, 0.2))
        # Test hook: an eval case can embed "MOCK_ANSWER: xyz" in its prompt
        # so the mock passes/fails predictably while developing graders.
        answer = None
        for line in prompt.splitlines():
            if line.startswith("MOCK_ANSWER:"):
                answer = line[len("MOCK_ANSWER:"):].strip()
        if answer is None:
            answer = prompt.strip().splitlines()[-1] if prompt.strip() else ""
        return Completion(
            text=answer,
            model="mock",
            input_tokens=len(prompt) // 4,
            output_tokens=len(answer) // 4,
            latency_s=time.perf_counter() - start,
        )
