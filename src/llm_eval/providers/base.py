from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class Provider(abc.ABC):
    """One LLM backend. Implementations must be safe to call concurrently."""

    name: str

    @abc.abstractmethod
    async def complete(self, prompt: str, system: str = "") -> Completion:
        ...
