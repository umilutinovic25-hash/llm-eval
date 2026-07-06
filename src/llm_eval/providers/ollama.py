from __future__ import annotations

import time

import httpx

from llm_eval.providers.base import Completion, Provider


class OllamaProvider(Provider):
    """Local models via Ollama's /api/chat endpoint. Free — cost is always $0."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.name = f"ollama:{model}"

    async def complete(self, prompt: str, system: str = "") -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
        data = resp.json()
        return Completion(
            text=data["message"]["content"],
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_s=time.perf_counter() - start,
        )
