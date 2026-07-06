from llm_eval.providers.base import Completion, Provider
from llm_eval.providers.mock import MockProvider
from llm_eval.providers.ollama import OllamaProvider


def create_provider(spec: str) -> Provider:
    """Create a provider from a spec string.

    Formats:
        mock
        ollama:<model>          e.g. ollama:llama3.2
        claude:<model>          e.g. claude:claude-opus-4-8
    """
    name, _, model = spec.partition(":")
    if name == "mock":
        return MockProvider()
    if name == "ollama":
        return OllamaProvider(model or "llama3.2")
    if name == "claude":
        from llm_eval.providers.claude import ClaudeProvider  # lazy: needs anthropic pkg
        return ClaudeProvider(model or "claude-opus-4-8")
    raise ValueError(f"Unknown provider: {spec!r} (expected mock, ollama:<model> or claude:<model>)")
