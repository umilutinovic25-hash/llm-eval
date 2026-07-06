from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Tuple

# USD per 1M tokens (input, output). Source: platform.claude.com pricing,
# cached 2026-06. Ollama and mock run locally, so they're free.
PRICING: Dict[str, Tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Cost of one completion, or None if the model isn't in the price list (local models)."""
    for known, (in_price, out_price) in PRICING.items():
        if model.startswith(known):
            return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
    return None


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[max(0, min(98, int(pct) - 1))]


def summarize(latencies: List[float]) -> Dict[str, float]:
    return {
        "p50": percentile(sorted(latencies), 50),
        "p95": percentile(sorted(latencies), 95),
        "mean": statistics.mean(latencies) if latencies else 0.0,
    }
