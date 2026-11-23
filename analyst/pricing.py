# Copied from conversational-form-agent (commit 6e67866), app/pricing.py. Unchanged.
"""LLM token prices, used to estimate the cost of each call.

Prices are USD per 1M tokens, copied from the providers' pricing pages on
2026-11-20. Update them here when prices change; nothing else hard-codes them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input: float  # cache miss
    cached_input: float  # cache hit
    output: float


PRICES: dict[str, ModelPrice] = {
    # api-docs.deepseek.com/quick_start/pricing. DeepSeek charges half outside peak
    # hours (peak: 01:00-04:00 and 06:00-10:00 UTC on weekdays); we always use the
    # peak price, so estimates are an upper bound.
    "deepseek-flash": ModelPrice(input=0.30, cached_input=0.006, output=1.20),
    "deepseek-v4-pro": ModelPrice(input=1.32, cached_input=0.044, output=3.96),
    # developers.openai.com/api/docs/pricing (standard tier).
    "gpt-6-luna": ModelPrice(input=0.10, cached_input=0.01, output=0.50),
}


def estimate_cost_usd(
    model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int
) -> float | None:
    """Cost of one call in USD, or None if the model's price is unknown."""
    price = PRICES.get(model)
    if price is None:
        return None
    uncached = max(input_tokens - cached_input_tokens, 0)
    return (
        uncached * price.input
        + cached_input_tokens * price.cached_input
        + output_tokens * price.output
    ) / 1_000_000
