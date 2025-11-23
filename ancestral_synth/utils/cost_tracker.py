"""Cost tracking utilities for LLM API calls."""

from dataclasses import dataclass, field
from typing import Literal


# Pricing per 1M tokens (as of late 2024/early 2025)
# Format: (input_price_per_1m, output_price_per_1m)
MODEL_PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "openai": {
        # GPT-4o models
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-2024-11-20": (2.50, 10.00),
        "gpt-4o-2024-08-06": (2.50, 10.00),
        "gpt-4o-2024-05-13": (5.00, 15.00),
        # GPT-4o mini
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o-mini-2024-07-18": (0.15, 0.60),
        # GPT-4 Turbo
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4-turbo-2024-04-09": (10.00, 30.00),
        "gpt-4-turbo-preview": (10.00, 30.00),
        # GPT-4
        "gpt-4": (30.00, 60.00),
        "gpt-4-32k": (60.00, 120.00),
        # GPT-3.5
        "gpt-3.5-turbo": (0.50, 1.50),
        "gpt-3.5-turbo-0125": (0.50, 1.50),
        # o1 models
        "o1": (15.00, 60.00),
        "o1-2024-12-17": (15.00, 60.00),
        "o1-preview": (15.00, 60.00),
        "o1-preview-2024-09-12": (15.00, 60.00),
        "o1-mini": (3.00, 12.00),
        "o1-mini-2024-09-12": (3.00, 12.00),
        # o3-mini
        "o3-mini": (1.10, 4.40),
        "o3-mini-2025-01-31": (1.10, 4.40),
    },
    "anthropic": {
        # Claude 3.5 models
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "claude-3-5-sonnet-latest": (3.00, 15.00),
        "claude-3-5-sonnet-20240620": (3.00, 15.00),
        "claude-3-5-haiku-20241022": (0.80, 4.00),
        "claude-3-5-haiku-latest": (0.80, 4.00),
        # Claude 3 models
        "claude-3-opus-20240229": (15.00, 75.00),
        "claude-3-opus-latest": (15.00, 75.00),
        "claude-3-sonnet-20240229": (3.00, 15.00),
        "claude-3-haiku-20240307": (0.25, 1.25),
        # Claude 4 (Sonnet 4)
        "claude-sonnet-4-20250514": (3.00, 15.00),
    },
    "google": {
        # Gemini 2.0 models
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-2.0-flash-exp": (0.00, 0.00),  # Free tier
        "gemini-2.0-flash-thinking-exp": (0.00, 0.00),  # Free tier
        # Gemini 1.5 models
        "gemini-1.5-pro": (1.25, 5.00),
        "gemini-1.5-pro-latest": (1.25, 5.00),
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.5-flash-latest": (0.075, 0.30),
        "gemini-1.5-flash-8b": (0.0375, 0.15),
        "gemini-1.5-flash-8b-latest": (0.0375, 0.15),
        # Gemini Pro (legacy)
        "gemini-pro": (0.50, 1.50),
    },
    "ollama": {
        # Ollama models are free (local)
        "default": (0.00, 0.00),
    },
}

# Default pricing for unknown models (conservative estimate)
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "openai": (5.00, 15.00),  # Default to GPT-4o pricing
    "anthropic": (3.00, 15.00),  # Default to Claude 3.5 Sonnet pricing
    "google": (1.25, 5.00),  # Default to Gemini 1.5 Pro pricing
    "ollama": (0.00, 0.00),  # Ollama is free
}


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.input_tokens + self.output_tokens


@dataclass
class CostResult:
    """Cost calculation result."""

    input_cost: float
    output_cost: float
    usage: TokenUsage

    @property
    def total_cost(self) -> float:
        """Get total cost."""
        return self.input_cost + self.output_cost


def get_model_pricing(
    provider: str, model: str
) -> tuple[float, float]:
    """Get pricing for a model.

    Args:
        provider: The provider name (openai, anthropic, google, ollama).
        model: The model name.

    Returns:
        Tuple of (input_price_per_1m, output_price_per_1m).
    """
    provider_pricing = MODEL_PRICING.get(provider, {})

    # Try exact match first
    if model in provider_pricing:
        return provider_pricing[model]

    # Try prefix match (e.g., "gpt-4o-mini" matches "gpt-4o-mini-2024-07-18")
    for model_name, pricing in provider_pricing.items():
        if model.startswith(model_name) or model_name.startswith(model):
            return pricing

    # Return default for provider
    return DEFAULT_PRICING.get(provider, (5.00, 15.00))


def calculate_cost(
    provider: str,
    model: str,
    usage: TokenUsage,
) -> CostResult:
    """Calculate cost for token usage.

    Args:
        provider: The provider name.
        model: The model name.
        usage: Token usage.

    Returns:
        Cost calculation result.
    """
    input_price, output_price = get_model_pricing(provider, model)

    # Convert from per-1M to actual cost
    input_cost = (usage.input_tokens / 1_000_000) * input_price
    output_cost = (usage.output_tokens / 1_000_000) * output_price

    return CostResult(
        input_cost=input_cost,
        output_cost=output_cost,
        usage=usage,
    )


@dataclass
class PersonCost:
    """Cost tracking for a single person's generation."""

    biography_cost: CostResult | None = None
    extraction_cost: CostResult | None = None
    correction_costs: list[CostResult] = field(default_factory=list)
    dedup_costs: list[CostResult] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        """Get total cost for this person."""
        total = 0.0
        if self.biography_cost:
            total += self.biography_cost.total_cost
        if self.extraction_cost:
            total += self.extraction_cost.total_cost
        for cost in self.correction_costs:
            total += cost.total_cost
        for cost in self.dedup_costs:
            total += cost.total_cost
        return total

    @property
    def total_tokens(self) -> TokenUsage:
        """Get total tokens for this person."""
        total = TokenUsage()
        if self.biography_cost:
            total.input_tokens += self.biography_cost.usage.input_tokens
            total.output_tokens += self.biography_cost.usage.output_tokens
        if self.extraction_cost:
            total.input_tokens += self.extraction_cost.usage.input_tokens
            total.output_tokens += self.extraction_cost.usage.output_tokens
        for cost in self.correction_costs:
            total.input_tokens += cost.usage.input_tokens
            total.output_tokens += cost.usage.output_tokens
        for cost in self.dedup_costs:
            total.input_tokens += cost.usage.input_tokens
            total.output_tokens += cost.usage.output_tokens
        return total

    @property
    def llm_call_count(self) -> int:
        """Get number of LLM calls for this person."""
        count = 0
        if self.biography_cost:
            count += 1
        if self.extraction_cost:
            count += 1
        count += len(self.correction_costs)
        count += len(self.dedup_costs)
        return count


class CostTracker:
    """Tracks costs across multiple API calls and persons."""

    def __init__(self, provider: str, model: str) -> None:
        """Initialize the cost tracker.

        Args:
            provider: The LLM provider name.
            model: The model name.
        """
        self.provider = provider
        self.model = model
        self._current_person: PersonCost | None = None
        self._completed_persons: list[PersonCost] = []

    def start_person(self) -> None:
        """Start tracking costs for a new person."""
        self._current_person = PersonCost()

    def record_biography(self, usage: TokenUsage) -> CostResult:
        """Record biography generation cost.

        Args:
            usage: Token usage from the API call.

        Returns:
            The calculated cost.
        """
        cost = calculate_cost(self.provider, self.model, usage)
        if self._current_person:
            self._current_person.biography_cost = cost
        return cost

    def record_extraction(self, usage: TokenUsage) -> CostResult:
        """Record data extraction cost.

        Args:
            usage: Token usage from the API call.

        Returns:
            The calculated cost.
        """
        cost = calculate_cost(self.provider, self.model, usage)
        if self._current_person:
            self._current_person.extraction_cost = cost
        return cost

    def record_correction(self, usage: TokenUsage) -> CostResult:
        """Record correction cost.

        Args:
            usage: Token usage from the API call.

        Returns:
            The calculated cost.
        """
        cost = calculate_cost(self.provider, self.model, usage)
        if self._current_person:
            self._current_person.correction_costs.append(cost)
        return cost

    def record_dedup(self, usage: TokenUsage) -> CostResult:
        """Record deduplication check cost.

        Args:
            usage: Token usage from the API call.

        Returns:
            The calculated cost.
        """
        cost = calculate_cost(self.provider, self.model, usage)
        if self._current_person:
            self._current_person.dedup_costs.append(cost)
        return cost

    def finish_person(self) -> PersonCost | None:
        """Finish tracking the current person and return their costs.

        Returns:
            The completed person's costs, or None if no person was being tracked.
        """
        if self._current_person:
            completed = self._current_person
            self._completed_persons.append(completed)
            self._current_person = None
            return completed
        return None

    @property
    def current_person_cost(self) -> float:
        """Get the current person's cost so far."""
        if self._current_person:
            return self._current_person.total_cost
        return 0.0

    @property
    def running_total(self) -> float:
        """Get the running total cost."""
        total = sum(p.total_cost for p in self._completed_persons)
        if self._current_person:
            total += self._current_person.total_cost
        return total

    @property
    def total_tokens(self) -> TokenUsage:
        """Get total tokens across all persons."""
        total = TokenUsage()
        for person in self._completed_persons:
            person_tokens = person.total_tokens
            total.input_tokens += person_tokens.input_tokens
            total.output_tokens += person_tokens.output_tokens
        if self._current_person:
            current_tokens = self._current_person.total_tokens
            total.input_tokens += current_tokens.input_tokens
            total.output_tokens += current_tokens.output_tokens
        return total

    @property
    def total_llm_calls(self) -> int:
        """Get total number of LLM calls."""
        total = sum(p.llm_call_count for p in self._completed_persons)
        if self._current_person:
            total += self._current_person.llm_call_count
        return total

    def get_summary(self) -> dict:
        """Get a summary of all costs.

        Returns:
            Dictionary with cost summary.
        """
        tokens = self.total_tokens
        return {
            "total_cost": self.running_total,
            "total_persons": len(self._completed_persons),
            "total_llm_calls": self.total_llm_calls,
            "total_input_tokens": tokens.input_tokens,
            "total_output_tokens": tokens.output_tokens,
            "provider": self.provider,
            "model": self.model,
        }


def format_cost(cost: float) -> str:
    """Format a cost value for display.

    Args:
        cost: The cost in dollars.

    Returns:
        Formatted cost string.
    """
    if cost < 0.0001:
        return "$0.0000"
    elif cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.00:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


def format_tokens(tokens: int) -> str:
    """Format token count for display.

    Args:
        tokens: Number of tokens.

    Returns:
        Formatted token string.
    """
    if tokens < 1000:
        return str(tokens)
    elif tokens < 1_000_000:
        return f"{tokens / 1000:.1f}k"
    else:
        return f"{tokens / 1_000_000:.2f}M"
