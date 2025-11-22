"""Rate limiting utilities for LLM API calls."""

import asyncio
import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        requests_per_minute: Maximum requests per minute.
    """

    requests_per_minute: int = 60

    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second."""
        return self.requests_per_minute / 60.0


class RateLimiter:
    """Token bucket rate limiter for async operations.

    Uses a token bucket algorithm to enforce rate limits.
    Tokens are added at a constant rate, and each request
    consumes one token.
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        """Initialize the rate limiter.

        Args:
            config: Rate limit configuration.
        """
        self._config = config or RateLimitConfig()
        self._tokens = 1.0  # Start with one token available
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
        self._last_wait_time: float = 0.0

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (approximate)."""
        elapsed = time.monotonic() - self._last_update
        tokens = self._tokens + elapsed * self._config.requests_per_second
        return min(tokens, 1.0)  # Cap at 1 token max

    @property
    def last_wait_time(self) -> float:
        """Get the last wait time in seconds."""
        return self._last_wait_time

    def reset(self) -> None:
        """Reset the rate limiter to initial state."""
        self._tokens = 1.0
        self._last_update = time.monotonic()
        self._last_wait_time = 0.0

    async def acquire(self) -> float:
        """Acquire permission to make a request.

        This method will block until a token is available,
        ensuring the rate limit is respected.

        Returns:
            The time spent waiting in seconds (0 if no wait needed).
        """
        async with self._lock:
            # Update token count based on elapsed time
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens += elapsed * self._config.requests_per_second
            self._last_update = now

            # Cap tokens at 1 (no bursting)
            self._tokens = min(self._tokens, 1.0)

            # If we don't have a token, wait until we do
            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self._config.requests_per_second
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
                self._last_update = time.monotonic()
                self._last_wait_time = wait_time
                return wait_time
            else:
                # Consume a token
                self._tokens -= 1.0
                self._last_wait_time = 0.0
                return 0.0


def rate_limited(
    limiter: RateLimiter,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to apply rate limiting to an async function.

    Args:
        limiter: The rate limiter to use.

    Returns:
        Decorated function with rate limiting.

    Example:
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=60))

        @rate_limited(limiter)
        async def call_api():
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            await limiter.acquire()
            return await func(*args, **kwargs)

        return wrapper

    return decorator
