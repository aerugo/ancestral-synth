"""Tests for rate limiting utilities."""

import asyncio
import time

import pytest

from ancestral_synth.utils.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    rate_limited,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_config(self) -> None:
        """Should have sensible defaults."""
        config = RateLimitConfig()

        assert config.requests_per_minute == 60
        assert config.requests_per_second == 1.0

    def test_custom_config(self) -> None:
        """Should accept custom values."""
        config = RateLimitConfig(requests_per_minute=30)

        assert config.requests_per_minute == 30
        assert config.requests_per_second == 0.5


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_first_request_immediate(self) -> None:
        """First request should not be delayed."""
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=60))

        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_respects_rate_limit(self) -> None:
        """Should delay requests to respect rate limit."""
        # 60 requests per minute = 1 per second
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=60))

        # Make 3 rapid requests
        times: list[float] = []
        for _ in range(3):
            await limiter.acquire()
            times.append(time.time())

        # Check intervals between requests
        interval_1 = times[1] - times[0]
        interval_2 = times[2] - times[1]

        # Each interval should be approximately 1 second
        assert interval_1 >= 0.9
        assert interval_2 >= 0.9

    @pytest.mark.asyncio
    async def test_high_rate_limit(self) -> None:
        """Should allow rapid requests with high rate limit."""
        # 600 requests per minute = 10 per second
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=600))

        start = time.time()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.time() - start

        # 5 requests at 10/second should take ~0.4 seconds
        assert elapsed < 0.6

    @pytest.mark.asyncio
    async def test_tokens_property(self) -> None:
        """Should track available tokens."""
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=60))

        # Fresh limiter should have capacity
        assert limiter.available_tokens > 0

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """Should reset token count."""
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=60))

        # Use some tokens
        await limiter.acquire()
        await limiter.acquire()

        # Reset
        limiter.reset()

        # Should have full tokens again
        assert limiter.available_tokens >= 1.0


class TestRateLimitedDecorator:
    """Tests for rate_limited decorator."""

    @pytest.mark.asyncio
    async def test_decorated_function_works(self) -> None:
        """Decorated function should work normally."""
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=600))

        @rate_limited(limiter)
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_decorated_function_rate_limited(self) -> None:
        """Decorated function should be rate limited."""
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=60))

        @rate_limited(limiter)
        async def quick_func() -> str:
            return "done"

        start = time.time()
        await quick_func()
        await quick_func()
        elapsed = time.time() - start

        # Second call should be delayed by ~1 second
        assert elapsed >= 0.9

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self) -> None:
        """Should preserve function name and docstring."""
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=60))

        @rate_limited(limiter)
        async def documented_func() -> str:
            """This is a documented function."""
            return "result"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a documented function."

    @pytest.mark.asyncio
    async def test_passes_arguments(self) -> None:
        """Should pass arguments correctly."""
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=600))

        @rate_limited(limiter)
        async def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = await greet("World", greeting="Hi")
        assert result == "Hi, World!"

    @pytest.mark.asyncio
    async def test_propagates_exceptions(self) -> None:
        """Should propagate exceptions from decorated function."""
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=600))

        @rate_limited(limiter)
        async def fails() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError) as exc_info:
            await fails()

        assert "test error" in str(exc_info.value)


class TestConcurrentRateLimiting:
    """Tests for concurrent access to rate limiter."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_serialized(self) -> None:
        """Concurrent requests should be serialized through rate limiter."""
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=60))
        results: list[float] = []

        async def task() -> None:
            await limiter.acquire()
            results.append(time.time())

        # Launch 3 concurrent tasks
        await asyncio.gather(*[task() for _ in range(3)])

        # Check that requests were spaced out
        assert len(results) == 3
        interval_1 = results[1] - results[0]
        interval_2 = results[2] - results[1]

        assert interval_1 >= 0.9
        assert interval_2 >= 0.9
