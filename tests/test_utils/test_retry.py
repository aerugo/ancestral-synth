"""Tests for retry utilities with exponential backoff."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ancestral_synth.utils.retry import (
    RetryConfig,
    RetryableError,
    is_retryable_error,
    llm_retry,
    retry_with_backoff,
)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self) -> None:
        """Should have sensible defaults."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 2.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter == True

    def test_custom_config(self) -> None:
        """Should accept custom values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
        )

        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter == False

    def test_calculate_delay_without_jitter(self) -> None:
        """Should calculate exponential delay without jitter."""
        config = RetryConfig(base_delay=2.0, exponential_base=2.0, jitter=False)

        # 2 * 2^0 = 2
        assert config.calculate_delay(0) == 2.0
        # 2 * 2^1 = 4
        assert config.calculate_delay(1) == 4.0
        # 2 * 2^2 = 8
        assert config.calculate_delay(2) == 8.0
        # 2 * 2^3 = 16
        assert config.calculate_delay(3) == 16.0

    def test_calculate_delay_respects_max(self) -> None:
        """Should cap delay at max_delay."""
        config = RetryConfig(
            base_delay=2.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False,
        )

        # 2 * 2^3 = 16, but capped at 10
        assert config.calculate_delay(3) == 10.0
        # 2 * 2^4 = 32, but capped at 10
        assert config.calculate_delay(4) == 10.0

    def test_calculate_delay_with_jitter(self) -> None:
        """Should add jitter to delay."""
        config = RetryConfig(base_delay=2.0, jitter=True)

        # Run multiple times to verify jitter adds variation
        delays = [config.calculate_delay(1) for _ in range(10)]

        # All delays should be positive
        assert all(d > 0 for d in delays)
        # With jitter, we expect some variation (not all the same)
        # Base delay for attempt 1 is 4.0, jitter adds 0-1 second
        assert all(4.0 <= d <= 5.0 for d in delays)


class TestRetryableError:
    """Tests for RetryableError."""

    def test_retryable_error_is_exception(self) -> None:
        """Should be an exception."""
        error = RetryableError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_retryable_error_preserves_cause(self) -> None:
        """Should preserve the original cause."""
        original = ValueError("original error")
        error = RetryableError("wrapped", cause=original)

        assert error.cause is original
        assert str(error) == "wrapped"


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self) -> None:
        """Should not retry on success."""
        call_count = 0

        @retry_with_backoff()
        async def success_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await success_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self) -> None:
        """Should retry on RetryableError."""
        call_count = 0

        @retry_with_backoff(RetryConfig(base_delay=0.01, jitter=False))
        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("temporary failure")
            return "success"

        result = await flaky_func()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        """Should raise after max retries exceeded."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_retries=2, base_delay=0.01, jitter=False))
        async def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise RetryableError("always fails")

        with pytest.raises(RetryableError) as exc_info:
            await always_fails()

        assert "always fails" in str(exc_info.value)
        # Initial call + 2 retries = 3 total
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self) -> None:
        """Should not retry on non-retryable errors."""
        call_count = 0

        @retry_with_backoff(RetryConfig(base_delay=0.01))
        async def raises_value_error() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError) as exc_info:
            await raises_value_error()

        assert "not retryable" in str(exc_info.value)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_custom_retryable_exceptions(self) -> None:
        """Should retry on custom exception types."""
        call_count = 0

        @retry_with_backoff(
            RetryConfig(base_delay=0.01, jitter=False),
            retryable_exceptions=(ConnectionError, TimeoutError),
        )
        async def network_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("connection lost")
            return "connected"

        result = await network_func()

        assert result == "connected"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self) -> None:
        """Should preserve function name and docstring."""

        @retry_with_backoff()
        async def documented_func() -> str:
            """This is a documented function."""
            return "result"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a documented function."

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self) -> None:
        """Should apply exponential backoff delays."""
        call_times: list[float] = []

        @retry_with_backoff(RetryConfig(max_retries=2, base_delay=0.1, jitter=False))
        async def timed_func() -> str:
            call_times.append(asyncio.get_event_loop().time())
            if len(call_times) < 3:
                raise RetryableError("fail")
            return "success"

        await timed_func()

        assert len(call_times) == 3
        # First retry delay should be ~0.1s (2 * 2^0 * 0.05 = 0.1)
        first_delay = call_times[1] - call_times[0]
        # Second retry delay should be ~0.2s (2 * 2^1 * 0.05 = 0.2)
        second_delay = call_times[2] - call_times[1]

        # Allow some tolerance for test execution time
        assert 0.08 <= first_delay <= 0.15
        assert 0.18 <= second_delay <= 0.25

    @pytest.mark.asyncio
    async def test_passes_arguments(self) -> None:
        """Should pass arguments to wrapped function."""

        @retry_with_backoff()
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_passes_kwargs(self) -> None:
        """Should pass keyword arguments to wrapped function."""

        @retry_with_backoff()
        async def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = await greet("World", greeting="Hi")
        assert result == "Hi, World!"

    @pytest.mark.asyncio
    async def test_on_retry_callback(self) -> None:
        """Should call on_retry callback on each retry."""
        retry_info: list[tuple[int, Exception]] = []

        def on_retry(attempt: int, error: Exception) -> None:
            retry_info.append((attempt, error))

        @retry_with_backoff(
            RetryConfig(max_retries=2, base_delay=0.01, jitter=False),
            on_retry=on_retry,
        )
        async def flaky() -> str:
            if len(retry_info) < 2:
                raise RetryableError("failing")
            return "ok"

        await flaky()

        assert len(retry_info) == 2
        assert retry_info[0][0] == 1  # First retry
        assert retry_info[1][0] == 2  # Second retry
        assert all(isinstance(e, RetryableError) for _, e in retry_info)


class TestRetryWithBackoffSync:
    """Tests for retry with synchronous functions."""

    @pytest.mark.asyncio
    async def test_works_with_sync_wrapper(self) -> None:
        """Should work when wrapping sync function in async."""
        call_count = 0

        @retry_with_backoff(RetryConfig(base_delay=0.01, jitter=False))
        async def wrapped_sync() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("sync failure")
            return "sync success"

        result = await wrapped_sync()

        assert result == "sync success"
        assert call_count == 2


class TestIsRetryableError:
    """Tests for is_retryable_error function."""

    def test_rate_limit_error(self) -> None:
        """Should detect rate limit errors."""
        assert is_retryable_error(Exception("Rate limit exceeded"))
        assert is_retryable_error(Exception("rate_limit_exceeded"))
        assert is_retryable_error(Exception("Too many requests"))

    def test_timeout_error(self) -> None:
        """Should detect timeout errors."""
        assert is_retryable_error(Exception("Request timed out"))
        assert is_retryable_error(Exception("Connection timeout"))

    def test_connection_error(self) -> None:
        """Should detect connection errors."""
        assert is_retryable_error(Exception("Connection refused"))
        assert is_retryable_error(Exception("Network error"))

    def test_server_error(self) -> None:
        """Should detect server errors."""
        assert is_retryable_error(Exception("503 Service Unavailable"))
        assert is_retryable_error(Exception("502 Bad Gateway"))
        assert is_retryable_error(Exception("500 Internal Server Error"))
        assert is_retryable_error(Exception("Server temporarily unavailable"))

    def test_overloaded_error(self) -> None:
        """Should detect overloaded errors."""
        assert is_retryable_error(Exception("Model is overloaded"))

    def test_non_retryable_error(self) -> None:
        """Should not detect non-retryable errors."""
        assert not is_retryable_error(Exception("Invalid API key"))
        assert not is_retryable_error(Exception("Model not found"))
        assert not is_retryable_error(Exception("Invalid request"))
        assert not is_retryable_error(ValueError("Something went wrong"))


class TestLLMRetry:
    """Tests for llm_retry decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self) -> None:
        """Should not retry on success."""
        call_count = 0

        @llm_retry(RetryConfig(base_delay=0.01, jitter=False))
        async def success_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await success_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self) -> None:
        """Should retry on rate limit error."""
        call_count = 0

        @llm_retry(RetryConfig(base_delay=0.01, jitter=False))
        async def rate_limited_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Rate limit exceeded")
            return "success"

        result = await rate_limited_func()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self) -> None:
        """Should retry on timeout error."""
        call_count = 0

        @llm_retry(RetryConfig(base_delay=0.01, jitter=False))
        async def timeout_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Request timed out")
            return "success"

        result = await timeout_func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_auth_error(self) -> None:
        """Should not retry on authentication error."""
        call_count = 0

        @llm_retry(RetryConfig(base_delay=0.01, jitter=False))
        async def auth_func() -> str:
            nonlocal call_count
            call_count += 1
            raise Exception("Invalid API key")

        with pytest.raises(Exception) as exc_info:
            await auth_func()

        assert "Invalid API key" in str(exc_info.value)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error_type(self) -> None:
        """Should retry on RetryableError type."""
        call_count = 0

        @llm_retry(RetryConfig(base_delay=0.01, jitter=False))
        async def retryable_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("custom retryable")
            return "success"

        result = await retryable_func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        """Should raise after max retries."""
        call_count = 0

        @llm_retry(RetryConfig(max_retries=2, base_delay=0.01, jitter=False))
        async def always_rate_limited() -> str:
            nonlocal call_count
            call_count += 1
            raise Exception("Rate limit exceeded")

        with pytest.raises(Exception) as exc_info:
            await always_rate_limited()

        assert "Rate limit" in str(exc_info.value)
        # Initial + 2 retries = 3 total
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self) -> None:
        """Should preserve function metadata."""

        @llm_retry()
        async def documented_llm_func() -> str:
            """This is an LLM function."""
            return "result"

        assert documented_llm_func.__name__ == "documented_llm_func"
        assert documented_llm_func.__doc__ == "This is an LLM function."

    @pytest.mark.asyncio
    async def test_on_retry_callback(self) -> None:
        """Should call on_retry callback."""
        retry_info: list[tuple[int, Exception, float]] = []

        def on_retry(attempt: int, error: Exception, delay: float) -> None:
            retry_info.append((attempt, error, delay))

        @llm_retry(
            RetryConfig(max_retries=2, base_delay=0.01, jitter=False),
            on_retry=on_retry,
        )
        async def flaky_llm() -> str:
            if len(retry_info) < 2:
                raise Exception("Rate limit exceeded")
            return "ok"

        await flaky_llm()

        assert len(retry_info) == 2
        assert retry_info[0][0] == 1
        assert retry_info[1][0] == 2
        assert retry_info[0][2] >= 0  # delay is non-negative
