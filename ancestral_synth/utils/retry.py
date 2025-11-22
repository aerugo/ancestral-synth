"""Retry utilities with exponential backoff for LLM calls."""

import asyncio
import functools
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Common LLM error patterns that indicate transient failures
RETRYABLE_ERROR_PATTERNS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "timeout",
    "timed out",
    "connection",
    "network",
    "temporarily unavailable",
    "service unavailable",
    "503",
    "502",
    "500",
    "overloaded",
)


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum delay in seconds.
        exponential_base: Base for exponential calculation.
        jitter: Whether to add random jitter to delays.
    """

    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.

        Args:
            attempt: The attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        # Exponential backoff: base_delay * exponential_base^attempt
        delay = self.base_delay * (self.exponential_base**attempt)

        # Cap at max_delay
        delay = min(delay, self.max_delay)

        # Add jitter if enabled (0-1 seconds)
        if self.jitter:
            delay += random.random()

        return delay


class RetryableError(Exception):
    """Exception indicating an operation can be retried.

    Use this to wrap transient errors from LLM providers.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize the error.

        Args:
            message: Error message.
            cause: Original exception that caused this error.
        """
        super().__init__(message)
        self.cause = cause


def retry_with_backoff(
    config: RetryConfig | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (RetryableError,),
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for async functions with exponential backoff retry.

    Args:
        config: Retry configuration. Uses defaults if not provided.
        retryable_exceptions: Exception types that should trigger retry.
        on_retry: Optional callback called on each retry with attempt number and exception.

    Returns:
        Decorated function with retry logic.

    Example:
        @retry_with_backoff(RetryConfig(max_retries=3))
        async def call_llm():
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    # Check if we have retries left
                    if attempt < config.max_retries:
                        # Calculate delay for this retry
                        delay = config.calculate_delay(attempt)

                        # Call on_retry callback if provided
                        if on_retry is not None:
                            on_retry(attempt + 1, e)

                        # Wait before retrying
                        await asyncio.sleep(delay)
                    else:
                        # No more retries, raise the exception
                        raise

            # This should not be reached, but just in case
            if last_exception is not None:
                raise last_exception

        return wrapper

    return decorator


def is_retryable_error(error: Exception) -> bool:
    """Check if an exception is likely a transient error that can be retried.

    Args:
        error: The exception to check.

    Returns:
        True if the error appears to be transient and retryable.
    """
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in RETRYABLE_ERROR_PATTERNS)


def llm_retry(
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator specifically for LLM calls with automatic error classification.

    This decorator wraps LLM calls and automatically retries on transient errors
    like rate limits, timeouts, and connection errors.

    Args:
        config: Retry configuration. Uses defaults if not provided.
        on_retry: Optional callback called on each retry.

    Returns:
        Decorated function with retry logic.

    Example:
        @llm_retry()
        async def call_model():
            return await agent.run(prompt)
    """
    if config is None:
        config = RetryConfig()

    def default_on_retry(attempt: int, error: Exception) -> None:
        logger.warning(
            "LLM call failed (attempt %d/%d): %s. Retrying...",
            attempt,
            config.max_retries + 1,
            str(error)[:100],
        )

    retry_callback = on_retry or default_on_retry

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check if this is a retryable error
                    if isinstance(e, RetryableError) or is_retryable_error(e):
                        last_exception = e

                        if attempt < config.max_retries:
                            delay = config.calculate_delay(attempt)
                            retry_callback(attempt + 1, e)
                            await asyncio.sleep(delay)
                        else:
                            raise
                    else:
                        # Non-retryable error, raise immediately
                        raise

            if last_exception is not None:
                raise last_exception

        return wrapper

    return decorator
