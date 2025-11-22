"""Timing utilities for measuring operation durations."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Generator

from rich.console import Console


# Global verbose log callback for use by other modules (retry, etc.)
_verbose_log_callback: Callable[[str], None] | None = None


def set_verbose_log_callback(callback: Callable[[str], None] | None) -> None:
    """Set the global verbose log callback.

    Args:
        callback: Function to call with verbose log messages, or None to disable.
    """
    global _verbose_log_callback
    _verbose_log_callback = callback


def verbose_log(message: str) -> None:
    """Log a message if verbose mode is enabled.

    Args:
        message: The message to log.
    """
    if _verbose_log_callback is not None:
        _verbose_log_callback(message)


@dataclass
class TimingResult:
    """Result of a timed operation."""

    operation: str
    duration_seconds: float

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return self.duration_seconds * 1000

    def format_duration(self) -> str:
        """Format duration for display."""
        if self.duration_seconds < 1:
            return f"{self.duration_ms:.0f}ms"
        elif self.duration_seconds < 60:
            return f"{self.duration_seconds:.1f}s"
        else:
            minutes = int(self.duration_seconds // 60)
            seconds = self.duration_seconds % 60
            return f"{minutes}m {seconds:.1f}s"


@dataclass
class VerboseTimer:
    """Timer for verbose mode that tracks and displays operation timings."""

    enabled: bool = True
    console: Console | None = None
    _results: list[TimingResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.console is None:
            self.console = Console()

    @contextmanager
    def time_operation(
        self,
        operation: str,
        *,
        show_start: bool = True,
        show_end: bool = True,
    ) -> Generator[None, None, None]:
        """Context manager to time an operation.

        Args:
            operation: Name of the operation being timed.
            show_start: Whether to print when operation starts.
            show_end: Whether to print when operation ends with timing.

        Yields:
            None
        """
        if not self.enabled:
            yield
            return

        if show_start:
            self.console.print(f"    [dim]→ {operation}...[/dim]")

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            result = TimingResult(operation=operation, duration_seconds=duration)
            self._results.append(result)

            if show_end:
                self.console.print(
                    f"    [dim]✓ {operation} completed in "
                    f"[cyan]{result.format_duration()}[/cyan][/dim]"
                )

    def log(self, message: str) -> None:
        """Log a verbose message.

        Args:
            message: The message to log.
        """
        if self.enabled:
            self.console.print(f"    [dim]{message}[/dim]")

    def get_results(self) -> list[TimingResult]:
        """Get all timing results."""
        return self._results.copy()

    def get_total_duration(self) -> float:
        """Get total duration of all timed operations."""
        return sum(r.duration_seconds for r in self._results)

    def clear(self) -> None:
        """Clear all timing results."""
        self._results.clear()

    def summary(self) -> str:
        """Get a summary of all timings."""
        if not self._results:
            return "No operations timed"

        total = self.get_total_duration()
        lines = ["Timing Summary:"]
        for result in self._results:
            pct = (result.duration_seconds / total * 100) if total > 0 else 0
            lines.append(f"  {result.operation}: {result.format_duration()} ({pct:.0f}%)")
        lines.append(f"  Total: {TimingResult('total', total).format_duration()}")
        return "\n".join(lines)
