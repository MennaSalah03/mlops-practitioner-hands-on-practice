"""Reusable decorators."""

import time
from functools import wraps

import structlog

logger = structlog.get_logger()


def timed(func):
    """Logs execution time of the wrapped function via structlog."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "timed_execution",
                function=func.__qualname__,
                duration_ms=round(elapsed_ms, 2),
            )

    return wrapper
