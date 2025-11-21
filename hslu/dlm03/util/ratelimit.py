"""Provides a decorator for rate-limiting function calls."""
import functools
import threading
import time
import types
from collections import deque
from collections.abc import Callable
from typing import ParamSpec, Self, TypeVar

Param = ParamSpec("Param")
RetType = TypeVar("RetType")


class RateLimiter:
    """A thread-safe rate limiter that enforces a maximum number of calls per minute (RPM).

    It uses a sliding window algorithm to allow bursts of calls while ensuring the
    average rate does not exceed the specified RPM.
    """
    _calls: deque[float]
    _lock: threading.Lock
    _rpm: float

    def __init__(self, rpm: float) -> None:
        """Initializes a `RateLimiter` instance.

        Args:
            rpm: The maximum number of requests per minute.
        """
        self._calls = deque()
        self._lock = threading.Lock()
        self._rpm = rpm

    def acquire(self) -> None:
        """Acquire a permit from the rate limiter or wait if the limit is reached."""
        while True:
            with self._lock:
                current_time = time.monotonic()

                while self._calls and self._calls[0] <= current_time - 60:
                    self._calls.popleft()

                if len(self._calls) < self._rpm:
                    self._calls.append(current_time)
                    break

                time_to_wait = self._calls[0] - (current_time - 60)

            if time_to_wait > 0:
                time.sleep(time_to_wait)

    def __enter__(self) -> Self:
        """Enters the rate limiter context."""
        self.acquire()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None,
                 exc_tb: types.TracebackType | None) -> bool | None:
        """Exits the rate limiter context."""


def ratelimit(
        *, rpm: float | None = None,
) -> Callable[[Callable[Param, RetType]], Callable[Param, RetType]]:
    """Decorator that limits the rate at which a function can be called.

    Args:
        rpm: The maximum number of requests per minute. If None, no rate
             limiting is applied.

    Returns:
        A decorator that applies rate limiting to the decorated function.
    """
    if rpm is None:
        return lambda x: x
    rate_limiter = RateLimiter(rpm)

    def decorator(func: Callable[Param, RetType]) -> Callable[Param, RetType]:
        @functools.wraps(func)
        def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
            with rate_limiter:
                return func(*args, **kwargs)

        return wrapper

    return decorator
