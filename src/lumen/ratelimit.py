import threading
import time
from collections.abc import Callable


class TokenBucket:
    def __init__(
        self,
        rate_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self._capacity = float(rate_per_minute)
        self._tokens = float(rate_per_minute)
        self._rate = rate_per_minute / 60.0
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                elapsed = max(0.0, now - self._updated)
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = (tokens - self._tokens) / self._rate
            self._sleep(deficit)
            waited += deficit
