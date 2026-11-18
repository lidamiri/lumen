import pytest
from helpers import ScriptedProvider

from lumen.cache import CompletionCache, completion_key
from lumen.models import Completion, Message, Role
from lumen.providers.wrappers import CachingProvider
from lumen.ratelimit import TokenBucket


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_bucket_allows_burst_then_waits_for_refill():
    clock = FakeClock()
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.now += seconds

    bucket = TokenBucket(60, clock=clock, sleep=sleep)
    for _ in range(60):
        bucket.acquire()
    assert slept == []
    bucket.acquire()
    assert len(slept) == 1
    assert slept[0] == pytest.approx(1.0)


def test_bucket_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        TokenBucket(0)


def test_cache_round_trip_preserves_completion():
    cache = CompletionCache(":memory:")
    completion = Completion(text="cached", model="m")
    cache.put("key", completion)
    restored = cache.get("key")
    assert restored is not None
    assert restored.text == "cached"
    assert len(cache) == 1
    assert cache.get("missing") is None


def test_completion_key_depends_on_inputs():
    base = [Message(role=Role.USER, content="hello")]
    key = completion_key("p", "m", base, 0.0, 100, [])
    assert key == completion_key("p", "m", list(base), 0.0, 100, [])
    assert key != completion_key("p", "m", [Message(role=Role.USER, content="bye")], 0.0, 100, [])
    assert key != completion_key("p", "other", base, 0.0, 100, [])
    assert key != completion_key("p", "m", base, 0.5, 100, [])


def test_caching_provider_serves_repeated_request_from_cache():
    provider = ScriptedProvider([Completion(text="fresh", model="scripted")])
    cached = CachingProvider(provider, CompletionCache(":memory:"))
    messages = [Message(role=Role.USER, content="same question")]
    first = cached.complete(messages)
    second = cached.complete(messages)
    assert first.cached is False
    assert second.cached is True
    assert second.text == "fresh"
    assert len(provider.calls) == 1
