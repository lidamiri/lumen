from __future__ import annotations

from collections.abc import Sequence

from lumen.cache import CompletionCache, completion_key
from lumen.models import Completion, Message, ToolSpec
from lumen.providers.base import ChatProvider
from lumen.ratelimit import TokenBucket


class RateLimitedProvider(ChatProvider):
    def __init__(self, inner: ChatProvider, bucket: TokenBucket) -> None:
        self._inner = inner
        self._bucket = bucket

    @property
    def name(self) -> str:
        return self._inner.name

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Sequence[ToolSpec] = (),
    ) -> Completion:
        self._bucket.acquire()
        return self._inner.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )

    def close(self) -> None:
        self._inner.close()


class CachingProvider(ChatProvider):
    def __init__(self, inner: ChatProvider, cache: CompletionCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def name(self) -> str:
        return self._inner.name

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Sequence[ToolSpec] = (),
    ) -> Completion:
        key = completion_key(self._inner.name, model, messages, temperature, max_tokens, tools)
        cached = self._cache.get(key)
        if cached is not None:
            return _mark_cached(cached)
        completion = self._inner.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        self._cache.put(key, completion)
        return completion

    def close(self) -> None:
        self._inner.close()
        self._cache.close()


def _mark_cached(completion: Completion) -> Completion:
    return Completion(
        text=completion.text,
        model=completion.model,
        usage=completion.usage,
        tool_calls=completion.tool_calls,
        finish_reason=completion.finish_reason,
        cached=True,
    )
