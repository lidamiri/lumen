from __future__ import annotations

from dataclasses import dataclass

from lumen.cache import CompletionCache
from lumen.config import Settings, get_settings
from lumen.embeddings.base import Embedder
from lumen.embeddings.hashing import HashingEmbedder
from lumen.embeddings.openai_embedder import OpenAIEmbedder
from lumen.prompts.library import default_registry
from lumen.prompts.template import PromptRegistry
from lumen.providers.anthropic_provider import AnthropicProvider
from lumen.providers.base import ChatProvider
from lumen.providers.echo import EchoProvider
from lumen.providers.openai_provider import OpenAIProvider
from lumen.providers.wrappers import CachingProvider, RateLimitedProvider
from lumen.ratelimit import TokenBucket
from lumen.store.base import VectorStore
from lumen.store.sqlite import SQLiteVectorStore


@dataclass
class Container:
    settings: Settings
    provider: ChatProvider
    embedder: Embedder
    store: VectorStore
    prompts: PromptRegistry

    def close(self) -> None:
        self.provider.close()
        self.store.close()


def build_provider(settings: Settings) -> ChatProvider:
    if settings.provider == "openai":
        inner: ChatProvider = OpenAIProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            default_model=settings.model,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )
    elif settings.provider == "anthropic":
        inner = AnthropicProvider(
            api_key=settings.anthropic_api_key,
            base_url=settings.anthropic_base_url,
            default_model=settings.model,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )
    else:
        inner = EchoProvider(default_model=settings.model)
    provider: ChatProvider = RateLimitedProvider(
        inner, TokenBucket(settings.rate_limit_per_minute)
    )
    if settings.cache_enabled:
        provider = CachingProvider(provider, CompletionCache(settings.cache_path))
    return provider


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )
    return HashingEmbedder(dimensions=settings.embedding_dimensions)


def build_container(settings: Settings | None = None) -> Container:
    resolved = settings or get_settings()
    embedder = build_embedder(resolved)
    return Container(
        settings=resolved,
        provider=build_provider(resolved),
        embedder=embedder,
        store=SQLiteVectorStore(resolved.vector_db_path, embedder.dimensions),
        prompts=default_registry(),
    )
