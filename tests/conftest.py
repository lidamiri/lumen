from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from lumen.config import Settings
from lumen.container import Container
from lumen.embeddings.hashing import HashingEmbedder
from lumen.prompts.library import default_registry
from lumen.providers.base import ChatProvider
from lumen.providers.echo import EchoProvider
from lumen.service import LumenService
from lumen.store.sqlite import SQLiteVectorStore


@pytest.fixture
def make_container(tmp_path: Path) -> Iterator[Callable[..., Container]]:
    created: list[Container] = []

    def _make(provider: ChatProvider | None = None, **overrides: Any) -> Container:
        values: dict[str, Any] = {
            "provider": "echo",
            "data_dir": tmp_path / "data",
            "cache_enabled": False,
            "embedding_dimensions": 256,
            "chunk_size": 200,
            "chunk_overlap": 20,
            "top_k": 3,
        }
        values.update(overrides)
        settings = Settings(_env_file=None, **values)
        embedder = HashingEmbedder(settings.embedding_dimensions)
        store = SQLiteVectorStore(settings.vector_db_path, embedder.dimensions)
        container = Container(
            settings=settings,
            provider=provider or EchoProvider(),
            embedder=embedder,
            store=store,
            prompts=default_registry(),
        )
        created.append(container)
        return container

    yield _make
    for container in created:
        container.close()


@pytest.fixture
def make_service(make_container: Callable[..., Container]) -> Callable[..., LumenService]:
    def _make(provider: ChatProvider | None = None, **overrides: Any) -> LumenService:
        return LumenService(make_container(provider, **overrides))

    return _make
