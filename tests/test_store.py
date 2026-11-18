import numpy as np
import pytest

from lumen.errors import ConfigurationError
from lumen.embeddings.hashing import HashingEmbedder
from lumen.models import Chunk
from lumen.store.memory import InMemoryVectorStore
from lumen.store.sqlite import SQLiteVectorStore


def make_chunks(document_id: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(id=f"{document_id}#{index}", document_id=document_id, text=text, index=index)
        for index, text in enumerate(texts)
    ]


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        instance = InMemoryVectorStore(64)
    else:
        instance = SQLiteVectorStore(tmp_path / "vectors.sqlite3", 64)
    yield instance
    instance.close()


def test_search_ranks_best_match_first(store):
    embedder = HashingEmbedder(64)
    chunks = make_chunks(
        "doc",
        [
            "python generators and iterators",
            "baking sourdough bread at home",
            "kubernetes pod scheduling",
        ],
    )
    store.add(chunks, embedder.embed([chunk.text for chunk in chunks]))
    hits = store.search(embedder.embed_one("how do python iterators work"), top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk.id == "doc#0"
    assert hits[0].score >= hits[1].score


def test_upsert_replaces_existing_chunk(store):
    embedder = HashingEmbedder(64)
    chunks = make_chunks("doc", ["first version of the text"])
    store.add(chunks, embedder.embed([chunks[0].text]))
    store.add(chunks, embedder.embed(["second version of the text"]))
    assert store.count() == 1


def test_delete_document_removes_only_its_chunks(store):
    embedder = HashingEmbedder(64)
    first = make_chunks("a", ["alpha text", "more alpha"])
    second = make_chunks("b", ["beta text"])
    store.add(first, embedder.embed([chunk.text for chunk in first]))
    store.add(second, embedder.embed([chunk.text for chunk in second]))
    assert store.delete_document("a") == 2
    assert store.count() == 1
    assert store.delete_document("missing") == 0


def test_document_filter_restricts_results(store):
    embedder = HashingEmbedder(64)
    first = make_chunks("a", ["shared topic words"])
    second = make_chunks("b", ["shared topic words"])
    store.add(first + second, embedder.embed([c.text for c in first + second]))
    hits = store.search(embedder.embed_one("shared topic"), top_k=5, document_ids=frozenset({"b"}))
    assert [hit.chunk.document_id for hit in hits] == ["b"]


def test_empty_store_returns_no_hits(store):
    assert store.search(np.zeros(64, dtype=np.float32), top_k=3) == []


def test_mismatched_vector_shape_is_rejected(store):
    chunks = make_chunks("doc", ["text"])
    with pytest.raises(ValueError):
        store.add(chunks, np.zeros((1, 32), dtype=np.float32))


def test_sqlite_store_persists_between_connections(tmp_path):
    path = tmp_path / "persist.sqlite3"
    embedder = HashingEmbedder(64)
    chunks = make_chunks("doc", ["persistent memory"])
    first = SQLiteVectorStore(path, 64)
    first.add(chunks, embedder.embed([chunks[0].text]))
    first.close()
    second = SQLiteVectorStore(path, 64)
    assert second.count() == 1
    second.close()


def test_sqlite_store_rejects_dimension_change(tmp_path):
    path = tmp_path / "dims.sqlite3"
    SQLiteVectorStore(path, 64).close()
    with pytest.raises(ConfigurationError):
        SQLiteVectorStore(path, 32)
