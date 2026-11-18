from __future__ import annotations

from lumen.embeddings.base import Embedder
from lumen.models import SearchHit
from lumen.store.base import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 4) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        if not query.strip():
            return []
        vector = self._embedder.embed_one(query)
        return self._store.search(
            vector,
            top_k=top_k or self._top_k,
            document_ids=document_ids,
        )
