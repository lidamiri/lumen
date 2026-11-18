from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lumen.models import Chunk, SearchHit
from lumen.store.base import VectorStore, rank, validate_batch


class InMemoryVectorStore(VectorStore):
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self._records: dict[str, tuple[Chunk, np.ndarray]] = {}
        self._snapshot: tuple[list[Chunk], np.ndarray] | None = None

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        validate_batch(chunks, vectors, self.dimensions)
        for chunk, vector in zip(chunks, vectors):
            self._records[chunk.id] = (chunk, vector.astype(np.float32))
        self._snapshot = None

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 4,
        document_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        chunks, matrix = self._materialize()
        return rank(chunks, matrix, vector, top_k, document_ids)

    def delete_document(self, document_id: str) -> int:
        stale = [key for key, (chunk, _) in self._records.items() if chunk.document_id == document_id]
        for key in stale:
            del self._records[key]
        if stale:
            self._snapshot = None
        return len(stale)

    def count(self) -> int:
        return len(self._records)

    def _materialize(self) -> tuple[list[Chunk], np.ndarray]:
        if self._snapshot is None:
            chunks = [chunk for chunk, _ in self._records.values()]
            if self._records:
                matrix = np.vstack([vector for _, vector in self._records.values()])
            else:
                matrix = np.zeros((0, self.dimensions), dtype=np.float32)
            self._snapshot = (chunks, matrix)
        return self._snapshot
