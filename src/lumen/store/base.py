from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from lumen.models import Chunk, SearchHit


class VectorStore(ABC):
    dimensions: int

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        vector: np.ndarray,
        top_k: int = 4,
        document_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        return None


def validate_batch(chunks: Sequence[Chunk], vectors: np.ndarray, dimensions: int) -> None:
    if vectors.ndim != 2 or vectors.shape != (len(chunks), dimensions):
        raise ValueError(
            f"expected vectors of shape ({len(chunks)}, {dimensions}), got {vectors.shape}"
        )


def rank(
    chunks: Sequence[Chunk],
    matrix: np.ndarray,
    vector: np.ndarray,
    top_k: int,
    document_ids: frozenset[str] | None,
) -> list[SearchHit]:
    if not chunks or top_k <= 0:
        return []
    scores = matrix @ vector.astype(np.float32)
    hits: list[SearchHit] = []
    for index in np.argsort(-scores, kind="stable"):
        chunk = chunks[int(index)]
        if document_ids is not None and chunk.document_id not in document_ids:
            continue
        hits.append(SearchHit(chunk=chunk, score=float(scores[index])))
        if len(hits) >= top_k:
            break
    return hits
