from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lumen.embeddings.base import Embedder
from lumen.models import Document
from lumen.rag.chunking import chunk_document
from lumen.store.base import VectorStore


@dataclass(frozen=True)
class IngestReport:
    documents: int
    chunks: int
    replaced_chunks: int


class Ingestor:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int = 64,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._batch_size = batch_size

    def ingest(self, documents: Sequence[Document]) -> IngestReport:
        total_chunks = 0
        replaced = 0
        for document in documents:
            replaced += self._store.delete_document(document.id)
            chunks = chunk_document(document, self._chunk_size, self._chunk_overlap)
            for start in range(0, len(chunks), self._batch_size):
                batch = chunks[start : start + self._batch_size]
                vectors = self._embedder.embed([chunk.text for chunk in batch])
                self._store.add(batch, vectors)
            total_chunks += len(chunks)
        return IngestReport(
            documents=len(documents),
            chunks=total_chunks,
            replaced_chunks=replaced,
        )
