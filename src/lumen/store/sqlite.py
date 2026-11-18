from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lumen.errors import ConfigurationError
from lumen.models import Chunk, SearchHit
from lumen.store.base import VectorStore, rank, validate_batch

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SQLiteVectorStore(VectorStore):
    def __init__(self, path: str | Path, dimensions: int) -> None:
        self.dimensions = dimensions
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._check_dimensions()
            self._conn.commit()

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        validate_batch(chunks, vectors, self.dimensions)
        rows = [
            (
                chunk.id,
                chunk.document_id,
                chunk.index,
                chunk.text,
                json.dumps(chunk.metadata, ensure_ascii=False),
                vector.astype(np.float32).tobytes(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(id, document_id, chunk_index, text, metadata, vector) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 4,
        document_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, document_id, chunk_index, text, metadata, vector "
                "FROM chunks ORDER BY rowid"
            ).fetchall()
        chunks = [
            Chunk(
                id=row[0],
                document_id=row[1],
                index=row[2],
                text=row[3],
                metadata=json.loads(row[4]),
            )
            for row in rows
        ]
        if rows:
            matrix = np.vstack([np.frombuffer(row[5], dtype=np.float32) for row in rows])
        else:
            matrix = np.zeros((0, self.dimensions), dtype=np.float32)
        return rank(chunks, matrix, vector, top_k, document_ids)

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self._conn.commit()
            return cursor.rowcount

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _check_dimensions(self) -> None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'dimensions'").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('dimensions', ?)", (str(self.dimensions),)
            )
        elif int(row[0]) != self.dimensions:
            raise ConfigurationError(
                f"store at {self._path} holds {row[0]}-dimensional vectors, "
                f"but {self.dimensions} dimensions are configured"
            )
