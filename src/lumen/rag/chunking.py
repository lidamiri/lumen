from __future__ import annotations

import re

from lumen.models import Chunk, Document

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    pieces: list[str] = []
    for unit in _units(text):
        pieces.extend(_hard_wrap(unit, size))
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        joined = f"{current} {piece}".strip()
        if len(joined) <= size:
            current = joined
            continue
        if current:
            chunks.append(current)
        tail = current[-overlap:].strip() if overlap and current else ""
        candidate = f"{tail} {piece}".strip()
        current = candidate if len(candidate) <= size else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_document(document: Document, size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(
            id=f"{document.id}#{index}",
            document_id=document.id,
            text=text,
            index=index,
            metadata=dict(document.metadata),
        )
        for index, text in enumerate(split_text(document.text, size, overlap))
    ]


def _units(text: str) -> list[str]:
    return [unit.strip() for unit in _BOUNDARY.split(text) if unit and unit.strip()]


def _hard_wrap(unit: str, size: int) -> list[str]:
    if len(unit) <= size:
        return [unit]
    wrapped: list[str] = []
    line = ""
    for word in unit.split():
        if len(word) > size:
            if line:
                wrapped.append(line)
                line = ""
            wrapped.extend(word[start : start + size] for start in range(0, len(word), size))
            continue
        candidate = f"{line} {word}".strip()
        if len(candidate) <= size:
            line = candidate
        else:
            wrapped.append(line)
            line = word
    if line:
        wrapped.append(line)
    return wrapped
