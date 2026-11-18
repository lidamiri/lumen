from __future__ import annotations

from pathlib import Path

from lumen.errors import InputError
from lumen.models import Document

SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".rst"})


def load_documents(path: str | Path, max_chars: int) -> list[Document]:
    root = Path(path)
    if not root.exists():
        raise InputError(f"path does not exist: {root}")
    if root.is_file():
        files = [root]
        base = root.parent
    else:
        files = sorted(
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
        )
        base = root
    documents: list[Document] = []
    for file in files:
        text = file.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        if len(text) > max_chars:
            raise InputError(f"{file} has {len(text)} characters, the limit is {max_chars}")
        relative = file.relative_to(base).as_posix()
        documents.append(Document(id=relative, text=text, metadata={"source": relative}))
    return documents
