from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from lumen.embeddings.base import Embedder, normalize_rows

_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder(Embedder):
    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in _features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign
        return normalize_rows(matrix)


def _features(text: str) -> list[str]:
    tokens = _WORD.findall(text.lower())
    features = list(tokens)
    features.extend(f"{left} {right}" for left, right in zip(tokens, tokens[1:]))
    return features
