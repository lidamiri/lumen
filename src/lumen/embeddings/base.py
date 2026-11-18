from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Embedder(ABC):
    dimensions: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
