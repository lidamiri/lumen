from __future__ import annotations

from collections.abc import Sequence

import httpx
import numpy as np

from lumen.embeddings.base import Embedder, normalize_rows
from lumen.errors import ConfigurationError, ProviderError
from lumen.providers.http import post_json


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        dimensions: int,
        timeout: float = 60.0,
        max_retries: int = 3,
        batch_size: int = 64,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for the openai embedder")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._client = client or httpx.Client(timeout=timeout)
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            data = post_json(
                self._client,
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                payload={"model": self._model, "input": batch},
                max_retries=self._max_retries,
            )
            try:
                items = sorted(data["data"], key=lambda item: item["index"])
                vectors.extend(item["embedding"] for item in items)
            except (KeyError, TypeError) as exc:
                raise ProviderError(f"unexpected embedding response: {exc}") from exc
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape[1] != self.dimensions:
            raise ProviderError(
                f"embedding model returned {matrix.shape[1]} dimensions, expected {self.dimensions}"
            )
        return normalize_rows(matrix)
