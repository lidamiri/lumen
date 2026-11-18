from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from lumen.errors import ProviderError
from lumen.logging_utils import get_logger

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

logger = get_logger("providers.http")


def post_json(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    max_retries: int = 3,
    backoff: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    last_error: ProviderError | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.post(url, headers=headers, json=payload)
        except httpx.TransportError as exc:
            last_error = ProviderError(f"transport error: {exc}", retryable=True)
        else:
            if response.status_code < 400:
                try:
                    return response.json()
                except ValueError as exc:
                    raise ProviderError("response body is not valid JSON") from exc
            last_error = ProviderError(
                f"HTTP {response.status_code}: {response.text[:500]}",
                status_code=response.status_code,
                retryable=response.status_code in RETRYABLE_STATUS,
            )
            if not last_error.retryable:
                raise last_error
        if attempt < max_retries:
            delay = backoff * (2**attempt) + random.uniform(0, backoff)
            logger.warning("request failed (%s), retrying in %.2fs", last_error, delay)
            sleep(delay)
    assert last_error is not None
    raise last_error
