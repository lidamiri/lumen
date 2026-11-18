from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lumen.models import Completion, Message, ToolCall, ToolSpec, Usage


def completion_key(
    provider: str,
    model: str | None,
    messages: Sequence[Message],
    temperature: float,
    max_tokens: int,
    tools: Sequence[ToolSpec],
) -> str:
    material = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [_encode_message(message) for message in messages],
        "tools": [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in tools
        ],
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


class CompletionCache:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS completions ("
                "key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            self._conn.commit()

    def get(self, key: str) -> Completion | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM completions WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return _decode(json.loads(row[0]))

    def put(self, key: str, completion: Completion) -> None:
        payload = json.dumps(_encode(completion), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO completions (key, payload, created_at) VALUES (?, ?, ?)",
                (key, payload, time.time()),
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _encode_message(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": message.content,
        "name": message.name,
        "tool_call_id": message.tool_call_id,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ],
    }


def _encode(completion: Completion) -> dict[str, Any]:
    return {
        "text": completion.text,
        "model": completion.model,
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in completion.tool_calls
        ],
        "finish_reason": completion.finish_reason,
    }


def _decode(payload: dict[str, Any]) -> Completion:
    return Completion(
        text=payload["text"],
        model=payload["model"],
        usage=Usage(
            prompt_tokens=payload["prompt_tokens"],
            completion_tokens=payload["completion_tokens"],
        ),
        tool_calls=tuple(
            ToolCall(id=call["id"], name=call["name"], arguments=call["arguments"])
            for call in payload["tool_calls"]
        ),
        finish_reason=payload["finish_reason"],
    )
