from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx

from lumen.errors import ConfigurationError, ProviderError
from lumen.models import Completion, Message, ToolCall, ToolSpec, Usage
from lumen.providers.base import ChatProvider
from lumen.providers.http import post_json


class OpenAIProvider(ChatProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        default_model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for the openai provider")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Sequence[ToolSpec] = (),
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": [_to_wire(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        data = post_json(
            self._client,
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            payload=payload,
            max_retries=self._max_retries,
        )
        return _parse(data, payload["model"])

    def close(self) -> None:
        self._client.close()


def _to_wire(message: Message) -> dict[str, Any]:
    body: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.name and message.role.value == "tool":
        body["name"] = message.name
    if message.tool_call_id:
        body["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        body["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return body


def _parse(data: dict[str, Any], model: str) -> Completion:
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"unexpected response shape: {exc}") from exc
    tool_calls = tuple(
        ToolCall(
            id=call["id"],
            name=call["function"]["name"],
            arguments=_load_arguments(call["function"].get("arguments")),
        )
        for call in message.get("tool_calls") or []
    )
    usage = data.get("usage") or {}
    return Completion(
        text=message.get("content") or "",
        model=data.get("model", model),
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        ),
        tool_calls=tool_calls,
        finish_reason=choice.get("finish_reason") or "stop",
    )


def _load_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"invalid tool arguments: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ProviderError("tool arguments must be a JSON object")
    return value
