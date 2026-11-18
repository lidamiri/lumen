from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from lumen.errors import ConfigurationError, ProviderError
from lumen.models import Completion, Message, Role, ToolCall, ToolSpec, Usage
from lumen.providers.base import ChatProvider
from lumen.providers.http import post_json

API_VERSION = "2023-06-01"


class AnthropicProvider(ChatProvider):
    name = "anthropic"

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
            raise ConfigurationError("ANTHROPIC_API_KEY is required for the anthropic provider")
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
        system = "\n\n".join(message.content for message in messages if message.role is Role.SYSTEM)
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": _convert(messages),
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
                for tool in tools
            ]
        data = post_json(
            self._client,
            f"{self._base_url}/v1/messages",
            headers={"x-api-key": self._api_key, "anthropic-version": API_VERSION},
            payload=payload,
            max_retries=self._max_retries,
        )
        return _parse(data, payload["model"])

    def close(self) -> None:
        self._client.close()


def _convert(messages: Sequence[Message]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            continue
        if message.role is Role.TOOL:
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                        }
                    ],
                }
            )
        elif message.role is Role.ASSISTANT and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                for call in message.tool_calls
            )
            converted.append({"role": "assistant", "content": blocks})
        else:
            converted.append({"role": message.role.value, "content": message.content})
    return converted


def _parse(data: dict[str, Any], model: str) -> Completion:
    blocks = data.get("content")
    if not isinstance(blocks, list):
        raise ProviderError("unexpected response shape: missing content blocks")
    text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    tool_calls = tuple(
        ToolCall(id=block["id"], name=block["name"], arguments=dict(block.get("input") or {}))
        for block in blocks
        if block.get("type") == "tool_use"
    )
    usage = data.get("usage") or {}
    stop_reason = data.get("stop_reason") or "end_turn"
    return Completion(
        text=text,
        model=data.get("model", model),
        usage=Usage(
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
        ),
        tool_calls=tool_calls,
        finish_reason="tool_calls" if stop_reason == "tool_use" else stop_reason,
    )
