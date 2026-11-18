from collections.abc import Sequence
from typing import Any

from lumen.models import Completion, Message, ToolSpec
from lumen.providers.base import ChatProvider


class ScriptedProvider(ChatProvider):
    name = "scripted"

    def __init__(self, responses: Sequence[Completion]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Sequence[ToolSpec] = (),
    ) -> Completion:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": [tool.name for tool in tools],
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise AssertionError("no scripted response left")
        return self._responses.pop(0)
