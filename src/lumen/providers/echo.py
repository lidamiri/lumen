from __future__ import annotations

from collections.abc import Sequence

from lumen.models import Completion, Message, Role, ToolSpec, Usage
from lumen.providers.base import ChatProvider
from lumen.tokens import estimate_tokens


class EchoProvider(ChatProvider):
    name = "echo"

    def __init__(self, default_model: str = "echo-1") -> None:
        self._default_model = default_model

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Sequence[ToolSpec] = (),
    ) -> Completion:
        last_user = next(
            (message.content for message in reversed(messages) if message.role is Role.USER),
            "",
        )
        text = f"Echo: {last_user}"
        return Completion(
            text=text,
            model=model or self._default_model,
            usage=Usage(
                prompt_tokens=sum(estimate_tokens(message.content) for message in messages),
                completion_tokens=estimate_tokens(text),
            ),
        )
