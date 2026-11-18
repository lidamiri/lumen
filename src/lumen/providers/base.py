from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from lumen.models import Completion, Message, ToolSpec


class ChatProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tools: Sequence[ToolSpec] = (),
    ) -> Completion:
        raise NotImplementedError

    def close(self) -> None:
        return None
