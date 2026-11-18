from __future__ import annotations

from collections.abc import Iterable

from lumen.logging_utils import get_logger
from lumen.errors import ToolError
from lumen.models import ToolCall, ToolSpec
from lumen.tools.base import Tool

logger = get_logger("tools")


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def execute(self, call: ToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return f"error: unknown tool {call.name!r}"
        try:
            return tool.run(call.arguments)
        except ToolError as exc:
            return f"error: {exc}"
        except Exception as exc:
            logger.exception("tool %s failed", call.name)
            return f"error: tool {call.name!r} failed: {exc}"
