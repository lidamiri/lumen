from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from lumen.errors import ToolError
from lumen.rag.retriever import Retriever
from lumen.tools.base import Tool
from lumen.tools.calculator import CalculatorTool
from lumen.tools.registry import ToolRegistry
from lumen.tools.retrieval import DocumentSearchTool


class CurrentTimeTool(Tool):
    name = "current_time"
    description = "Return the current date and time in an IANA timezone such as Europe/Berlin. Defaults to UTC."
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "IANA timezone name"},
        },
        "required": [],
    }

    def run(self, arguments: dict[str, Any]) -> str:
        name = arguments.get("timezone") or "UTC"
        if not isinstance(name, str):
            raise ToolError("timezone must be a string")
        try:
            zone = ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ToolError(f"unknown timezone: {name}") from exc
        return datetime.now(zone).isoformat(timespec="seconds")


def default_tools(retriever: Retriever | None = None) -> ToolRegistry:
    tools: list[Tool] = [CalculatorTool(), CurrentTimeTool()]
    if retriever is not None:
        tools.append(DocumentSearchTool(retriever))
    return ToolRegistry(tools)
