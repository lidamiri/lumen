from __future__ import annotations

from typing import Any

from lumen.errors import ToolError
from lumen.rag.retriever import Retriever
from lumen.tools.base import Tool

MAX_TOP_K = 10


class DocumentSearchTool(Tool):
    name = "search_documents"
    description = "Search the indexed document collection and return the most relevant passages with their sources."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": MAX_TOP_K},
        },
        "required": ["query"],
    }

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def run(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        top_k = _parse_top_k(arguments.get("top_k"))
        hits = self._retriever.retrieve(query, top_k=top_k)
        if not hits:
            return "no matching passages"
        return "\n\n".join(
            f"[{number}] source={hit.chunk.document_id} score={hit.score:.3f}\n{hit.chunk.text}"
            for number, hit in enumerate(hits, start=1)
        )


def _parse_top_k(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ToolError("top_k must be an integer") from exc
    return min(max(value, 1), MAX_TOP_K)
