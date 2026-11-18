from datetime import datetime

import pytest

from lumen.embeddings.hashing import HashingEmbedder
from lumen.errors import ToolError
from lumen.models import Chunk, ToolCall
from lumen.rag.retriever import Retriever
from lumen.store.memory import InMemoryVectorStore
from lumen.tools.base import Tool
from lumen.tools.builtin import CurrentTimeTool, default_tools
from lumen.tools.calculator import CalculatorTool, evaluate
from lumen.tools.registry import ToolRegistry
from lumen.tools.retrieval import DocumentSearchTool


class FailingTool(Tool):
    name = "failing"
    description = "always fails"
    parameters = {"type": "object", "properties": {}}

    def run(self, arguments):
        raise RuntimeError("boom")


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2 + 3 * 4", 14),
        ("(1 + 2) ** 3", 27),
        ("-7 // 2", -4),
        ("sqrt(81)", 9),
        ("max(3, 9, 4)", 9),
        ("round(pi, 2)", 3.14),
        ("10 % 4", 2),
    ],
)
def test_calculator_evaluates_expressions(expression, expected):
    assert evaluate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "open('x')", "1 / 0", "2 ** 100000", "1 +", "True + 1", "[1, 2]"],
)
def test_calculator_rejects_unsafe_or_invalid_input(expression):
    with pytest.raises(ToolError):
        evaluate(expression)


def test_calculator_tool_formats_integers_without_decimal_point():
    assert CalculatorTool().run({"expression": "2 + 2"}) == "4"


def test_calculator_tool_requires_expression():
    with pytest.raises(ToolError):
        CalculatorTool().run({})


def test_current_time_tool_returns_iso_timestamp_in_zone():
    value = CurrentTimeTool().run({"timezone": "Europe/Berlin"})
    parsed = datetime.fromisoformat(value)
    assert parsed.utcoffset() is not None


def test_current_time_tool_rejects_unknown_zone():
    with pytest.raises(ToolError):
        CurrentTimeTool().run({"timezone": "Mars/Base"})


def test_registry_reports_unknown_tool_as_observation():
    registry = ToolRegistry([CalculatorTool()])
    output = registry.execute(ToolCall(id="1", name="missing", arguments={}))
    assert output.startswith("error: unknown tool")


def test_registry_converts_tool_errors_to_observations():
    registry = ToolRegistry([CalculatorTool(), FailingTool()])
    assert registry.execute(ToolCall(id="1", name="calculator", arguments={"expression": "1/0"})).startswith(
        "error: division by zero"
    )
    assert registry.execute(ToolCall(id="2", name="failing", arguments={})).startswith(
        "error: tool 'failing' failed"
    )


def test_registry_rejects_duplicate_names():
    with pytest.raises(ValueError):
        ToolRegistry([CalculatorTool(), CalculatorTool()])


def test_default_tools_include_search_only_with_retriever():
    assert "search_documents" not in default_tools()
    retriever = Retriever(HashingEmbedder(64), InMemoryVectorStore(64))
    assert "search_documents" in default_tools(retriever)


def test_document_search_tool_formats_sources():
    embedder = HashingEmbedder(64)
    store = InMemoryVectorStore(64)
    chunks = [Chunk(id="a#0", document_id="a", text="penguins live in antarctica", index=0)]
    store.add(chunks, embedder.embed([chunk.text for chunk in chunks]))
    tool = DocumentSearchTool(Retriever(embedder, store))
    output = tool.run({"query": "where do penguins live", "top_k": 99})
    assert "source=a" in output
    assert "penguins live in antarctica" in output


def test_document_search_tool_requires_query():
    tool = DocumentSearchTool(Retriever(HashingEmbedder(64), InMemoryVectorStore(64)))
    with pytest.raises(ToolError):
        tool.run({"query": "   "})
