from helpers import ScriptedProvider

from lumen.agent.loop import Agent
from lumen.models import Completion, Role, ToolCall, Usage
from lumen.tools.builtin import default_tools


def tool_turn(name: str, arguments: dict, call_id: str = "call-1", usage: Usage = Usage()) -> Completion:
    return Completion(
        text="",
        model="scripted",
        usage=usage,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
        finish_reason="tool_calls",
    )


def answer_turn(text: str, usage: Usage = Usage()) -> Completion:
    return Completion(text=text, model="scripted", usage=usage)


def test_agent_executes_tool_and_answers():
    provider = ScriptedProvider(
        [tool_turn("calculator", {"expression": "6 * 7"}), answer_turn("The answer is 42.")]
    )
    result = Agent(provider, default_tools()).run("what is 6 * 7?")
    assert result.answer == "The answer is 42."
    assert result.stop_reason == "answered"
    assert result.steps[0].tool == "calculator"
    assert result.steps[0].output == "42"
    assert provider.calls[1]["messages"][-1].role is Role.TOOL
    assert provider.calls[0]["tools"] == ["calculator", "current_time"]


def test_agent_stops_after_max_steps():
    provider = ScriptedProvider([tool_turn("current_time", {}), tool_turn("current_time", {})])
    result = Agent(provider, default_tools(), max_steps=2).run("loop forever")
    assert result.stop_reason == "max_steps"
    assert len(result.steps) == 2
    assert result.answer == ""


def test_agent_stops_when_token_budget_is_exhausted():
    provider = ScriptedProvider(
        [tool_turn("current_time", {}, usage=Usage(100, 100)), answer_turn("unused")]
    )
    result = Agent(provider, default_tools(), max_steps=5, token_budget=150).run("question")
    assert result.stop_reason == "token_budget"
    assert result.usage.total_tokens == 200
    assert len(provider.calls) == 1


def test_unknown_tool_is_fed_back_to_model():
    provider = ScriptedProvider([tool_turn("missing", {}), answer_turn("I could not use that tool.")])
    result = Agent(provider, default_tools()).run("try a missing tool")
    assert result.steps[0].output.startswith("error: unknown tool")
    assert result.answer == "I could not use that tool."


def test_usage_accumulates_across_steps():
    provider = ScriptedProvider(
        [
            tool_turn("calculator", {"expression": "1 + 1"}, usage=Usage(10, 5)),
            answer_turn("two", usage=Usage(20, 3)),
        ]
    )
    result = Agent(provider, default_tools()).run("1 + 1?")
    assert result.usage == Usage(30, 8)
