import json

import httpx
import pytest

from lumen.errors import ConfigurationError, ProviderError
from lumen.models import Message, Role, ToolCall, ToolSpec
from lumen.providers.anthropic_provider import AnthropicProvider
from lumen.providers.echo import EchoProvider
from lumen.providers.http import post_json
from lumen.providers.openai_provider import OpenAIProvider


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_request_and_response_mapping():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-x",
                "choices": [
                    {"message": {"role": "assistant", "content": "hi there"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    provider = OpenAIProvider(
        api_key="key",
        base_url="https://api.test/v1",
        default_model="gpt-x",
        client=mock_client(handler),
    )
    completion = provider.complete(
        [Message(Role.SYSTEM, "be kind"), Message(Role.USER, "hello")],
        temperature=0.2,
        max_tokens=50,
    )
    assert completion.text == "hi there"
    assert completion.usage.total_tokens == 7
    assert seen["auth"] == "Bearer key"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "be kind"}
    assert seen["body"]["max_tokens"] == 50


def test_openai_tool_calls_are_parsed_and_sent():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-x",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": "{\"expression\": \"2+2\"}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    provider = OpenAIProvider(api_key="key", base_url="https://api.test/v1", default_model="gpt-x", client=mock_client(handler))
    tools = [ToolSpec(name="calculator", description="calc", parameters={"type": "object"})]
    completion = provider.complete([Message(Role.USER, "2+2?")], tools=tools)
    assert seen["body"]["tools"][0]["function"]["name"] == "calculator"
    assert completion.tool_calls == (ToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"}),)
    assert completion.finish_reason == "tool_calls"


def test_openai_rejects_unexpected_response_shape():
    provider = OpenAIProvider(
        api_key="key",
        base_url="https://api.test/v1",
        default_model="gpt-x",
        client=mock_client(lambda request: httpx.Response(200, json={"unexpected": True})),
    )
    with pytest.raises(ProviderError):
        provider.complete([Message(Role.USER, "hi")])


def test_openai_requires_api_key():
    with pytest.raises(ConfigurationError):
        OpenAIProvider(api_key=None, base_url="https://api.test/v1", default_model="m")


def test_anthropic_converts_messages_and_tools():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["api_key"] = request.headers["x-api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "claude-x",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": "3*3"}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
        )

    provider = AnthropicProvider(
        api_key="anthropic-key",
        base_url="https://anthropic.test",
        default_model="claude-x",
        client=mock_client(handler),
    )
    messages = [
        Message(Role.SYSTEM, "system prompt"),
        Message(Role.USER, "nine?"),
        Message(Role.ASSISTANT, "", tool_calls=(ToolCall("prev", "calculator", {"expression": "1"}),)),
        Message(Role.TOOL, "1", name="calculator", tool_call_id="prev"),
    ]
    tools = [ToolSpec(name="calculator", description="calc", parameters={"type": "object"})]
    completion = provider.complete(messages, tools=tools)

    body = seen["body"]
    assert seen["api_key"] == "anthropic-key"
    assert body["system"] == "system prompt"
    assert all(item["role"] != "system" for item in body["messages"])
    assert body["messages"][1]["content"][0]["type"] == "tool_use"
    assert body["messages"][2]["content"][0]["type"] == "tool_result"
    assert body["tools"][0]["input_schema"] == {"type": "object"}
    assert completion.text == "Let me check."
    assert completion.tool_calls[0].arguments == {"expression": "3*3"}
    assert completion.finish_reason == "tool_calls"
    assert completion.usage.total_tokens == 14


def test_post_json_retries_transient_server_errors():
    responses = iter([httpx.Response(503, text="busy"), httpx.Response(200, json={"ok": True})])
    slept: list[float] = []
    result = post_json(
        mock_client(lambda request: next(responses)),
        "https://service.test",
        headers={},
        payload={},
        max_retries=2,
        sleep=slept.append,
    )
    assert result == {"ok": True}
    assert len(slept) == 1


def test_post_json_does_not_retry_client_errors():
    slept: list[float] = []
    with pytest.raises(ProviderError) as info:
        post_json(
            mock_client(lambda request: httpx.Response(400, text="bad request")),
            "https://service.test",
            headers={},
            payload={},
            max_retries=3,
            sleep=slept.append,
        )
    assert info.value.status_code == 400
    assert info.value.retryable is False
    assert slept == []


def test_post_json_gives_up_after_retry_budget():
    slept: list[float] = []
    with pytest.raises(ProviderError) as info:
        post_json(
            mock_client(lambda request: httpx.Response(503, text="down")),
            "https://service.test",
            headers={},
            payload={},
            max_retries=2,
            sleep=slept.append,
        )
    assert info.value.status_code == 503
    assert len(slept) == 2


def test_echo_provider_repeats_last_user_message():
    completion = EchoProvider().complete([Message(Role.USER, "ping")])
    assert completion.text == "Echo: ping"
    assert completion.usage.completion_tokens >= 1
