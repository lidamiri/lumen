from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumen.models import Message, Role, Usage
from lumen.providers.base import ChatProvider
from lumen.tools.registry import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. Use the available tools when they help you answer accurately. "
    "Never invent tool results. When you have enough information, answer the user directly."
)
MAX_OBSERVATION_CHARS = 2000


@dataclass(frozen=True)
class AgentStep:
    tool: str
    arguments: dict[str, Any]
    output: str


@dataclass(frozen=True)
class AgentResult:
    answer: str
    steps: tuple[AgentStep, ...]
    usage: Usage
    stop_reason: str


class Agent:
    def __init__(
        self,
        provider: ChatProvider,
        tools: ToolRegistry,
        *,
        model: str | None = None,
        max_steps: int = 6,
        token_budget: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._provider = provider
        self._tools = tools
        self._model = model
        self._max_steps = max_steps
        self._token_budget = token_budget
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

    def run(self, question: str) -> AgentResult:
        messages = [
            Message(role=Role.SYSTEM, content=self._system_prompt),
            Message(role=Role.USER, content=question),
        ]
        steps: list[AgentStep] = []
        usage = Usage()
        for _ in range(self._max_steps):
            completion = self._provider.complete(
                messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                tools=self._tools.specs(),
            )
            usage = usage + completion.usage
            if not completion.tool_calls:
                return AgentResult(
                    answer=completion.text.strip(),
                    steps=tuple(steps),
                    usage=usage,
                    stop_reason="answered",
                )
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=completion.text,
                    tool_calls=completion.tool_calls,
                )
            )
            for call in completion.tool_calls:
                output = self._tools.execute(call)
                steps.append(AgentStep(tool=call.name, arguments=dict(call.arguments), output=output))
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=_truncate(output),
                        name=call.name,
                        tool_call_id=call.id,
                    )
                )
            if self._token_budget is not None and usage.total_tokens >= self._token_budget:
                return AgentResult("", tuple(steps), usage, "token_budget")
        return AgentResult("", tuple(steps), usage, "max_steps")


def _truncate(text: str) -> str:
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    return text[:MAX_OBSERVATION_CHARS] + "...[truncated]"
