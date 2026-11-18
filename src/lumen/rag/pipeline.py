from __future__ import annotations

from dataclasses import dataclass

from lumen.models import SearchHit, Usage
from lumen.prompts.template import PromptRegistry
from lumen.providers.base import ChatProvider
from lumen.rag.retriever import Retriever

NO_ANSWER = "I do not know based on the indexed documents."


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: tuple[SearchHit, ...]
    model: str
    usage: Usage
    grounded: bool


def build_context(hits: list[SearchHit], max_chars: int) -> tuple[str, list[SearchHit]]:
    blocks: list[str] = []
    used: list[SearchHit] = []
    size = 0
    for number, hit in enumerate(hits, start=1):
        block = f"[{number}] {hit.chunk.text}"
        if size + len(block) > max_chars:
            break
        blocks.append(block)
        used.append(hit)
        size += len(block)
    return "\n\n".join(blocks), used


class RagPipeline:
    def __init__(
        self,
        provider: ChatProvider,
        retriever: Retriever,
        prompts: PromptRegistry,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_context_chars: int = 6000,
        template_name: str = "rag_answer",
    ) -> None:
        self._provider = provider
        self._retriever = retriever
        self._prompts = prompts
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_context_chars = max_context_chars
        self._template_name = template_name

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: frozenset[str] | None = None,
    ) -> RagAnswer:
        hits = self._retriever.retrieve(question, top_k=top_k, document_ids=document_ids)
        if not hits:
            return RagAnswer(
                answer=NO_ANSWER,
                sources=(),
                model=self._model or "",
                usage=Usage(),
                grounded=False,
            )
        context, used = build_context(hits, self._max_context_chars)
        messages = self._prompts.get(self._template_name).build(context=context, question=question)
        completion = self._provider.complete(
            messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return RagAnswer(
            answer=completion.text.strip(),
            sources=tuple(used),
            model=completion.model,
            usage=completion.usage,
            grounded=True,
        )
