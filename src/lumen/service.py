from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lumen.agent.loop import Agent, AgentResult
from lumen.config import Settings
from lumen.container import Container
from lumen.errors import InputError
from lumen.models import Completion, Document, Message
from lumen.rag.ingest import IngestReport, Ingestor
from lumen.rag.pipeline import RagAnswer, RagPipeline
from lumen.rag.retriever import Retriever
from lumen.tools.builtin import default_tools


class LumenService:
    def __init__(self, container: Container) -> None:
        self._container = container
        self._settings = container.settings
        self._retriever = Retriever(
            container.embedder,
            container.store,
            top_k=self._settings.top_k,
        )
        self._ingestor = Ingestor(
            container.embedder,
            container.store,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        self._pipeline = RagPipeline(
            container.provider,
            self._retriever,
            container.prompts,
            model=self._settings.model,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
            max_context_chars=self._settings.max_context_chars,
        )
        self._tools = default_tools(self._retriever)

    @property
    def settings(self) -> Settings:
        return self._settings

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        self._check_size(sum(len(message.content) for message in messages))
        return self._container.provider.complete(
            messages,
            model=model or self._settings.model,
            temperature=self._settings.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self._settings.max_tokens,
        )

    def ingest(self, documents: Sequence[Document]) -> IngestReport:
        self._check_size(sum(len(document.text) for document in documents))
        return self._ingestor.ingest(documents)

    def delete_document(self, document_id: str) -> int:
        return self._container.store.delete_document(document_id)

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_ids: frozenset[str] | None = None,
    ) -> RagAnswer:
        self._check_size(len(question))
        return self._pipeline.answer(question, top_k=top_k, document_ids=document_ids)

    def run_agent(self, question: str, *, max_steps: int | None = None) -> AgentResult:
        self._check_size(len(question))
        agent = Agent(
            self._container.provider,
            self._tools,
            model=self._settings.model,
            max_steps=max_steps or self._settings.agent_max_steps,
            token_budget=self._settings.agent_token_budget,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
        )
        return agent.run(question)

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        self._check_size(len(text))
        messages = self._container.prompts.get("summarize").build(
            text=text,
            max_sentences=str(max_sentences),
        )
        completion = self._container.provider.complete(
            messages,
            model=self._settings.model,
            temperature=0.0,
            max_tokens=self._settings.max_tokens,
        )
        return completion.text.strip()

    def classify(self, text: str, labels: Sequence[str]) -> str:
        cleaned = [label.strip() for label in labels if label.strip()]
        if not cleaned:
            raise InputError("at least one label is required")
        self._check_size(len(text))
        messages = self._container.prompts.get("classify").build(
            text=text,
            labels=", ".join(cleaned),
        )
        completion = self._container.provider.complete(
            messages,
            model=self._settings.model,
            temperature=0.0,
            max_tokens=32,
        )
        return completion.text.strip()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "provider": self._container.provider.name,
            "embedding_dimensions": self._container.embedder.dimensions,
            "chunks": self._container.store.count(),
            "tools": self._tools.names(),
        }

    def _check_size(self, size: int) -> None:
        limit = self._settings.max_input_chars
        if size > limit:
            raise InputError(f"input has {size} characters, the limit is {limit}")
