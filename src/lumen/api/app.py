from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from lumen import __version__
from lumen.api.schemas import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    ClassifyRequest,
    ClassifyResponse,
    DeleteResponse,
    IngestRequest,
    IngestResponse,
    SourceOut,
    StepOut,
    SummarizeRequest,
    SummarizeResponse,
    UsageOut,
)
from lumen.api.security import authorize
from lumen.container import Container, build_container
from lumen.errors import ConfigurationError, InputError, NotFoundError, ProviderError
from lumen.models import Document, Message, Role, Usage
from lumen.service import LumenService


def create_app(container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        active = container or build_container()
        app.state.container = active
        app.state.service = LumenService(active)
        try:
            yield
        finally:
            if owned:
                active.close()

    app = FastAPI(title="Lumen", version=__version__, lifespan=lifespan)
    _register_error_handlers(app)

    router = APIRouter(prefix="/v1", dependencies=[Depends(authorize)])

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return request.app.state.service.health()

    @router.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        completion = request.app.state.service.chat(
            [Message(role=Role(item.role), content=item.content) for item in payload.messages],
            model=payload.model,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
        return ChatResponse(
            text=completion.text,
            model=completion.model,
            finish_reason=completion.finish_reason,
            cached=completion.cached,
            usage=_usage(completion.usage),
        )

    @router.post("/documents", response_model=IngestResponse)
    def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
        report = request.app.state.service.ingest(
            [Document(id=item.id, text=item.text, metadata=item.metadata) for item in payload.documents]
        )
        return IngestResponse(
            documents=report.documents,
            chunks=report.chunks,
            replaced_chunks=report.replaced_chunks,
        )

    @router.delete("/documents/{document_id:path}", response_model=DeleteResponse)
    def delete_document(document_id: str, request: Request) -> DeleteResponse:
        removed = request.app.state.service.delete_document(document_id)
        if removed == 0:
            raise NotFoundError(f"no indexed chunks for document {document_id!r}")
        return DeleteResponse(document_id=document_id, removed_chunks=removed)

    @router.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest, request: Request) -> AskResponse:
        answer = request.app.state.service.ask(
            payload.question,
            top_k=payload.top_k,
            document_ids=frozenset(payload.document_ids) if payload.document_ids else None,
        )
        return AskResponse(
            answer=answer.answer,
            grounded=answer.grounded,
            model=answer.model,
            sources=[
                SourceOut(
                    document_id=hit.chunk.document_id,
                    chunk_id=hit.chunk.id,
                    score=round(hit.score, 4),
                    text=hit.chunk.text,
                )
                for hit in answer.sources
            ],
            usage=_usage(answer.usage),
        )

    @router.post("/agent", response_model=AgentResponse)
    def agent(payload: AgentRequest, request: Request) -> AgentResponse:
        result = request.app.state.service.run_agent(payload.question, max_steps=payload.max_steps)
        return AgentResponse(
            answer=result.answer,
            stop_reason=result.stop_reason,
            steps=[StepOut(tool=step.tool, arguments=step.arguments, output=step.output) for step in result.steps],
            usage=_usage(result.usage),
        )

    @router.post("/summaries", response_model=SummarizeResponse)
    def summarize(payload: SummarizeRequest, request: Request) -> SummarizeResponse:
        summary = request.app.state.service.summarize(payload.text, max_sentences=payload.max_sentences)
        return SummarizeResponse(summary=summary)

    @router.post("/classifications", response_model=ClassifyResponse)
    def classify(payload: ClassifyRequest, request: Request) -> ClassifyResponse:
        label = request.app.state.service.classify(payload.text, payload.labels)
        return ClassifyResponse(label=label)

    app.include_router(router)
    return app


def _usage(usage: Usage) -> UsageOut:
    return UsageOut(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(InputError)
    async def _input_error(request: Request, exc: InputError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(NotFoundError)
    async def _not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def _provider_error(request: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ConfigurationError)
    async def _configuration_error(request: Request, exc: ConfigurationError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
