from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatRequest(BaseModel):
    messages: list[MessageIn] = Field(min_length=1, max_length=100)
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


class ChatResponse(BaseModel):
    text: str
    model: str
    finish_reason: str
    cached: bool
    usage: UsageOut


class DocumentIn(BaseModel):
    id: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=200)


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    replaced_chunks: int


class DeleteResponse(BaseModel):
    document_id: str
    removed_chunks: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] = Field(default_factory=list, max_length=100)


class SourceOut(BaseModel):
    document_id: str
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    model: str
    sources: list[SourceOut]
    usage: UsageOut


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    max_steps: int | None = Field(default=None, ge=1, le=20)


class StepOut(BaseModel):
    tool: str
    arguments: dict[str, Any]
    output: str


class AgentResponse(BaseModel):
    answer: str
    stop_reason: str
    steps: list[StepOut]
    usage: UsageOut


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    max_sentences: int = Field(default=3, ge=1, le=20)


class SummarizeResponse(BaseModel):
    summary: str


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    labels: list[str] = Field(min_length=1, max_length=50)


class ClassifyResponse(BaseModel):
    label: str
