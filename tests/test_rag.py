from helpers import ScriptedProvider

from lumen.models import Completion, Document
from lumen.rag.pipeline import NO_ANSWER

DOCUMENTS = [
    Document(
        id="guide.md",
        text=(
            "The warranty lasts twenty four months from the purchase date. "
            "Returns are accepted within thirty days."
        ),
        metadata={"source": "guide.md"},
    ),
    Document(
        id="faq.md",
        text=(
            "Shipping is free for orders above fifty dollars. "
            "Express shipping takes two business days."
        ),
        metadata={"source": "faq.md"},
    ),
]


def test_ingest_then_ask_returns_grounded_answer_with_sources(make_service):
    service = make_service()
    report = service.ingest(DOCUMENTS)
    assert report.documents == 2
    assert report.chunks >= 2
    answer = service.ask("How long is the warranty?")
    assert answer.grounded
    assert answer.sources[0].chunk.document_id == "guide.md"
    assert answer.answer.startswith("Echo:")


def test_question_without_documents_is_refused_without_calling_provider(make_service):
    provider = ScriptedProvider([])
    answer = make_service(provider).ask("anything at all")
    assert answer.answer == NO_ANSWER
    assert not answer.grounded
    assert provider.calls == []


def test_reingesting_replaces_previous_chunks(make_service):
    service = make_service()
    first = service.ingest(DOCUMENTS)
    second = service.ingest(DOCUMENTS)
    assert second.replaced_chunks == first.chunks
    assert service.health()["chunks"] == first.chunks


def test_document_filter_limits_sources(make_service):
    service = make_service()
    service.ingest(DOCUMENTS)
    answer = service.ask("shipping", document_ids=frozenset({"guide.md"}))
    assert answer.sources
    assert {hit.chunk.document_id for hit in answer.sources} == {"guide.md"}


def test_prompt_contains_numbered_context_and_question(make_service):
    provider = ScriptedProvider([Completion(text="Twenty four months.", model="scripted")])
    service = make_service(provider)
    service.ingest(DOCUMENTS)
    answer = service.ask("How long is the warranty?")
    assert answer.answer == "Twenty four months."
    user_message = provider.calls[0]["messages"][-1].content
    assert "[1] " in user_message
    assert "Question: How long is the warranty?" in user_message
