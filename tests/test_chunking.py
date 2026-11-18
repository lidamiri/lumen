import pytest

from lumen.models import Document
from lumen.rag.chunking import chunk_document, split_text


def test_short_text_is_single_chunk():
    assert split_text("Hello world.", size=50, overlap=5) == ["Hello world."]


def test_chunks_never_exceed_size():
    text = " ".join(f"Sentence number {index} is here." for index in range(200))
    chunks = split_text(text, size=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_consecutive_chunks_overlap():
    text = " ".join(f"word{index}." for index in range(300))
    chunks = split_text(text, size=100, overlap=30)
    assert len(chunks) > 2
    for previous, following in zip(chunks, chunks[1:]):
        assert following.startswith(previous[-30:].strip())


def test_long_word_is_hard_wrapped():
    assert split_text("a" * 250, size=100, overlap=0) == ["a" * 100, "a" * 100, "a" * 50]


def test_empty_text_produces_no_chunks():
    assert split_text("   \n\n  ", size=50, overlap=0) == []


def test_chunk_document_assigns_stable_ids():
    document = Document(id="guide.md", text="One. Two. Three.", metadata={"source": "guide.md"})
    chunks = chunk_document(document, size=8, overlap=0)
    assert [chunk.id for chunk in chunks] == [f"guide.md#{index}" for index in range(len(chunks))]
    assert all(chunk.document_id == "guide.md" for chunk in chunks)
    assert all(chunk.metadata == {"source": "guide.md"} for chunk in chunks)


@pytest.mark.parametrize("size, overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        split_text("some text", size=size, overlap=overlap)
