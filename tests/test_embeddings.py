import numpy as np
import pytest

from lumen.embeddings.hashing import HashingEmbedder


def test_vectors_have_unit_length():
    vectors = HashingEmbedder(64).embed(["alpha beta", "gamma"])
    assert vectors.shape == (2, 64)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_related_texts_are_closer_than_unrelated_ones():
    embedder = HashingEmbedder(512)
    first, second, third = embedder.embed(
        ["the cat sat on the mat", "a cat sat on a mat", "quarterly revenue grew sharply"]
    )
    assert float(first @ second) > float(first @ third)


def test_embedding_is_deterministic():
    embedder = HashingEmbedder(128)
    assert np.array_equal(embedder.embed(["same text"]), embedder.embed(["same text"]))


def test_empty_input_yields_empty_matrix():
    assert HashingEmbedder(32).embed([]).shape == (0, 32)


def test_dimensions_below_minimum_are_rejected():
    with pytest.raises(ValueError):
        HashingEmbedder(4)
