import pytest

from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.buffer_merger import SemanticChunkMerger


# ---------------------------
# Fixtures (reusable setup)
# ---------------------------

@pytest.fixture
def embedding_model():
    return "all-MiniLM-L6-v2"


@pytest.fixture
def sample_text():
    return (
        "Dr. Ambedkar was a social reformer. "
        "He fought against caste discrimination. "
        "The sun is very hot today. "
        "It is a bright summer day."
    )


@pytest.fixture
def semantic_chunker(embedding_model):
    return SemanticChunker(
        embedding_model=embedding_model,
        sim_threshold=0.6
    )


@pytest.fixture
def semantic_chunk_merger(embedding_model):
    return SemanticChunkMerger(
        max_tokens=50,
        subchunk_size=20,
        overlap=5,
        embedding_model=embedding_model
    )


# ---------------------------
# Tests for SemanticChunker
# ---------------------------

def test_chunker_returns_list(semantic_chunker, sample_text):
    chunks = semantic_chunker.chunk(sample_text)
    assert isinstance(chunks, list)


def test_chunker_non_empty_output(semantic_chunker, sample_text):
    chunks = semantic_chunker.chunk(sample_text)
    assert len(chunks) > 0


def test_chunker_splits_semantically(semantic_chunker, sample_text):
    chunks = semantic_chunker.chunk(sample_text)

    # Ambedkar sentences should be together
    assert any("Ambedkar" in chunk for chunk in chunks)

    # Weather sentences should be together
    assert any("sun" in chunk or "summer" in chunk for chunk in chunks)


# ---------------------------
# Tests for SemanticChunkMerger
# ---------------------------

def test_merger_returns_chunks_and_embeddings(semantic_chunk_merger):
    chunks = [
        "This is the first chunk.",
        "This is the second chunk."
    ]

    final_chunks, embeddings = semantic_chunk_merger.merge_and_embed(chunks)

    assert len(final_chunks) == len(embeddings)


def test_merger_respects_token_limit(semantic_chunk_merger):
    long_chunk = "word " * 200
    chunks = [long_chunk]

    final_chunks = semantic_chunk_merger.merge_chunks(chunks)

    # Should split into smaller sub-chunks
    assert len(final_chunks) > 1


def test_merger_buffer_added():
    merger = SemanticChunkMerger(
        max_tokens=100,
        subchunk_size=50,
        overlap=10,
        embedding_model="all-MiniLM-L6-v2"
    )

    chunks = [
        "Sentence one. Sentence two.",
        "Sentence three. Sentence four."
    ]

    merged = merger.merge_chunks(chunks)

    # Buffer from previous chunk should appear
    assert "Sentence two" in merged[1]
