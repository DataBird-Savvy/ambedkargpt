import numpy as np
import networkx as nx
import pytest

from src.retrieval.global_search import GlobalGraphRAGRetriever
from src.retrieval.local_search import LocalGraphRAGRetriever
from src.retrieval.ranker import ChunkReranker


# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture
def query_embedding():
    return np.array([1.0, 0.0, 0.0])


@pytest.fixture
def community_vectors():
    return {
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([0.0, 1.0, 0.0]),
    }


@pytest.fixture
def community_chunks():
    return {
        1: [101, 102],
        2: [201],
    }


@pytest.fixture
def chunk_embeddings():
    return {
        101: np.array([1.0, 0.0, 0.0]),
        102: np.array([0.9, 0.1, 0.0]),
        201: np.array([0.0, 1.0, 0.0]),
    }


# ---------------------------
# GlobalGraphRAGRetriever tests
# ---------------------------

def test_global_retriever_returns_results(
    query_embedding,
    community_vectors,
    community_chunks,
    chunk_embeddings,
):
    retriever = GlobalGraphRAGRetriever(
        community_vectors=community_vectors,
        community_chunks=community_chunks,
        chunk_embeddings=chunk_embeddings,
        top_k_communities=1,
        top_k_chunks=2,
    )

    results = retriever.retrieve(query_embedding)

    assert isinstance(results, list)
    assert len(results) > 0
    assert isinstance(results[0], tuple)


def test_global_retriever_ranks_correct_chunk(
    query_embedding,
    community_vectors,
    community_chunks,
    chunk_embeddings,
):
    retriever = GlobalGraphRAGRetriever(
        community_vectors=community_vectors,
        community_chunks=community_chunks,
        chunk_embeddings=chunk_embeddings,
        top_k_communities=1,
        top_k_chunks=1,
    )

    results = retriever.retrieve(query_embedding)

    # Chunk 101 is most similar
    assert results[0][0] == 101


# ---------------------------
# LocalGraphRAGRetriever tests
# ---------------------------

@pytest.fixture
def simple_graph():
    G = nx.Graph()
    G.add_node("Ambedkar", chunks=[101, 102])
    G.add_node("Weather", chunks=[201])
    return G


@pytest.fixture
def entity_embeddings():
    return {
        "Ambedkar": np.array([1.0, 0.0, 0.0]),
        "Weather": np.array([0.0, 1.0, 0.0]),
    }


def test_local_retriever_filters_entities(
    simple_graph,
    entity_embeddings,
    chunk_embeddings,
    query_embedding,
):
    retriever = LocalGraphRAGRetriever(
        graph=simple_graph,
        entity_embeddings=entity_embeddings,
        chunk_embeddings=chunk_embeddings,
        tau_e=0.8,
        tau_d=0.5,
        top_k=3,
    )

    results = retriever.retrieve(query_embedding)

    assert len(results) > 0
    assert all(chunk_id in [101, 102] for chunk_id, _ in results)


def test_local_retriever_returns_empty_when_no_entity_matches(
    simple_graph,
    entity_embeddings,
    chunk_embeddings,
):
    retriever = LocalGraphRAGRetriever(
        graph=simple_graph,
        entity_embeddings=entity_embeddings,
        chunk_embeddings=chunk_embeddings,
        tau_e=0.99,   # too strict
        tau_d=0.99,
        top_k=3,
    )

    query = np.array([0.0, 0.0, 1.0])
    results = retriever.retrieve(query)

    assert results == []


# ---------------------------
# ChunkReranker tests
# ---------------------------

def test_chunk_reranker_merges_and_sorts():
    chunk_texts = {
        101: "Ambedkar on caste",
        201: "Weather information",
    }

    local = [(101, 0.8)]
    global_ = [(201, 0.5)]

    reranker = ChunkReranker(chunk_texts)
    results = reranker.rerank(local, global_, top_k=2)

    assert len(results) == 2
    assert results[0][0] == 101   # highest score first
    assert results[0][2] in ["Local", "Global"]


def test_chunk_reranker_empty_input():
    reranker = ChunkReranker({})
    results = reranker.rerank([], [], top_k=5)

    assert results == []
