import numpy as np

from src.pipeline.ambedkargpt import AmbedkarGPTPipeline


# ------------------------------------------------------
# Dummy pipeline to avoid loading real files / models
# ------------------------------------------------------
class DummyPipeline(AmbedkarGPTPipeline):
    def __init__(self):
        
        pass


# ------------------------------------------------------
# Fake helper functions
# ------------------------------------------------------
def fake_retrieve(*args, **kwargs):
    """
    Simulates retrieval output.
    """
    return [(1, 0.95)]


def fake_generate(self, query, local_results, global_results):
    """
    Simulates LLM answer generation.
    """
    return "This is a fake generated answer."


# ------------------------------------------------------
# Integration test
# ------------------------------------------------------
def test_pipeline_runs_end_to_end():
    """
    Integration test:
    - Ensures AmbedkarGPTPipeline.run() works end-to-end
    - Uses mocked components
    - No real files, no heavy models
    """

    pipeline = DummyPipeline()

    # ---------------------------
    # Fake embedder
    # ---------------------------
    pipeline.embedder = type(
        "FakeEmbedder",
        (),
        {
            "encode": lambda self, query: np.array([0.1, 0.2, 0.3])
        },
    )()

    # ---------------------------
    # Fake local retriever
    # ---------------------------
    pipeline.local_retriever = type(
        "FakeLocalRetriever",
        (),
        {
            "retrieve": fake_retrieve
        },
    )()

    # ---------------------------
    # Fake global retriever
    # ---------------------------
    pipeline.global_retriever = type(
        "FakeGlobalRetriever",
        (),
        {
            "retrieve": fake_retrieve
        },
    )()

    # ---------------------------
    # Fake answer generator
    # ---------------------------
    pipeline.answer_generator = type(
        "FakeAnswerGenerator",
        (),
        {
            "generate": fake_generate
        },
    )()

    # ---------------------------
    # Run pipeline
    # ---------------------------
    result = pipeline.run("What did Ambedkar say about caste?")

    # ---------------------------
    # Assertions
    # ---------------------------
    assert isinstance(result, str)
    assert len(result) > 0
