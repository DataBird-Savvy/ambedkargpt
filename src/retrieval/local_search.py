import networkx as nx
import numpy as np
from retrieval.ranker import ChunkReranker
from logger import logger
from sklearn.metrics.pairwise import cosine_similarity
from src.exception import RAGException
import sys


class LocalGraphRAGRetriever:
    """
    Implements Local Graph RAG Retrieval (SEMRAG Equation 4)

    D_retrieved = Top_k({
        v ∈ V, g ∈ G |
        sim(v, Q+H) > τ_e AND sim(g, Q) > τ_d
    })
    """

    def __init__(
        self,
        graph: nx.Graph,
        entity_embeddings: dict,
        chunk_embeddings: dict,
        tau_e: float = 0.7,
        tau_d: float = 0.6,
        top_k: int = 5,
    ):
        try:
            self.graph = graph
            self.entity_embeddings = entity_embeddings
            self.chunk_embeddings = chunk_embeddings
            self.tau_e = tau_e
            self.tau_d = tau_d
            self.top_k = top_k
        except Exception as e:
            raise RAGException(f"Failed to initialize LocalGraphRAGRetriever: {e}", sys)

    def retrieve(self, query_embedding: np.ndarray):
        try:
            logger.info("===== LocalGraphRAGRetriever: START RETRIEVAL =====")

            # ---------- Step 0: Sanity checks ----------
            logger.info(
                "Inputs | entities=%d | chunks=%d | tau_e=%.2f | tau_d=%.2f | top_k=%d",
                len(self.entity_embeddings),
                len(self.chunk_embeddings),
                self.tau_e,
                self.tau_d,
                self.top_k,
            )

            # ---------- Step 1: Entity similarity ----------
            entities = list(self.entity_embeddings.keys())
            entity_vectors = [self.entity_embeddings[e] for e in entities]

            logger.info("Step 1 | Computing similarity for %d entities", len(entities))
            logger.info("entities sample: %s", entities[:5])

            entity_similarities = cosine_similarity(query_embedding.reshape(1, -1), entity_vectors)[0]
            filtered_entities = [
                (entities[i], entity_similarities[i])
                for i in range(len(entities))
                if entity_similarities[i] >= self.tau_e
            ]

            if not filtered_entities:
                logger.info("STOP | No entities passed tau_e")
                return []

            # ---------- Step 2: Retrieve related chunks ----------
            candidate_chunks = []
            candidate_scores = []

            for entity, entity_score in filtered_entities:
                if entity not in self.graph:
                    logger.info("Entity missing in graph | %s", entity)
                    continue

                chunk_ids = self.graph.nodes[entity].get("chunks", [])
                for chunk_id in chunk_ids:
                    if chunk_id not in self.chunk_embeddings:
                        continue
                    chunk_emb = self.chunk_embeddings[chunk_id]
                    chunk_sim = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        np.array(chunk_emb).reshape(1, -1)
                    )[0][0]

                    if chunk_sim >= self.tau_d:
                        candidate_chunks.append(chunk_id)
                        candidate_scores.append(entity_score * chunk_sim)

            if not candidate_chunks:
                logger.info("STOP | No chunks passed tau_d")
                return []

            # ---------- Step 3: Rank & return ----------
            idxs = np.argsort(candidate_scores)[::-1][: self.top_k]
            results = [(candidate_chunks[i], candidate_scores[i]) for i in idxs]

            logger.info("===== LocalGraphRAGRetriever: END RETRIEVAL =====")
            return results

        except Exception as e:
            raise RAGException(f"LocalGraphRAGRetriever.retrieve failed: {e}", sys)


if __name__ == "__main__":
    try:
        import pickle
        import json
        from sentence_transformers import SentenceTransformer

        # Load graph & chunks
        with open("data/processed/knowledge_graph.pkl", "rb") as f:
            G = pickle.load(f)

        with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Entity embeddings
        entity_embs = {
            n: G.nodes[n]["embedding"]
            for n in G.nodes
            if "embedding" in G.nodes[n]
        }

        chunk_embs = {
            c["id"]: np.array(c["embedding"])
            for c in chunks
            if "embedding" in c
        }

        retriever = LocalGraphRAGRetriever(
            graph=G,
            entity_embeddings=entity_embs,
            chunk_embeddings=chunk_embs,
            tau_e=0.45,
            tau_d=0.35,
            top_k=5,
        )

        query_emb = model.encode(
            "What did Dr. B. R. Ambedkar say about the caste system?"
        )

        results = retriever.retrieve(query_emb)
        print(results)

    except RAGException as e:
        print(f"[ERROR] {e}")
