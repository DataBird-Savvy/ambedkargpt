import networkx as nx
import numpy as np
from retrieval.ranker import ChunkReranker
from logger import logger
from sklearn.metrics.pairwise import cosine_similarity

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
        """
        Args:
            graph: Knowledge graph (nodes = entities)
            entity_embeddings: {entity_name: embedding}
            chunk_embeddings: {chunk_id: embedding}
            tau_e: Entity similarity threshold
            tau_d: Chunk similarity threshold
            top_k: Number of chunks to retrieve
        """
        self.graph = graph
        self.entity_embeddings = entity_embeddings
        self.chunk_embeddings = chunk_embeddings
        self.tau_e = tau_e
        self.tau_d = tau_d
        self.top_k = top_k

    def retrieve(self, query_embedding: np.ndarray):
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
        logger.info("entity_similarities sample: %s", entity_similarities[:5])
        filtered_entities = [
            (entities[i], entity_similarities[i])
            for i in range(len(entities))
            if entity_similarities[i] >= self.tau_e
        ]

        logger.info(
            "Step 1 | Filtered entities above tau_e: %d",
            len(filtered_entities),
        )
        logger.info("Filtered entities sample: %s", filtered_entities[:5])

        # Log top few entity scores
        for e, s in sorted(filtered_entities, key=lambda x: x[1], reverse=True)[:5]:
            logger.info("Entity match | %s | score=%.4f", e, s)

        if not filtered_entities:
            logger.info("STOP | No entities passed tau_e")
            return []

        # ---------- Step 2: Retrieve related chunks ----------
        candidate_chunks = []
        candidate_scores = []

        logger.info("Step 2 | Retrieving chunks linked to filtered entities")

        for entity, entity_score in filtered_entities:
            if entity not in self.graph:
                logger.info("Entity missing in graph | %s", entity)
                continue

            chunk_ids = self.graph.nodes[entity].get("chunks", [])

            logger.info(
                "Entity '%s' | linked_chunks=%d | entity_score=%.4f",
                entity,
                len(chunk_ids),
                entity_score,
            )

            for chunk_id in chunk_ids:
                if chunk_id not in self.chunk_embeddings:
                    logger.info(
                        "Chunk embedding missing | chunk_id=%s | entity=%s",
                        chunk_id,
                        entity,
                    )
                    continue

                chunk_emb = self.chunk_embeddings[chunk_id]
                chunk_sim = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    np.array(chunk_emb).reshape(1, -1)
                )[0][0]

                logger.info(
                    "Chunk similarity | chunk_id=%s | sim=%.4f",
                    chunk_id,
                    chunk_sim,
                )

                if chunk_sim >= self.tau_d:
                    final_score = entity_score * chunk_sim
                    candidate_chunks.append(chunk_id)
                    candidate_scores.append(final_score)

                    logger.info(
                        "ACCEPTED chunk | %s | final_score=%.4f",
                        chunk_id,
                        final_score,
                    )

        if not candidate_chunks:
            logger.info("STOP | No chunks passed tau_d")
            return []

        # ---------- Step 3: Rank & return ----------
        logger.info(
            "Step 3 | Ranking %d candidate chunks",
            len(candidate_chunks),
        )

        idxs = np.argsort(candidate_scores)[::-1][: self.top_k]
        results = [(candidate_chunks[i], candidate_scores[i]) for i in idxs]

        for cid, score in results:
            logger.info("FINAL RESULT | chunk_id=%s | score=%.4f", cid, score)

        logger.info("===== LocalGraphRAGRetriever: END RETRIEVAL =====")

        return results



if __name__ == "__main__":
    # Example usage
    import pickle
    import json
    from src.retrieval.local_search import LocalGraphRAGRetriever
    from sentence_transformers import SentenceTransformer
    # Load graph & embeddings
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