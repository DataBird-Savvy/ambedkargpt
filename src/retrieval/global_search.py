import numpy as np
from typing import Dict, List, Tuple
from logger import logger
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class GlobalGraphRAGRetriever:
    """
    Global GraphRAG Retriever

    1) Rank communities using community embeddings
    2) Collect chunks from top-K communities
    3) Rank chunks using chunk embeddings
    """

    def __init__(
        self,
        community_embeddings: Dict[int, np.ndarray],
        community_chunks: Dict[int, List[int]],
        chunk_embeddings: Dict[int, np.ndarray],
        top_k_communities: int = 3,
        top_k_chunks: int = 5,
    ):
        self.community_embeddings = community_embeddings
        self.community_chunks = community_chunks
        self.chunk_embeddings = chunk_embeddings
        self.top_k_communities = top_k_communities
        self.top_k_chunks = top_k_chunks
        
    @staticmethod    
    def build_community_embeddings(G, community_nodes):
        

        community_embs = {}

        for cid, nodes in community_nodes.items():
            embs = [
                G.nodes[n]["embedding"]
                for n in nodes
                if "embedding" in G.nodes[n]
            ]

            if embs:
                community_embs[int(cid)] = np.mean(embs, axis=0)

        return community_embs


    def retrieve(self, query_embedding: np.ndarray) -> List[Tuple[int, float]]:
        logger.info("===== GlobalGraphRAGRetriever: START =====")

        # ---------- Step 1: Rank communities ----------
        community_ids = list(self.community_embeddings.keys())
        community_vectors = [
            self.community_embeddings[cid] for cid in community_ids
        ]

        logger.info(
            "Step 1 | Ranking %d communities",
            len(community_ids),
        )

        comm_scores = cosine_similarity([query_embedding], community_vectors)[0]

        top_comm_idx = np.argsort(comm_scores)[::-1][: self.top_k_communities]
        top_communities = [
            (community_ids[i], comm_scores[i]) for i in top_comm_idx
        ]

        for cid, score in top_communities:
            logger.info(
                "Community match | community_id=%s | score=%.4f",
                cid,
                score,
            )

        if not top_communities:
            logger.info("STOP | No communities selected")
            return []

        # ---------- Step 2: Collect chunks ----------
        candidate_chunks = []
        candidate_scores = []

        logger.info("Step 2 | Collecting chunks from top communities")

        for cid, comm_score in top_communities:
            chunk_ids = self.community_chunks.get(cid, [])

            logger.info(
                "Community %s | chunks=%d",
                cid,
                len(chunk_ids),
            )

            for chunk_id in chunk_ids:
                if chunk_id not in self.chunk_embeddings:
                    logger.debug(
                        "Missing chunk embedding | chunk_id=%s",
                        chunk_id,
                    )
                    continue

                chunk_emb = self.chunk_embeddings[chunk_id]
                chunk_sim = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    np.array(chunk_emb).reshape(1, -1)
                )[0][0]

                final_score = comm_score * chunk_sim

                candidate_chunks.append(chunk_id)
                candidate_scores.append(final_score)

                logger.debug(
                    "Chunk score | chunk_id=%s | chunk_sim=%.4f | final=%.4f",
                    chunk_id,
                    chunk_sim,
                    final_score,
                )

        if not candidate_chunks:
            logger.info("STOP | No chunks collected")
            return []

        # ---------- Step 3: Rank & return ----------
        logger.info(
            "Step 3 | Ranking %d candidate chunks",
            len(candidate_chunks),
        )

        idxs = np.argsort(candidate_scores)[::-1][: self.top_k_chunks]
        results = [(candidate_chunks[i], candidate_scores[i]) for i in idxs]


        for cid, score in results:
            logger.info(
                "FINAL RESULT | chunk_id=%s | score=%.4f",
                cid,
                score,
            )

        logger.info("===== GlobalGraphRAGRetriever: END =====")
        return results
if __name__ == "__main__":
    import json
    import pickle
    from sentence_transformers import SentenceTransformer

    # Load graph
    with open("data/processed/knowledge_graph.pkl", "rb") as f:
        G = pickle.load(f)

    # Load community → entities
    with open("data/processed/community_nodes.json") as f:
        community_nodes = json.load(f)

    # Load community → chunks
    with open("data/processed/community_chunks.json") as f:
        community_chunks = {
            int(k): v for k, v in json.load(f).items()
        }

    # Load chunks with embeddings
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_embs = {
        c["id"]: np.array(c["embedding"]) for c in chunks
    }

    # 🔥 Build community embeddings dynamically
    community_embs = GlobalGraphRAGRetriever.build_community_embeddings(G, community_nodes)

    retriever = GlobalGraphRAGRetriever(
        community_embeddings=community_embs,
        community_chunks=community_chunks,
        chunk_embeddings=chunk_embs,
        top_k_communities=3,
        top_k_chunks=5,
    )

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = embedder.encode(
        "What did Dr. B. R. Ambedkar say about caste?"
    )
    query_emb = query_emb.reshape(1, -1)


    results = retriever.retrieve(query_emb)
    print(results)
