import numpy as np
from typing import Dict, List, Tuple
from logger import logger
from sklearn.metrics.pairwise import cosine_similarity
from src.exception import RAGException
import sys


class GlobalGraphRAGRetriever:
    """
    Global GraphRAG Retriever

    1) Rank communities using community embeddings
    2) Collect chunks from top-K communities
    3) Rank chunks using chunk embeddings
    """

    def __init__(
        self,
        community_vectors: Dict[int, np.ndarray],
        community_chunks: Dict[int, List[int]],
        chunk_embeddings: Dict[int, np.ndarray],
        top_k_communities: int = 3,
        top_k_chunks: int = 5,
    ):
        try:
            self.community_vectors = community_vectors
            self.community_chunks = community_chunks
            self.chunk_embeddings = chunk_embeddings
            self.top_k_communities = top_k_communities
            self.top_k_chunks = top_k_chunks
        except Exception as e:
            raise RAGException(f"Failed to initialize GlobalGraphRAGRetriever: {e}", sys)

    def retrieve(self, query_embedding: np.ndarray) -> List[Tuple[int, float]]:
        try:
            logger.info("===== GlobalGraphRAGRetriever: START =====")

            # ---------- Step 1: Rank communities ----------
            community_ids = list(self.community_vectors.keys())
            community_vectors = [
                np.asarray(self.community_vectors[cid]).reshape(-1)
                for cid in community_ids
            ]

            if not community_vectors:
                logger.info("STOP | No community vectors available")
                return []

            comm_scores = cosine_similarity(
                query_embedding.reshape(1, -1),
                np.vstack(community_vectors)
            )[0]

            top_comm_idx = np.argsort(comm_scores)[::-1][: self.top_k_communities]
            top_communities = [
                (community_ids[i], comm_scores[i]) for i in top_comm_idx
            ]

            if not top_communities:
                logger.info("STOP | No communities selected")
                return []

            # ---------- Step 2: Collect chunks ----------
            candidate_chunks = []
            candidate_scores = []

            for cid, comm_score in top_communities:
                chunk_ids = self.community_chunks.get(cid, [])
                for chunk_id in chunk_ids:
                    if chunk_id not in self.chunk_embeddings:
                        continue
                    chunk_emb = self.chunk_embeddings[chunk_id]
                    chunk_sim = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        np.array(chunk_emb).reshape(1, -1)
                    )[0][0]
                    final_score = comm_score * chunk_sim
                    candidate_chunks.append(chunk_id)
                    candidate_scores.append(final_score)

            if not candidate_chunks:
                logger.info("STOP | No chunks collected")
                return []

            # ---------- Step 3: Rank & return ----------
            idxs = np.argsort(candidate_scores)[::-1][: self.top_k_chunks]
            results = [(candidate_chunks[i], candidate_scores[i]) for i in idxs]

            logger.info("===== GlobalGraphRAGRetriever: END =====")
            return results

        except Exception as e:
            raise RAGException(f"GlobalGraphRAGRetriever.retrieve failed: {e}", sys)


if __name__ == "__main__":
    try:
        import json
        import pickle
        from sentence_transformers import SentenceTransformer

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

        with open("data/processed/community_embeddings.pkl", "rb") as f:
            community_embs = pickle.load(f)

        retriever = GlobalGraphRAGRetriever(
            community_vectors=community_embs,
            community_chunks=community_chunks,
            chunk_embeddings=chunk_embs,
            top_k_communities=3,
            top_k_chunks=5,
        )

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = embedder.encode(
            "What did Dr. B. R. Ambedkar say about caste?"
        )

        results = retriever.retrieve(query_emb)
        print(results)

    except RAGException as e:
        print(f"[ERROR] {e}")
