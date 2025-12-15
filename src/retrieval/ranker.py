from typing import List, Tuple, Dict
from logger import logger

class ChunkReranker:
    """
    Merge and rerank chunks from local and global RAG retrievers.
    """

    def __init__(self, chunk_texts: Dict[int, str]):
        """
        Args:
            chunk_texts (dict): Mapping from chunk_id -> chunk text
        """
        self.chunk_texts = chunk_texts
        logger.info(
            "ChunkReranker initialized | total_chunks=%d",
            len(chunk_texts)
        )

    def rerank(
        self,
        retrieved_local: List[Tuple[int, float]],
        retrieved_global: List[Tuple[int, float]],
        top_k: int = 10,
    ) -> List[Tuple[int, float, str, str]]:
        """
        Merge local and global chunks and rerank by score.
        """
        logger.info("===== ChunkReranker: START =====")

        logger.info(
            "Inputs | local_chunks=%d | global_chunks=%d | top_k=%d",
            len(retrieved_local),
            len(retrieved_global),
            top_k,
        )

        # ---------- Step 1: Merge ----------
        all_chunks = []

        for chunk_id, score in retrieved_local:
            text = self.chunk_texts.get(chunk_id, "")
            all_chunks.append((chunk_id, score, "Local", text))
            logger.info(
                "Added Local chunk | chunk_id=%s | score=%.4f",
                chunk_id,
                score,
            )

        for chunk_id, score in retrieved_global:
            text = self.chunk_texts.get(chunk_id, "")
            all_chunks.append((chunk_id, score, "Global", text))
            logger.info(
                "Added Global chunk | chunk_id=%s | score=%.4f",
                chunk_id,
                score,
            )

        if not all_chunks:
            logger.info("STOP | No chunks to rerank")
            return []

        logger.info(
            "Step 1 | Total merged chunks=%d",
            len(all_chunks),
        )

        # ---------- Step 2: Rerank ----------
        logger.info("Step 2 | Reranking chunks by score")

        all_chunks.sort(key=lambda x: x[1], reverse=True)

        # ---------- Step 3: Select top-K ----------
        reranked = all_chunks[:top_k]

        logger.info(
            "Step 3 | Selected top-%d chunks",
            len(reranked),
        )

        for rank, (cid, score, source, _) in enumerate(reranked, start=1):
            logger.info(
                "FINAL RANK %d | chunk_id=%s | source=%s | score=%.4f",
                rank,
                cid,
                source,
                score,
            )

        logger.info("===== ChunkReranker: END =====")
        return reranked

if __name__ == "__main__":
    # Example usage
    import pickle
    import json
    from src.retrieval.local_search import LocalGraphRAGRetriever
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from src.retrieval.global_search import GlobalGraphRAGRetriever
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "Who was Dr. B. R. Ambedkar?"
    
    with open("data/processed/knowledge_graph.pkl", "rb") as f:
        G = pickle.load(f)

    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)


    

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

    query_emb = model.encode(query)
    retrieved_local = retriever.retrieve(query_emb)
    chunk_texts = {
        c["id"]: c["text"] for c in chunks
    }
    
  

    # Load community → entities
    with open("data/processed/community_nodes.json") as f:
        community_nodes = json.load(f)

    # Load community → chunks
    with open("data/processed/community_chunks.json") as f:
        community_chunks = {
            int(k): v for k, v in json.load(f).items()
        }



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

   
  
    retrieved_global = retriever.retrieve(query_emb)
    reranker = ChunkReranker(chunk_texts)
    reranked_chunks = reranker.rerank(retrieved_local, retrieved_global, top_k=3)

    for chunk_id, score, source, text in reranked_chunks:
        print(f"ID: {chunk_id}, Score: {score:.4f}, Source: {source}, Text: {text}")