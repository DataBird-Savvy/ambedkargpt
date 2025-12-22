from llm.llm_client import LLMClient
from typing import List, Tuple, Dict
from logger import logger
from retrieval.ranker import ChunkReranker


from typing import List, Tuple, Dict
from logger import logger
from llm.llm_client import LLMClient
from retrieval.ranker import ChunkReranker
from llm.prompt_templates import SEMRAGPromptBuilder


class RAGAnswerGenerator:
    """
    Generate answers using reranked local + global SEMRAG context.
    """

    def __init__(
        self,
        chunk_texts: Dict[int, str],
        top_k_context: int = 5,
    ):
        self.chunk_texts = chunk_texts
        self.top_k_context = top_k_context

        self.llm_client = LLMClient()
        self.reranker = ChunkReranker(chunk_texts)
        self.prompt_builder = SEMRAGPromptBuilder()

        logger.info(
            "RAGAnswerGenerator initialized | top_k_context=%d",
        
            top_k_context,
        )

    def generate(
        self,
        query: str,
        retrieved_local: List[Tuple[int, float]],
        retrieved_global: List[Tuple[int, float]],
    ) -> str:
        """
        Generate answer using reranked local and global chunks.
        """
        logger.info("===== RAGAnswerGenerator: START =====")
        logger.info("Query: %s", query)

        # ---------- Step 1: Rerank ----------
        reranked_context = self.reranker.rerank(
            retrieved_local=retrieved_local,
            retrieved_global=retrieved_global,
            top_k=self.top_k_context,
        )
        context_text = "\n\n".join(
            text for _, (_, _, _, text) in enumerate(reranked_context)
        )



        if not reranked_context:
            logger.warning("No chunks available after reranking")
            return "The provided documents do not contain enough information."

        # ---------- Step 3: Prompt ----------
        prompt = self.prompt_builder.build(
            query=query,
            context=context_text          
        )

        logger.info("Prompt constructed")

        # ---------- Step 4: LLM ----------
        answer = self.llm_client.generate(prompt)

        logger.info("LLM response generated")
        logger.info("===== RAGAnswerGenerator: END =====")

        return answer


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
    query_emb = query_emb.reshape(1, -1)
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
    
    
    generator = RAGAnswerGenerator(chunk_texts)

    answer = generator.generate(query, retrieved_local, retrieved_global)
    print("Generated Answer:\n", answer)