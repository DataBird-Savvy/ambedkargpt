import json
import pickle
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.local_search import LocalGraphRAGRetriever
from retrieval.global_search import GlobalGraphRAGRetriever
from llm.answer_generator import RAGAnswerGenerator

_EMBEDDER = None

class AmbedkarGPTPipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        global _EMBEDDER
        if _EMBEDDER is None:
            _EMBEDDER = SentenceTransformer(
                self.config["embedding"]["model_name"]
            )
        self.embedder = _EMBEDDER
        self.graph = self._load_graph()
        self.chunks = self._load_chunks()

        self.chunk_texts = {c["id"]: c["text"] for c in self.chunks}
        self.chunk_embeddings = {
            c["id"]: np.array(c["embedding"])
            for c in self.chunks
            if "embedding" in c
        }

        self.entity_embeddings = {
            n: self.graph.nodes[n]["embedding"]
            for n in self.graph.nodes
            if "embedding" in self.graph.nodes[n]
        }

        self.community_nodes = self._load_json(
            self.config["paths"]["community_nodes"]
        )
        
        with open(self.config["paths"]["community_embeddings"], "rb") as f:
            self.community_embeddings = pickle.load(f)
                    
        self.community_chunks = {
            int(k): v
            for k, v in self._load_json(
                self.config["paths"]["community_chunks"]
            ).items()
        }

        self.local_retriever = LocalGraphRAGRetriever(
            graph=self.graph,
            entity_embeddings=self.entity_embeddings,
            chunk_embeddings=self.chunk_embeddings,
            tau_e=0.45,
            tau_d=0.35,
            top_k=5,
        )

       
    
        self.global_retriever = GlobalGraphRAGRetriever(
            community_vectors=self.community_embeddings,
            community_chunks=self.community_chunks,
            chunk_embeddings=self.chunk_embeddings,
            top_k_communities=3,
            top_k_chunks=5,
        )

        self.answer_generator = RAGAnswerGenerator(self.chunk_texts)

    def _load_graph(self):
        with open(self.config["paths"]["knowledge_graph"], "rb") as f:
            return pickle.load(f)

    def _load_chunks(self):
        with open(self.config["paths"]["chunks"], "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self, query: str) -> str:
        query_emb = self.embedder.encode(query)

        local_results = self.local_retriever.retrieve(query_emb)
        global_results = self.global_retriever.retrieve(
            query_emb.reshape(1, -1)
        )

        return self.answer_generator.generate(
            query,
            local_results,
            global_results,
        )


# ✅ THIS IS WHAT CLI NEEDS
def main():
    pipeline = AmbedkarGPTPipeline()

    print("\nAmbedkarGPT (SEMRAG-based RAG)")
    print("Type your question (Ctrl+C to exit)\n")

    while True:
        try:
            query = input(">> ")
            answer = pipeline.run(query)
            print("\nAnswer:\n", answer, "\n")
        except KeyboardInterrupt:
            print("\nExiting AmbedkarGPT.")
            break
if __name__ == "__main__":
    main()