import networkx as nx
import pickle
from sentence_transformers import SentenceTransformer
from src.exception import RAGException
import sys
import os

class GraphBuilder:
    def __init__(self, embedding_model):
        try:
            self.G = nx.MultiDiGraph()
            self.embedder = SentenceTransformer(embedding_model)
            self.entity_cache = {}  # avoid re-embedding
        except Exception as e:
            raise RAGException(f"Failed to initialize GraphBuilder with model '{embedding_model}': {e}", sys)

    def get_entity_embedding(self, entity: str):
        try:
            if entity not in self.entity_cache:
                self.entity_cache[entity] = self.embedder.encode(entity)
            return self.entity_cache[entity]
        except Exception as e:
            raise RAGException(f"Failed to encode entity '{entity}': {e}", sys)

    def build(self, extracted):
        """
        extracted = [
          {
            "chunk_id": int,
            "entities": [...],
            "relations": [(subj, verb, obj)]
          }
        ]
        """
        try:
            if not extracted:
                raise RAGException("No extracted data provided for graph building.", sys)

            for item in extracted:
                chunk_id = item.get("chunk_id")
                if chunk_id is None:
                    raise RAGException(f"Missing chunk_id in item: {item}", sys)

                # ---- entity nodes ----
                for entity in item.get("entities", []):
                    if not self.G.has_node(entity):
                        self.G.add_node(
                            entity,
                            embedding=self.get_entity_embedding(entity),
                            chunks=[chunk_id]
                        )
                    else:
                        self.G.nodes[entity].setdefault("chunks", []).append(chunk_id)

                # ---- relation edges ----
                for subj, verb, obj in item.get("relations", []):
                    self.G.add_edge(
                        subj,
                        obj,
                        key=verb,
                        relation=verb,
                        chunks=[chunk_id]
                    )

            return self.G

        except Exception as e:
            raise RAGException(f"Graph building failed: {e}", sys)

    def save(self, path="data/processed/knowledge_graph.pkl"):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self.G, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise RAGException(f"Failed to save graph to '{path}': {e}", sys)

    def load(self, path="data/processed/knowledge_graph.pkl"):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise RAGException(f"Failed to load graph from '{path}': {e}", sys)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    try:
        import json
        from src.graph.entity_extractor import EntityExtractor

        chunk_path = "data/processed/chunks.json"
        embedding_model = "all-MiniLM-L6-v2"
        spacy_model = "en_core_web_sm"

        if not os.path.exists(chunk_path):
            raise RAGException(f"Chunk file not found: {chunk_path}", sys)

        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if not chunks:
            raise RAGException(f"No chunks loaded from {chunk_path}", sys)

        extractor = EntityExtractor(spacy_model)
        extracted = extractor.extract(chunks)
        if not extracted:
            raise RAGException("EntityExtractor returned no data.", sys)

        builder = GraphBuilder(embedding_model)
        graph = builder.build(extracted)
        builder.save()
        print(f"Graph saved successfully with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    except RAGException as e:
        print(f"[ERROR] {e}")
