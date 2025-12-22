
import networkx as nx
import pickle
from sentence_transformers import SentenceTransformer

class GraphBuilder:
    def __init__(self, embedding_model):
        self.G = nx.MultiDiGraph()
        self.embedder = SentenceTransformer(embedding_model)
        self.entity_cache = {}  # avoid re-embedding

    def get_entity_embedding(self, entity: str):
        if entity not in self.entity_cache:
            self.entity_cache[entity] = self.embedder.encode(entity)
        return self.entity_cache[entity]

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

        for item in extracted:
            chunk_id = item["chunk_id"]

            # ---- entity nodes ----
            for entity in item["entities"]:
                if not self.G.has_node(entity):
                    self.G.add_node(
                        entity,
                        embedding=self.get_entity_embedding(entity),
                        chunks=[chunk_id]
                    )
                else:
                    self.G.nodes[entity].setdefault("chunks", []).append(chunk_id)

            # ---- relation edges ----
            for subj, verb, obj in item["relations"]:
                self.G.add_edge(
                    subj,
                    obj,
                    key=verb,
                    relation=verb,
                    chunks=[chunk_id]
                )

        return self.G


    def save(self, path="data/processed/knowledge_graph.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.G, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path="data/processed/knowledge_graph.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
if __name__ == "__main__":

    import json
    
    from src.graph.entity_extractor import EntityExtractor
    chunk_path = "data/processed/chunks.json"
    embedding_model = "all-MiniLM-L6-v2"
    spacy_model = "en_core_web_sm"

    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    extractor = EntityExtractor(spacy_model)
    extracted = extractor.extract(chunks)

    builder = GraphBuilder(embedding_model)
    graph = builder.build(extracted)
    builder.save()