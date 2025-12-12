# src/graph/graph_builder.py

import networkx as nx
from typing import List, Dict
import pickle

class GraphBuilder:
    def __init__(self):
        self.G = nx.Graph()

    def build(self, extracted: List[Dict]):
        """
        extracted format:
        [
            {
                "chunk_id": ...,
                "entities": [...],
                "relations": [(subj, verb, obj), ...]
            }
        ]
        """
        for item in extracted:
            chunk_id = item["chunk_id"]

            # Add entity nodes
            for entity in item["entities"]:
                self.G.add_node(entity, chunks=[chunk_id])

            # Add relationship edges
            for subj, verb, obj in item["relations"]:
                self.G.add_edge(subj, obj, relation=verb, chunks=[chunk_id])

        return self.G

    def save(self, path="data/processed/knowledge_graph.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(self.G, f, pickle.HIGHEST_PROTOCOL)
        

    def load(self, path="data/processed/knowledge_graph.pkl"):
        
        with open(path, 'rb') as f:
            G_loaded = pickle.load(f)
        return G_loaded
if __name__ == "__main__":

    import json
    from entity_extractor import EntityExtractor
    chunk_path = "data/processed/chunks.json"

    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    extractor = EntityExtractor()
    extracted = extractor.extract(chunks)

    builder = GraphBuilder()
    graph = builder.build(extracted)
    builder.save()