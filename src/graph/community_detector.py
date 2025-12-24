import networkx as nx
import community  # python-louvain
from typing import Dict
from .graph_builder import GraphBuilder
import json
from collections import defaultdict
from src.exception import RAGException
import os
import sys

class CommunityDetector:
    def __init__(self):
        pass

    def to_weighted_graph(self, G_multi: nx.MultiDiGraph) -> nx.Graph:
        try:
            W = nx.Graph()

            for u, v, data in G_multi.edges(data=True):
                weight = len(data.get("chunks", []))
                if weight == 0:
                    continue

                if W.has_edge(u, v):
                    W[u][v]["weight"] += weight
                else:
                    W.add_edge(u, v, weight=weight)

            return W
        except Exception as e:
            raise RAGException(f"Failed to create weighted graph: {e}", sys)

    def detect(self, G_multi: nx.MultiDiGraph) -> Dict[str, int]:
        """
        Detect communities from a weighted graph derived from MultiDiGraph.
        Returns {node: community_id}
        """
        try:
            W = self.to_weighted_graph(G_multi)
            partition = community.best_partition(W, weight="weight")
            return partition
        except Exception as e:
            raise RAGException(f"Community detection failed: {e}", sys)

    def build_artifacts(
        self, G: nx.MultiDiGraph, community_map: Dict[str, int]
    ):
        try:
            community_nodes = defaultdict(list)
            community_chunks = defaultdict(set)

            for node, cid in community_map.items():
                community_nodes[cid].append(node)

                for ch in G.nodes[node].get("chunks", []):
                    community_chunks[cid].add(ch)

            return (
                dict(community_nodes),
                {k: list(v) for k, v in community_chunks.items()},
            )
        except Exception as e:
            raise RAGException(f"Failed to build community artifacts: {e}", sys)

    def save(
        self,
        community_map: Dict[str, int],
        community_nodes: Dict[int, list],
        community_chunks: Dict[int, list],
        out_dir="data/processed",
    ):
        try:
            os.makedirs(out_dir, exist_ok=True)

            with open(f"{out_dir}/community_map.json", "w", encoding="utf-8") as f:
                json.dump(community_map, f, indent=2)

            with open(f"{out_dir}/community_nodes.json", "w", encoding="utf-8") as f:
                json.dump(community_nodes, f, indent=2)

            with open(f"{out_dir}/community_chunks.json", "w", encoding="utf-8") as f:
                json.dump(community_chunks, f, indent=2)
        except Exception as e:
            raise RAGException(f"Failed to save community files: {e}", sys)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    try:
        gb = GraphBuilder(embedding_model="all-MiniLM-L6-v2")
        G_multi = gb.load("data/processed/knowledge_graph.pkl")
        if G_multi is None:
            raise RAGException("Failed to load knowledge graph.", sys)

        detector = CommunityDetector()
        community_map = detector.detect(G_multi)

        community_nodes, community_chunks = detector.build_artifacts(
            G_multi, community_map
        )

        detector.save(community_map, community_nodes, community_chunks)

        print(
            f"Saved {len(community_nodes)} communities "
            f"with {sum(len(v) for v in community_chunks.values())} total chunks"
        )

    except RAGException as e:
        print(f"[ERROR] {e}")
