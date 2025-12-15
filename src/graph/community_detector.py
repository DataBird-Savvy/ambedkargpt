# src/graph/community_detector.py

import networkx as nx
import community  # python-louvain
from typing import Dict
from .graph_builder import GraphBuilder
import json
from collections import defaultdict

class CommunityDetector:
    def __init__(self):
        pass

    def to_weighted_graph(self, G_multi: nx.MultiDiGraph) -> nx.Graph:
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

    def detect(self, G_multi: nx.MultiDiGraph) -> Dict[str, int]:
        """
        Detect communities from a weighted graph derived from MultiDiGraph.
        Returns {node: community_id}
        """
        W = self.to_weighted_graph(G_multi)
        partition = community.best_partition(W, weight="weight")
        return partition
    
    
    
    def build_artifacts(self, G: nx.MultiDiGraph, community_map: Dict[str, int]):
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

    def save(
        self,
        community_map: Dict[str, int],
        community_nodes: Dict[int, list],
        community_chunks: Dict[int, list],
        out_dir="data/processed",
    ):
        with open(f"{out_dir}/community_map.json", "w", encoding="utf-8") as f:
            json.dump(community_map, f, indent=2)

        with open(f"{out_dir}/community_nodes.json", "w", encoding="utf-8") as f:
            json.dump(community_nodes, f, indent=2)

        with open(f"{out_dir}/community_chunks.json", "w", encoding="utf-8") as f:
            json.dump(community_chunks, f, indent=2)


if __name__ == "__main__":
    # Load saved MultiDiGraph
    gb = GraphBuilder()
    G_multi = gb.load("data/processed/knowledge_graph.pkl")

    # Detect communities
    detector = CommunityDetector()
    community_map = detector.detect(G_multi)

    # 2️⃣ Build artifacts
    community_nodes, community_chunks = detector.build_artifacts(
        G_multi, community_map
    )

    # 3️⃣ Save
    detector.save(community_map, community_nodes, community_chunks)

    print(
        f"Saved {len(community_nodes)} communities "
        f"with {sum(len(v) for v in community_chunks.values())} total chunks"
    )