import networkx as nx
from typing import Dict
from src.llm.llm_client import LLMClient
from logger import logger
import numpy as np
import pickle
from src.exception import RAGException
import sys
import os

class GraphSummarizer:
    def __init__(self, save_embeddings_path=None):
        """
        save_embeddings_path: optional path to save computed community embeddings as .pkl
        """
        try:
            self.llm = LLMClient()
            self.save_embeddings_path = save_embeddings_path
        except Exception as e:
            raise RAGException(f"Failed to initialize GraphSummarizer: {e}", sys)

    def summarize_community(
        self,
        G: nx.MultiDiGraph,
        community_map: Dict[str, int],
        chunk_lookup: dict,
    ) -> Dict[int, str]:
        try:
            logger.info("===== GraphSummarizer: START =====")
            logger.info("Total nodes in graph: %d", G.number_of_nodes())
            logger.info("Total communities detected: %d", len(set(community_map.values())))

            # 1. Group entities by community
            communities = {}
            for node, cid in community_map.items():
                communities.setdefault(cid, []).append(node)

            # Compute community embeddings (mean of node embeddings)
            community_embs = {}
            for cid, nodes in communities.items():
                embs = [
                    G.nodes[n]["embedding"]
                    for n in nodes
                    if "embedding" in G.nodes[n]
                ]
                if embs:
                    community_embs[int(cid)] = np.mean(embs, axis=0)

            # Save embeddings if path is provided
            if self.save_embeddings_path:
                try:
                    os.makedirs(os.path.dirname(self.save_embeddings_path), exist_ok=True)
                    with open(self.save_embeddings_path, "wb") as f:
                        pickle.dump(community_embs, f)
                    logger.info(
                        "Saved %d community embeddings to %s",
                        len(community_embs),
                        self.save_embeddings_path
                    )
                except Exception as e:
                    raise RAGException(f"Failed to save community embeddings: {e}", sys)

            summaries = {}

            # 2. Process each community for summarization
            for cid, entities in communities.items():
                logger.info("Processing community ID %d | %d entities", cid, len(entities))
                relations = []
                supporting_chunks = set()

                for u, v, data in G.edges(data=True):
                    if u in entities and v in entities:
                        relations.append(f"{u} -[{data.get('relation')}]-> {v}")
                        for ch in data.get("chunks", []):
                            supporting_chunks.add(ch)

                logger.info(
                    "Community %d | Relations found: %d | Supporting chunks: %d",
                    cid,
                    len(relations),
                    len(supporting_chunks)
                )

                # Build evidence text
                chunk_texts = [
                    chunk_lookup[ch][:300]
                    for ch in list(supporting_chunks)[:5]
                    if ch in chunk_lookup
                ]

                if not relations:
                    logger.info("Community %d skipped: No relations", cid)
                    continue

                prompt = f"""
You are an expert historian.

Entities:
{', '.join(entities)}

Relations:
{chr(10).join(relations)}

Supporting evidence:
{chr(10).join(chunk_texts)}

Write a factual 5–7 sentence summary.
"""
                logger.info(
                    "Community %d | Generating summary with prompt length: %d chars",
                    cid,
                    len(prompt)
                )
                try:
                    summaries[cid] = self.llm.generate(prompt)
                    logger.info("Community %d | Summary generated", cid)
                except Exception as e:
                    raise RAGException(f"LLM generation failed for community {cid}: {e}", sys)

            logger.info("===== GraphSummarizer: END =====")
            return summaries

        except Exception as e:
            raise RAGException(f"Graph summarization failed: {e}", sys)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    try:
        from .graph_builder import GraphBuilder
        from .community_detector import CommunityDetector
        import json

        embedding_model = "all-MiniLM-L6-v2"
        gb = GraphBuilder(embedding_model)
        G_multi = gb.load("data/processed/knowledge_graph.pkl")
        if G_multi is None:
            raise RAGException("Failed to load knowledge graph.", sys)

        # Load chunks
        chunk_file = "data/processed/chunks.json"
        if not os.path.exists(chunk_file):
            raise RAGException(f"Chunk file not found: {chunk_file}", sys)

        with open(chunk_file, "r", encoding="utf-8") as f:
            chunk_list = json.load(f)
        if not chunk_list:
            raise RAGException("No chunks loaded from file.", sys)

        # Build chunk_id → text map
        chunks = {c["id"]: c["text"] for c in chunk_list}

        # Detect communities
        detector = CommunityDetector()
        community_map = detector.detect(G_multi)

        # Initialize summarizer and save embeddings
        summarizer = GraphSummarizer(save_embeddings_path="data/processed/community_embeddings.pkl")
        result = summarizer.summarize_community(G_multi, community_map, chunks)

        for cid, summary in result.items():
            print(f"Community ID: {cid}\nSummary: {summary}\n")

    except RAGException as e:
        print(f"[ERROR] {e}")
