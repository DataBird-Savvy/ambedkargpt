import networkx as nx
from typing import Dict
from src.llm.llm_client import LLMClient
from logger import logger

class GraphSummarizer:
    def __init__(self):
        self.llm = LLMClient()

    def summarize_community(
        self,
        G: nx.MultiDiGraph,
        community_map: Dict[str, int],
        chunk_lookup: dict,
    ) -> Dict[int, str]:

        logger.info("===== GraphSummarizer: START =====")
        logger.info("Total nodes in graph: %d", G.number_of_nodes())
        logger.info("Total communities detected: %d", len(set(community_map.values())))

        # 1. Group entities by community
        communities = {}
        for node, cid in community_map.items():
            communities.setdefault(cid, []).append(node)

        summaries = {}

        # 2. Process each community
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

            # 3. Build evidence text
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
            logger.info("Community %d | Generating summary with prompt length: %d chars", cid, len(prompt))
            summaries[cid] = self.llm.generate(prompt)
            logger.info("Community %d | Summary generated", cid)

        logger.info("===== GraphSummarizer: END =====")
        return summaries


if __name__ == "__main__":
    
    from .graph_builder import GraphBuilder
    from .community_detector import CommunityDetector
    import json

    gb = GraphBuilder()
    G_multi = gb.load("data/processed/knowledge_graph.pkl")

    # Load chunks
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunk_list = json.load(f)

    # Build chunk_id → text map
    chunks = {c["id"]: c["text"] for c in chunk_list}

    # Detect communities
    detector = CommunityDetector()
    community_map = detector.detect(G_multi)

    summerizer = GraphSummarizer()
    result = summerizer.summarize_community(G_multi, community_map, chunks)

    for cid, summary in result.items():
        print(f"Community ID: {cid}\nSummary: {summary}\n")
