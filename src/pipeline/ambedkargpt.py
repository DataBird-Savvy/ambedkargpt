from src.retrieval.local_search import local_graph_rag_search
from src.retrieval.global_search import global_graph_rag_search
from src.llm.answer_generator import generate_answer
from src.graph.graph_builder import GraphBuilder
import json, pickle

# Load graph & embeddings
with open("data/processed/knowledge_graph.pkl", "rb") as f:
    G = pickle.load(f)

with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

# Example: assuming embeddings are stored in node/chunk attributes
entity_embs = {n: G.nodes[n]['embedding'] for n in G.nodes if 'embedding' in G.nodes[n]}
chunk_embs = {c['id']: c['embedding'] for c in chunks}

# Suppose community embeddings are precomputed
community_embs = {comm_id: comm_data['embedding'] for comm_id, comm_data in G.graph.get('communities', {}).items()}
community_chunks = {comm_id: comm_data['chunks'] for comm_id, comm_data in G.graph.get('communities', {}).items()}

# Query embedding (from your sentence-transformers model)
query_emb = model.encode("Tell me about caste in India")

local_results = local_graph_rag_search(query_emb, G, entity_embs, chunk_embs)
global_results = global_graph_rag_search(query_emb, community_embs, community_chunks, chunk_embs)

answer = generate_answer("Tell me about caste in India", local_results, global_results)
print(answer)
