# AmbedkarGPT – SEMRAG-based RAG System

## 📌 Overview
AmbedkarGPT is a Retrieval-Augmented Generation (RAG) system implemented
based on the **SEMRAG research paper**.  
The system answers questions about **Dr. B. R. Ambedkar’s works** by
combining semantic chunking, knowledge graphs, and graph-based retrieval
with a local Large Language Model (LLM).

---

## 🧠 System Architecture
The system follows the SEMRAG pipeline:

1. **Semantic Chunking**
   - Sentence embeddings with cosine similarity
   - Buffer merging for contextual continuity
   - Token-aware chunk splitting

2. **Knowledge Graph Construction**
   - Entity extraction using spaCy
   - Relationship extraction via dependency parsing
   - Graph construction using NetworkX
   - Community detection (Louvain / Leiden)

3. **Retrieval Strategies**
   - **Local Graph RAG Search** (Equation 4 – SEMRAG)
   - **Global Graph RAG Search** (Equation 5 – SEMRAG)
   - Similarity thresholding and ranking
   | **Strategy**                            | **Focus Level** | **Retrieval Method**                                                | **Key Steps**                                                                                                                                                                                                                                                                                                              | **Output**                                        |
| --------------------------------------- | --------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **Local Graph RAG**<br/>(SEMRAG Eq. 4)  | Entity-level    | Retrieves chunks directly linked to entities in the knowledge graph | • Compute cosine similarity between query and entity embeddings<br/>• Filter entities with similarity ≥ τ<sub>e</sub><br/>• Retrieve chunks linked to filtered entities<br/>• Compute similarity between query and chunk embeddings<br/>• Filter chunks with similarity ≥ τ<sub>d</sub><br/>• Rank and select top-K chunks | Highly precise, entity-specific evidence          |
| **Global Graph RAG**<br/>(SEMRAG Eq. 5) | Community-level | Retrieves chunks from relevant graph communities                    | • Compute community embeddings (mean of entity embeddings)<br/>• Compute similarity between query and community embeddings<br/>• Select top-K communities<br/>• Collect chunks from selected communities<br/>• Rank chunks using combined similarity score                                                                 | Broader, thematic context for multi-hop reasoning |
| **Similarity Thresholding**             | Both            | Filters low-relevance entities and chunks                           | • Apply τ<sub>e</sub> for entity similarity<br/>• Apply τ<sub>d</sub> for chunk similarity                                                                                                                                                                                                                                 | Noise reduction, improved precision               |
| **Top-K Ranking**                       | Both            | Limits context size and prioritizes relevance                       | • Rank candidates by final score<br/>• Select top-K results                                                                                                                                                                                                                                                                | Controlled context size, lower latency            |



4. **LLM Integration**
   - Local LLM (Llama3 / Mistral via Ollama)
   - Prompt templates with retrieved entities & summaries
   - Answer generation with citations

---

## 🛠️ Tech Stack
- **Python 3.9+**
- sentence-transformers
- spaCy
- networkx
- scikit-learn
- langchain
- Ollama (Llama3 / Mistral)

---

## 📂 Project Structure



ambedkargpt/
├── data/
│ ├── Ambedkar_works.pdf
│ └── processed/
│ ├── chunks.json
│ └── knowledge_graph.pkl
├── src/
│ ├── chunking/
│ │ ├── semantic_chunker.py # Algorithm 1 (SEMRAG)
│ │ └── buffer_merger.py
│ ├── graph/
│ │ ├── entity_extractor.py
│ │ ├── graph_builder.py
│ │ ├── community_detector.py
│ │ └── summarizer.py
│ ├── retrieval/
│ │ ├── local_search.py # Equation 4 (SEMRAG)
│ │ ├── global_search.py # Equation 5 (SEMRAG)
│ │ └── ranker.py
│ ├── llm/
│ │ ├── llm_client.py
│ │ ├── prompt_templates.py
│ │ └── answer_generator.py
│ └── pipeline/
│ └── ambedkargpt.py # Main pipeline
├── tests/
│ ├── test_chunking.py
│ ├── test_retrieval.py
│ └── test_integration.py
├── config.yaml
├── requirements.txt
├── setup.py
└── README.md

## 📚 References

- SEMRAG: Semantic Retrieval-Augmented Generation (Research Paper)
- Dr. B. R. Ambedkar – Collected Works

## Output:

![alt text](image.png)