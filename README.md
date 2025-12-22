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
   | Strategy                  | Focus Level      | Retrieval Method                                                                 | Key Steps                                                                                                   | Output                                           |
|----------------------------|----------------|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| **Local Graph RAG**        | Entity-level    | Retrieve chunks linked to individual entities in the knowledge graph          | 1. Compute cosine similarity between query and entity embeddings<br>2. Filter entities above τ_e<br>3. Retrieve chunks linked to filtered entities<br>4. Compute similarity between query and chunk embeddings<br>5. Filter chunks above τ_d<br>6. Rank top-k chunks | Small, precise set of entity-specific chunks   |
| **Global Graph RAG**       | Community-level | Retrieve chunks from relevant communities (groups of related entities)        | 1. Compute community embeddings (average of entity embeddings)<br>2. Compute similarity between query and community embeddings<br>3. Select top-k communities<br>4. Retrieve all chunks from selected communities<br>5. Compute similarity with query<br>6. Rank top-k chunks | Broader, thematic context for multi-hop reasoning |
| **Similarity Thresholding**| Both            | Filter results using a minimum similarity threshold                            | Apply τ_e for entities and τ_d for chunks in Local Graph RAG<br>Use similarity scores in Global Graph RAG | Ensures only relevant chunks are considered    |
| **Top-K Ranking**          | Both            | Rank retrieved chunks by their final score (entity/community similarity × chunk similarity) | Sort candidate chunks by score and select top-k                                                       | Controls number of chunks fed to LLM, reduces noise |


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