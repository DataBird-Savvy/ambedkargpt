import tiktoken
from semantic_chunker import SemanticChunker
from sentence_transformers import SentenceTransformer

class SemanticChunkMerger:
    def __init__(self, max_tokens=1024, subchunk_size=128, overlap=32):
        self.max_tokens = max_tokens
        self.subchunk_size = subchunk_size
        self.overlap = overlap
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def merge_chunks(self, chunks):
        """
        Merges semantic chunks with buffer context and applies sub-chunking
        when token limits are exceeded.
        """

        # ----- Step 4: Buffer Merging -----
        merged_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                merged_chunks.append(chunks[i])
            else:
                # Take last 2 sentences from previous chunk
                buffer = chunks[i - 1].split(".")[-2:]
                new_chunk = ". ".join(buffer) + ". " + chunks[i]
                merged_chunks.append(new_chunk)

        # ----- Step 5: Token-Limit Enforcement + Sub-Chunking -----
        final_chunks = []
        for chunk in merged_chunks:
            tokens = self.tokenizer.encode(chunk)

            if len(tokens) <= self.max_tokens:
                final_chunks.append(chunk)
            else:
                # Sub-chunking with overlap
                for i in range(0, len(tokens), self.subchunk_size - self.overlap):
                    sub_tokens = tokens[i: i + self.subchunk_size]
                    sub_text = self.tokenizer.decode(sub_tokens)
                    final_chunks.append(sub_text)

        return final_chunks
if __name__ == "__main__":
    data_path = "data/Ambedkar_book.pdf" 
    import json
    import os

    output_path = "data/processed/chunks.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    from langchain_community.document_loaders import PyPDFLoader 
    loader = PyPDFLoader(data_path) 
    docs = loader.load() 

    # Optional: print each page
    for doc in docs: 
        print(doc.page_content) 

    text = "\n".join([d.page_content for d in docs]) 

    chunker = SemanticChunker()
    chunks = chunker.chunk(text) 

    merger = SemanticChunkMerger()
    merged_chunks = merger.merge_chunks(chunks)

    # Wrap each chunk in dict with 'id' and 'text'
    embeddings = merger.embedder.encode(merged_chunks)

    chunk_data = [
        {
            "id": i + 1,
            "text": merged_chunks[i],
            "embedding": embeddings[i].tolist()  # IMPORTANT
        }
        for i in range(len(merged_chunks))
    ]


    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(chunk_data)} chunks to {output_path}")
