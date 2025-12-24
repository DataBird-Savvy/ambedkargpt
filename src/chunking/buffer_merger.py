import tiktoken
from src.chunking.semantic_chunker import SemanticChunker
from sentence_transformers import SentenceTransformer
from src.exception import RAGException
import os
import json

class SemanticChunkMerger:
    def __init__(self, max_tokens, subchunk_size, overlap, embedding_model):
        try:
            self.max_tokens = max_tokens
            self.subchunk_size = subchunk_size
            self.overlap = overlap
            self.embedder = SentenceTransformer(embedding_model)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            raise RAGException(f"Failed to initialize SemanticChunkMerger: {e}") from e

    def merge_chunks(self, chunks):
        """
        Merges semantic chunks with buffer context and applies sub-chunking
        when token limits are exceeded.
        """
        try:
            if not chunks:
                raise RAGException("No chunks provided to merge.")

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

        except Exception as e:
            raise RAGException(f"Chunk merging failed: {e}") from e

    def merge_and_embed(self, chunks):
        """
        Merge chunks with buffer + sub-chunking and generate embeddings.
        Returns:
            final_chunks: list of texts
            embeddings: list of vectors
        """
        try:
            final_chunks = self.merge_chunks(chunks)
            embeddings = self.embedder.encode(final_chunks)
            return final_chunks, embeddings
        except Exception as e:
            raise RAGException(f"Merging and embedding failed: {e}") from e

    def save_json(self, final_chunks, embeddings, output_path):
        try:
            chunk_data = [
                {"id": i + 1, "text": final_chunks[i], "embedding": embeddings[i].tolist()}
                for i in range(len(final_chunks))
            ]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise RAGException(f"Failed to save JSON to {output_path}: {e}") from e



if __name__ == "__main__":
    try:
        data_path = "data/Ambedkar_book.pdf"
        embedding_model = "all-MiniLM-L6-v2"
        max_tokens = 1024
        subchunk_size = 128
        overlap = 32
        output_path = "data/processed/chunks.json"
        sim_threshold = 0.65

        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(data_path)
        docs = loader.load()
        if not docs:
            raise RAGException(f"No documents loaded from {data_path}")

        text = "\n".join([d.page_content for d in docs])

        chunker = SemanticChunker(embedding_model=embedding_model, sim_threshold=sim_threshold)
        chunks = chunker.chunk(text)
        if not chunks:
            raise RAGException("SemanticChunker returned no chunks.")

        merger = SemanticChunkMerger(
            max_tokens=max_tokens,
            subchunk_size=subchunk_size,
            overlap=overlap,
            embedding_model=embedding_model
        )

        merged_chunks, embeddings = merger.merge_and_embed(chunks)
        merger.save_json(merged_chunks, embeddings, output_path)

        print(f"Chunks saved successfully to {output_path}")

    except RAGException as e:
        print(f"[ERROR] {e}")
