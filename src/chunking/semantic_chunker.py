from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from src.exception import RAGException

nlp = spacy.load("en_core_web_sm")


class SemanticChunker:
    def __init__(self, embedding_model, sim_threshold):
        try:
            self.embedder = SentenceTransformer(embedding_model)
        except Exception as e:
            raise RAGException(f"Failed to load embedding model '{embedding_model}': {e}") from e

        self.sim_threshold = sim_threshold

    def chunk(self, text: str):
        """
        Perform semantic chunking on the input text.
        Steps:
        1. Sentence splitting
        2. Embedding each sentence
        3. Grouping based on semantic similarity
        """
        try:
            # Step 1: Sentence tokenize
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]

            if len(sentences) == 0:
                raise RAGException("No sentences found in the input text.")

            # Step 2: Encode sentences
            try:
                embeddings = self.embedder.encode(
                    sentences,
                    batch_size=32,
                    show_progress_bar=False
                )
            except Exception as e:
                raise RAGException(f"Sentence embedding failed: {e}") from e

            # Step 3: Group similar sentences
            chunks = []
            current_chunk = [sentences[0]]

            for i in range(1, len(sentences)):
                sim = cosine_similarity(
                    [embeddings[i - 1]],
                    [embeddings[i]]
                )[0][0]

                if sim >= self.sim_threshold:
                    current_chunk.append(sentences[i])
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentences[i]]

            chunks.append(" ".join(current_chunk))
            return chunks

        except Exception as e:
            # Wrap any unexpected exception
            raise RAGException(f"Semantic chunking failed: {e}") from e



if __name__ == "__main__":
    try:
        data_path = "data/Ambedkar_book.pdf"
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(data_path)
        docs = loader.load()

        if not docs:
            raise RAGException(f"No documents loaded from {data_path}")

        text = "\n".join([d.page_content for d in docs])
        chunker = SemanticChunker(embedding_model="all-MiniLM-L6-v2", sim_threshold=0.65)
        chunks = chunker.chunk(text)

        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}\n")

    except RAGException as e:
        print(f"[ERROR] {e}")
