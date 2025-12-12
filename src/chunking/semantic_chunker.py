import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")


class SemanticChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2", similarity_threshold=0.65):
        """
        Initialize the semantic chunker with:
        - SentenceTransformer model
        - Cosine similarity threshold
        """
        self.model = SentenceTransformer(model_name)
        self.sim_threshold = similarity_threshold

    def chunk(self, text: str):
        """
        Perform semantic chunking on the input text.
        Steps:
        1. Sentence splitting
        2. Embedding each sentence
        3. Grouping based on semantic similarity
        """
        # Step 1: Sentence tokenize
        sentences = nltk.sent_tokenize(text)

        if len(sentences) == 0:
            return []

        # Step 2: Encode sentences
        embeddings = self.model.encode(sentences)

        chunks = []
        current_chunk = [sentences[0]]

        # Step 3: Group similar sentences
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

        chunks.append(" ".join(current_chunk))  # final chunk

        return chunks

if __name__ == "__main__": 
    data_path = "data/Ambedkar_book.pdf" 
    from langchain_community.document_loaders import PyPDFLoader 
    loader = PyPDFLoader(data_path) 
    docs = loader.load() 
    for doc in docs: 
        print(doc.page_content) 
    text = "\n".join([d.page_content for d in docs]) 
    chunker = SemanticChunker()
    chunks = chunker.chunk(text) 
    for i, chunk in enumerate(chunks): 
        print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}\n")