import spacy
from typing import List, Dict
from src.exception import RAGException
import json
import os

class EntityExtractor:
    def __init__(self, spacy_model):
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as e:
            raise RAGException(f"Failed to load spaCy model '{spacy_model}': {e}") from e

    def extract(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract entities + relations from each chunk.
        chunks = [{ "id": ..., "text": ... }]
        """
        if not chunks:
            raise RAGException("No chunks provided for entity extraction.")

        results = []

        try:
            for chunk in chunks:
                if "text" not in chunk or "id" not in chunk:
                    raise RAGException(f"Chunk missing 'id' or 'text': {chunk}")

                print("Processing chunk :", chunk)
                doc = self.nlp(chunk["text"])

                entities = list({ent.text for ent in doc.ents})

                relations = []
                for token in doc:
                    if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                        subj = token.text
                        verb = token.head.lemma_
                        obj = None

                        for child in token.head.children:
                            if child.dep_ in ("dobj", "pobj"):
                                obj = child.text

                        if obj:
                            relations.append((subj, verb, obj))

                results.append({
                    "chunk_id": chunk["id"],
                    "entities": entities,
                    "relations": relations
                })
            return results

        except Exception as e:
            raise RAGException(f"Entity extraction failed: {e}") from e


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    try:
        chunk_path = "data/processed/chunks.json"
        spacy_model = "en_core_web_sm"

        if not os.path.exists(chunk_path):
            raise RAGException(f"Chunk file not found: {chunk_path}")

        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            raise RAGException("No chunks loaded from file.")

        extractor = EntityExtractor(spacy_model)
        extracted = extractor.extract(chunks)

        print("Extracted Entities and Relations:", extracted)
        for item in extracted:
            print(f"Chunk ID: {item['chunk_id']}")
            print(f"Entities: {item['entities']}")
            print(f"Relations: {item['relations']}")
            print("-" * 40)

    except RAGException as e:
        print(f"[ERROR] {e}")
