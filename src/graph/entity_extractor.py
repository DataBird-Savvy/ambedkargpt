# src/graph/entity_extractor.py

import spacy
from typing import List, Dict

class EntityExtractor:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    def extract(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract entities + relations from each chunk.
        chunks = [{ "id": ..., "text": ... }]
        """
        results = []

        for chunk in chunks:
            print("Processing chunk ID:", chunk)
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

if __name__ == "__main__":
    import json
    chunk_path = "data/processed/chunks.json"

    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    extractor = EntityExtractor()
    extracted = extractor.extract(chunks)
    print("Extracted Entities and Relations:",extracted)
    for item in extracted:
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Entities: {item['entities']}")
        print(f"Relations: {item['relations']}")
        print("-" * 40)