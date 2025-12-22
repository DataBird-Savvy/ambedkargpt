class SEMRAGPromptBuilder:
    """
    Professional prompt builder for SEMRAG-based RAG systems.
    Produces concise answers grounded in retrieved context without mentioning sources.
    """

    def __init__(
        self,
        system_role: str | None = None,
        refusal_message: str = "I’m sorry, I don’t have enough information to answer that question."
    ):
        self.system_role = system_role or (
            "You are a knowledgeable research assistant specializing in Dr. B.R. Ambedkar's works. "
            "Provide factual, precise, and concise answers based only on the provided context."
        )
        self.refusal_message = refusal_message

    def build(self, query: str, context: str) -> str:
        return f"""
{self.system_role}

Instructions:
- Answer strictly using the provided context.
- Give clear, concise, and professional responses.
- Do NOT mention sources, context IDs, or retrieval steps.
- Do NOT use external knowledge or make assumptions.
- If the context is insufficient, respond exactly:
  "{self.refusal_message}"

Context:
{context}

Question:
{query}

Answer:
""".strip()
