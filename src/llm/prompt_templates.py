class SEMRAGPromptBuilder:
    """
    Optimized prompt: concise instructions, clear context, minimal repetition.
    """

    def __init__(
        self,
        system_role: str | None = None,
        refusal_message: str = "The provided documents do not contain enough information."
    ):
        self.system_role = system_role or (
            "You are an expert research assistant answering questions "
            "strictly using the provided context from Dr. B. R. Ambedkar's works."
        )
        self.refusal_message = refusal_message

    def build(self, query: str, local_context: str, global_context: str) -> str:
        return f"""
{self.system_role}

Instructions:
- Answer strictly using the provided context.
- Use local evidence first; global context only if local is insufficient.
- Do NOT use external knowledge or fabricate information.
- If insufficient info, reply exactly: "{self.refusal_message}".
- Citations are internal; do NOT output chunk IDs or labels.

Context:
Local (factual):
{local_context}

Global (themes/background):
{global_context}

Question:
{query}

Answer:
""".strip()
