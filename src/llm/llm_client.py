# llm/llm_client.py
from ollama import Client



class LLMClient:
    def __init__(self, model="mistral:latest"):
        
        self.client = Client(host='http://localhost:11434')
        self.model = model
        

    def generate(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            
        )
        return response['message']['content']
