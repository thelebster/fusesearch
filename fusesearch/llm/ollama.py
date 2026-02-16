import os

from fusesearch.llm.base import LLM


class OllamaLLM(LLM):
    """Ollama local LLM provider."""

    def __init__(self, model: str = "llama3.2", host: str | None = None):
        try:
            from ollama import Client
        except ImportError:
            raise ImportError(
                "Ollama provider requires the [ollama] extra. "
                "Install with: pip install fusesearch[ollama]"
            ) from None

        self.model = model
        self.client = Client(host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.message.content
