from fusesearch.llm.base import LLM


class AnthropicLLM(LLM):
    """Anthropic Claude LLM provider."""

    def __init__(
        self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None
    ):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the [anthropic] extra. "
                "Install with: pip install fusesearch[anthropic]"
            ) from None

        self.model = model
        self.client = Anthropic(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
