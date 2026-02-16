import os

from fusesearch.llm.base import LLM

_PROVIDERS = ["anthropic", "openai", "ollama"]


def _auto_detect() -> str:
    """Return the first available LLM provider, or raise ImportError."""
    for provider in _PROVIDERS:
        try:
            if provider == "anthropic":
                import anthropic  # noqa: F401
            elif provider == "openai":
                import openai  # noqa: F401
            elif provider == "ollama":
                import ollama  # noqa: F401
            return provider
        except ImportError:
            continue

    extras = ", ".join(f"[{p}]" for p in _PROVIDERS)
    raise ImportError(
        f"No LLM provider installed. Install one of: {extras}\n"
        f"Example: pip install fusesearch[anthropic]"
    )


def create_llm(provider: str | None = None) -> LLM:
    """Create an LLM instance based on provider name or FUSESEARCH_LLM env var."""
    provider = provider or os.getenv("FUSESEARCH_LLM") or _auto_detect()

    if provider == "anthropic":
        from fusesearch.llm.anthropic import AnthropicLLM

        return AnthropicLLM()
    elif provider == "openai":
        from fusesearch.llm.openai import OpenAILLM

        return OpenAILLM()
    elif provider == "ollama":
        from fusesearch.llm.ollama import OllamaLLM

        return OllamaLLM()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
