from abc import ABC, abstractmethod


class LLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a completion given system and user prompts."""
