"""Base protocol for model backends."""

from typing import Protocol, TypedDict, Literal, runtime_checkable


class Message(TypedDict):
    """A single message in a conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol that all model backends must implement.
    
    Backends wrap LLM clients and provide a unified interface for:
    - Sending chat messages and getting responses
    - Counting tokens for a list of messages
    """
    
    def chat(self, messages: list[Message]) -> str:
        """Send messages to the model and get a response.
        
        Args:
            messages: List of conversation messages.
            
        Returns:
            The model's response text.
        """
        ...
    
    def count_tokens(self, messages: list[Message]) -> int:
        """Count tokens in a list of messages.
        
        Args:
            messages: List of messages to count tokens for.
            
        Returns:
            Estimated token count.
        """
        ...
    
    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        ...
