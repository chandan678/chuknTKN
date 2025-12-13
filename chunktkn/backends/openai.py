"""OpenAI backend implementation."""

from typing import TYPE_CHECKING

from chunktkn.backends.base import Message
from chunktkn.token_counter import count_tokens_tiktoken

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIChatBackend:
    """Backend for OpenAI API models.
    
    Wraps an OpenAI client and provides the ModelBackend interface.
    
    Example:
        >>> from openai import OpenAI
        >>> from chunktkn.backends.openai import OpenAIChatBackend
        >>> 
        >>> client = OpenAI()
        >>> backend = OpenAIChatBackend(client, model="gpt-4.1")
    """
    
    def __init__(
        self,
        client: "OpenAI",
        model: str = "gpt-4.1",
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the OpenAI backend.
        
        Args:
            client: An initialized OpenAI client.
            model: Model name to use for completions.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response (None for model default).
        """
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
    
    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model
    
    def chat(self, messages: list[Message]) -> str:
        """Send messages to OpenAI and get a response.
        
        Args:
            messages: List of conversation messages.
            
        Returns:
            The model's response text.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        
        content = response.choices[0].message.content
        return content if content is not None else ""
    
    def count_tokens(self, messages: list[Message]) -> int:
        """Count tokens using tiktoken.
        
        Args:
            messages: List of messages to count tokens for.
            
        Returns:
            Estimated token count.
        """
        return count_tokens_tiktoken(messages, self._model)
