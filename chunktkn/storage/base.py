"""Base protocol for storage backends."""

from typing import Protocol, runtime_checkable

from chunktkn.backends.base import Message


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol that all storage backends must implement.
    
    Storage backends handle persistence of conversation context
    and chunked summaries.
    """
    
    def get_context(self) -> list[Message]:
        """Get the current conversation context.
        
        Returns:
            List of messages in the current context.
        """
        ...
    
    def set_context(self, messages: list[Message]) -> None:
        """Replace the current context with new messages.
        
        Args:
            messages: New context to store.
        """
        ...
    
    def append_message(self, message: Message) -> None:
        """Append a single message to the context.
        
        Args:
            message: Message to append.
        """
        ...
    
    def get_chunked_context(self) -> list[Message] | None:
        """Get the most recent chunked/summarized context.
        
        Returns:
            The chunked context if available, None otherwise.
        """
        ...
    
    def set_chunked_context(self, messages: list[Message]) -> None:
        """Store a chunked/summarized context.
        
        Args:
            messages: The summarized context to store.
        """
        ...
    
    def clear(self) -> None:
        """Clear all stored context."""
        ...
