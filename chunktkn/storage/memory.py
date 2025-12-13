"""In-memory storage for V0 production use.

Simple storage that keeps context and chunked_context in memory.
No persistence, no history — just current state.
"""

from chunktkn.backends.base import Message


class MemoryStorage:
    """In-memory storage backend for context.
    
    Stores:
        - context: Current conversation messages
        - chunked_context: Most recent summarized context (after chunking)
    
    This is the default storage for production V0.
    No persistence — data is lost when the process ends.
    
    Example:
        >>> storage = MemoryStorage()
        >>> storage.append_message({"role": "user", "content": "Hello"})
        >>> storage.get_context()
        [{"role": "user", "content": "Hello"}]
    """
    
    def __init__(self) -> None:
        """Initialize empty storage."""
        self._context: list[Message] = []
        self._chunked_context: list[Message] | None = None
    
    def get_context(self) -> list[Message]:
        """Get the current conversation context.
        
        Returns:
            List of messages in the current context.
        """
        return self._context.copy()
    
    def set_context(self, messages: list[Message]) -> None:
        """Replace the current context with new messages.
        
        Args:
            messages: New context to store.
        """
        self._context = list(messages)
    
    def append_message(self, message: Message) -> None:
        """Append a single message to the context.
        
        Args:
            message: Message to append.
        """
        self._context.append(message)
    
    def get_chunked_context(self) -> list[Message] | None:
        """Get the most recent chunked/summarized context.
        
        Returns:
            The chunked context if available, None otherwise.
        """
        return self._chunked_context.copy() if self._chunked_context else None
    
    def set_chunked_context(self, messages: list[Message]) -> None:
        """Store a chunked/summarized context.
        
        Args:
            messages: The summarized context to store.
        """
        self._chunked_context = list(messages)
    
    def clear(self) -> None:
        """Clear all stored context."""
        self._context = []
        self._chunked_context = None
    
    def to_dict(self) -> dict:
        """Export storage state as a dictionary.
        
        Useful for debugging or manual inspection.
        
        Returns:
            Dictionary with context and chunked_context.
        """
        return {
            "context": self._context,
            "chunked_context": self._chunked_context,
        }
