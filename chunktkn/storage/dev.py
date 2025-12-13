"""Development storage with full chunk history.

Stores complete history of all chunks for testing and debugging.
Used with --dev mode to inspect and potentially rewind to any chunk.
"""

from dataclasses import dataclass, field
from typing import Any
import json
from pathlib import Path

from chunktkn.backends.base import Message


@dataclass
class ChunkRecord:
    """A single chunk record with full context snapshot.
    
    Attributes:
        id: Unique chunk identifier (auto-increment).
        tokens: Token count at time of chunking.
        context: Full context before chunking.
        chunked_context: Summarized context after chunking.
    """
    id: int
    tokens: int
    context: list[Message]
    chunked_context: list[Message]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "tokens": self.tokens,
            "context": self.context,
            "chunked_context": self.chunked_context,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            tokens=data["tokens"],
            context=data["context"],
            chunked_context=data["chunked_context"],
        )


class DevStorage:
    """Development storage with full chunk history.
    
    Stores:
        - context: Current conversation messages
        - chunked_context: Most recent summarized context
        - chunks: List of all chunk records with full history
    
    This storage is used in --dev mode to:
        - Inspect any previous chunk state
        - Test future rewind functionality
        - Debug chunking behavior
    
    Example:
        >>> storage = DevStorage()
        >>> storage.append_message({"role": "user", "content": "Hello"})
        >>> # After chunking...
        >>> storage.get_chunk_history()
        [ChunkRecord(id=1, tokens=52000, ...)]
    """
    
    def __init__(self, session_name: str | None = None) -> None:
        """Initialize dev storage.
        
        Args:
            session_name: Optional name for this session (for file export).
        """
        self._context: list[Message] = []
        self._chunked_context: list[Message] | None = None
        self._chunks: list[ChunkRecord] = []
        self._next_chunk_id: int = 1
        self._session_name = session_name or "default"
    
    def get_context(self) -> list[Message]:
        """Get the current conversation context."""
        return self._context.copy()
    
    def set_context(self, messages: list[Message]) -> None:
        """Replace the current context with new messages."""
        self._context = list(messages)
    
    def append_message(self, message: Message) -> None:
        """Append a single message to the context."""
        self._context.append(message)
    
    def get_chunked_context(self) -> list[Message] | None:
        """Get the most recent chunked/summarized context."""
        return self._chunked_context.copy() if self._chunked_context else None
    
    def set_chunked_context(self, messages: list[Message]) -> None:
        """Store a chunked/summarized context."""
        self._chunked_context = list(messages)
    
    def clear(self) -> None:
        """Clear current context (preserves chunk history)."""
        self._context = []
        self._chunked_context = None
    
    def clear_all(self) -> None:
        """Clear everything including chunk history."""
        self.clear()
        self._chunks = []
        self._next_chunk_id = 1
    
    # --- Dev-specific methods ---
    
    def record_chunk(self, tokens: int, context: list[Message], chunked_context: list[Message]) -> ChunkRecord:
        """Record a chunking event with full context snapshot.
        
        Args:
            tokens: Token count at time of chunking.
            context: Full context before chunking.
            chunked_context: Summarized context after chunking.
            
        Returns:
            The created ChunkRecord.
        """
        record = ChunkRecord(
            id=self._next_chunk_id,
            tokens=tokens,
            context=list(context),
            chunked_context=list(chunked_context),
        )
        self._chunks.append(record)
        self._next_chunk_id += 1
        return record
    
    def get_chunk_history(self) -> list[ChunkRecord]:
        """Get all chunk records.
        
        Returns:
            List of all ChunkRecord objects.
        """
        return self._chunks.copy()
    
    def get_chunk_by_id(self, chunk_id: int) -> ChunkRecord | None:
        """Get a specific chunk by ID.
        
        Args:
            chunk_id: The chunk ID to find.
            
        Returns:
            The ChunkRecord if found, None otherwise.
        """
        for chunk in self._chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    def rewind_to_chunk(self, chunk_id: int) -> bool:
        """Rewind context to a specific chunk state.
        
        This sets the current context to the chunked_context of the
        specified chunk, effectively "rewinding" to that point.
        
        Args:
            chunk_id: The chunk ID to rewind to.
            
        Returns:
            True if successful, False if chunk not found.
        """
        chunk = self.get_chunk_by_id(chunk_id)
        if chunk is None:
            return False
        
        self._context = list(chunk.chunked_context)
        self._chunked_context = list(chunk.chunked_context)
        return True
    
    def to_dict(self) -> dict[str, Any]:
        """Export full storage state as a dictionary."""
        return {
            "session_name": self._session_name,
            "context": self._context,
            "chunked_context": self._chunked_context,
            "chunks": [c.to_dict() for c in self._chunks],
        }
    
    def save_to_file(self, path: str | Path | None = None) -> Path:
        """Save storage state to a JSON file.
        
        Args:
            path: File path. If None, uses ~/.chunktkn/dev/<session_name>.json
            
        Returns:
            Path to the saved file.
        """
        if path is None:
            base_dir = Path.home() / ".chunktkn" / "dev"
            base_dir.mkdir(parents=True, exist_ok=True)
            path = base_dir / f"{self._session_name}.json"
        else:
            path = Path(path)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return path
    
    @classmethod
    def load_from_file(cls, path: str | Path) -> "DevStorage":
        """Load storage state from a JSON file.
        
        Args:
            path: Path to the JSON file.
            
        Returns:
            DevStorage instance with loaded state.
        """
        with open(path) as f:
            data = json.load(f)
        
        storage = cls(session_name=data.get("session_name", "default"))
        storage._context = data.get("context", [])
        storage._chunked_context = data.get("chunked_context")
        storage._chunks = [ChunkRecord.from_dict(c) for c in data.get("chunks", [])]
        
        if storage._chunks:
            storage._next_chunk_id = max(c.id for c in storage._chunks) + 1
        
        return storage
