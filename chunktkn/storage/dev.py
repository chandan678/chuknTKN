"""Development storage with full chunk history.

Stores complete history of all chunks for testing and debugging.
Used with --dev mode to inspect and potentially rewind to any chunk.

In dev mode, all interactions are automatically saved to JSON files
with session IDs for replay and analysis.
"""

from dataclasses import dataclass, field, asdict
from typing import Any
import json
from pathlib import Path
from datetime import datetime
import uuid

from chunktkn.backends.base import Message


@dataclass
class ChunkRecord:
    """A single chunk record with full context snapshot.
    
    Attributes:
        id: Unique chunk identifier (auto-increment).
        timestamp: When this chunk occurred.
        tokens: Token count at time of chunking.
        context: Full context before chunking.
        chunked_context: Summarized context after chunking.
    """
    id: int
    timestamp: str
    tokens: int
    context: list[Message]
    chunked_context: list[Message]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "tokens": self.tokens,
            "context": self.context,
            "chunked_context": self.chunked_context,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=data.get("timestamp", ""),
            tokens=data["tokens"],
            context=data["context"],
            chunked_context=data["chunked_context"],
        )


@dataclass
class InteractionRecord:
    """A single chat interaction (user + assistant).
    
    Attributes:
        id: Unique interaction identifier (auto-increment).
        timestamp: When this interaction occurred.
        user_message: User's message.
        assistant_response: Assistant's response.
        tokens_total: Total tokens in context after this interaction.
    """
    id: int
    timestamp: str
    user_message: str
    assistant_response: str
    tokens_total: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionRecord":
        """Create from dictionary."""
        return cls(**data)


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
    
    def __init__(self, session_name: str | None = None, auto_save: bool = True, max_chunks: int | None = None) -> None:
        """Initialize dev storage with session tracking.
        
        Args:
            session_name: Optional name for this session (auto-generated if None).
            auto_save: Whether to automatically save interactions to JSON (default: True).
            max_chunks: Maximum number of chunks allowed (None = unlimited). Useful for limiting
                       chunking events during testing (e.g., max_chunks=3 stops after 3rd chunk).
        """
        self._context: list[Message] = []
        self._chunked_context: list[Message] | None = None
        self._chunks: list[ChunkRecord] = []
        self._interactions: list[InteractionRecord] = []
        self._next_chunk_id: int = 1
        self._next_interaction_id: int = 1
        self._max_chunks = max_chunks
        self._chunking_disabled = False  # Set to True when max_chunks reached
        
        # Generate session ID and setup file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        self._session_name = session_name or session_id
        self._session_id = session_id
        self._auto_save = auto_save
        
        # Create base directory and session folder in current working directory
        self._base_dir = Path.cwd() / "saved"
        self._session_dir = self._base_dir / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._session_file = self._session_dir / "session.json"
        
        # Register this session in the registry
        self._register_session()
        
        # Initialize session file
        if self._auto_save:
            self._save_session()
    
    def get_context(self) -> list[Message]:
        """Get the current conversation context."""
        return self._context.copy()
    
    def set_context(self, messages: list[Message]) -> None:
        """Replace the current context with new messages."""
        self._context = list(messages)
    
    def append_message(self, message: Message) -> None:
        """Append a single message to the context.
        
        Also tracks interactions and auto-saves if enabled.
        """
        self._context.append(message)
        
        # Track user-assistant pairs as interactions
        if self._auto_save and message["role"] == "assistant":
            # Find the last user message
            user_msg = ""
            for i in range(len(self._context) - 2, -1, -1):
                if self._context[i]["role"] == "user":
                    user_msg = self._context[i]["content"]
                    break
            
            # Record interaction
            interaction = InteractionRecord(
                id=self._next_interaction_id,
                timestamp=datetime.now().isoformat(),
                user_message=user_msg,
                assistant_response=message["content"],
                tokens_total=len(self._context),  # Approximate
            )
            self._interactions.append(interaction)
            self._next_interaction_id += 1
            
            # Auto-save session
            self._save_session()
    
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
    
    def record_chunk(self, tokens: int, context: list[Message], chunked_context: list[Message]) -> ChunkRecord | None:
        """Record a chunking event with full context snapshot.
        
        Args:
            tokens: Token count at time of chunking.
            context: Full context before chunking.
            chunked_context: Summarized context after chunking.
            
        Returns:
            The created ChunkRecord, or None if max_chunks limit reached.
        """
        # Check if max_chunks limit reached
        if self._max_chunks is not None and len(self._chunks) >= self._max_chunks:
            if not self._chunking_disabled:
                self._chunking_disabled = True
                print(f"\n⚠️  Max chunks limit ({self._max_chunks}) reached. No more chunking will occur.")
            return None
        
        record = ChunkRecord(
            id=self._next_chunk_id,
            timestamp=datetime.now().isoformat(),
            tokens=tokens,
            context=list(context),
            chunked_context=list(chunked_context),
        )
        self._chunks.append(record)
        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1
        
        # Save context snapshots to separate files
        self._save_chunk_files(chunk_id, context, chunked_context)
        
        # Auto-save session after chunking
        if self._auto_save:
            self._save_session()
        
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
    
    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id
    
    @property
    def session_file(self) -> Path:
        """Get the session file path."""
        return self._session_file
    
    @property
    def session_dir(self) -> Path:
        """Get the session directory path."""
        return self._session_dir
    
    @property
    def max_chunks_reached(self) -> bool:
        """Check if max_chunks limit has been reached."""
        return self._chunking_disabled
    
    def to_dict(self) -> dict[str, Any]:
        """Export full storage state as a dictionary."""
        return {
            "session_id": self._session_id,
            "session_name": self._session_name,
            "created_at": self._session_id.split("_")[0],  # Extract timestamp
            "max_chunks": self._max_chunks,
            "chunks_limit_reached": self._chunking_disabled,
            "context": self._context,
            "chunked_context": self._chunked_context,
            "interactions": [i.to_dict() for i in self._interactions],
            "chunks": [c.to_dict() for c in self._chunks],
            "metadata": {
                "total_interactions": len(self._interactions),
                "total_chunks": len(self._chunks),
                "current_context_size": len(self._context),
                "max_chunks_configured": self._max_chunks,
                "chunking_disabled": self._chunking_disabled,
            },
        }
    
    def _register_session(self) -> None:
        """Register this session in the global registry."""
        registry_file = self._base_dir / "sessions_registry.json"
        
        try:
            # Load existing registry
            if registry_file.exists():
                with open(registry_file) as f:
                    registry = json.load(f)
            else:
                registry = {"sessions": [], "total_sessions": 0}
            
            # Add this session
            session_entry = {
                "session_id": self._session_id,
                "session_name": self._session_name,
                "created_at": datetime.now().isoformat(),
                "session_dir": str(self._session_dir),
            }
            
            # Check if already registered (in case of reload)
            if not any(s["session_id"] == self._session_id for s in registry["sessions"]):
                registry["sessions"].append(session_entry)
                registry["total_sessions"] = len(registry["sessions"])
            
            # Save registry
            with open(registry_file, "w") as f:
                json.dump(registry, f, indent=2)
        except Exception:
            # Silent fail - registry is not critical
            pass
    
    def _save_chunk_files(self, chunk_id: int, context: list[Message], chunked_context: list[Message]) -> None:
        """Save chunk context snapshots to separate files.
        
        Creates:
            - context_{chunk_id}.json: Original context before chunking
            - context_{chunk_id}_chunked.json: Summarized context after chunking
        """
        try:
            # Save original context
            context_file = self._session_dir / f"context_{chunk_id}.json"
            with open(context_file, "w") as f:
                json.dump({
                    "chunk_id": chunk_id,
                    "timestamp": datetime.now().isoformat(),
                    "type": "original_context",
                    "messages": context,
                    "message_count": len(context),
                }, f, indent=2)
            
            # Save chunked context
            chunked_file = self._session_dir / f"context_{chunk_id}_chunked.json"
            with open(chunked_file, "w") as f:
                json.dump({
                    "chunk_id": chunk_id,
                    "timestamp": datetime.now().isoformat(),
                    "type": "chunked_context",
                    "messages": chunked_context,
                    "message_count": len(chunked_context),
                    "compression_ratio": f"{len(context)} → {len(chunked_context)} messages",
                }, f, indent=2)
        except Exception:
            # Silent fail - context files are supplementary
            pass
    
    def _save_session(self) -> None:
        """Save session to JSON file (internal method)."""
        try:
            with open(self._session_file, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            # Silent fail in case of I/O errors
            pass
    
    def save_to_file(self, path: str | Path | None = None) -> Path:
        """Save storage state to a JSON file.
        
        When path is None, saves to the session directory (same as auto-save).
        Use /save command to trigger manual save.
        
        Args:
            path: File path. If None, uses session directory session.json
            
        Returns:
            Path to the saved file.
        """
        if path is None:
            # Save to session directory (unified with auto-save)
            path = self._session_file
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
        
        max_chunks = data.get("max_chunks")
        storage = cls(session_name=data.get("session_name", "default"), auto_save=False, max_chunks=max_chunks)
        storage._session_id = data.get("session_id", storage._session_id)
        storage._context = data.get("context", [])
        storage._chunked_context = data.get("chunked_context")
        storage._interactions = [InteractionRecord.from_dict(i) for i in data.get("interactions", [])]
        storage._chunks = [ChunkRecord.from_dict(c) for c in data.get("chunks", [])]
        storage._chunking_disabled = data.get("chunks_limit_reached", False)
        
        if storage._chunks:
            storage._next_chunk_id = max(c.id for c in storage._chunks) + 1
        if storage._interactions:
            storage._next_interaction_id = max(i.id for i in storage._interactions) + 1
        
        return storage
    
    @staticmethod
    def get_all_sessions() -> list[dict[str, Any]]:
        """Get list of all sessions from the registry.
        
        Returns:
            List of session info dictionaries, or empty list if registry does not exist.
        """
        registry_file = Path.cwd() / "saved" / "sessions_registry.json"
        
        if not registry_file.exists():
            return []
        
        try:
            with open(registry_file) as f:
                registry = json.load(f)
            return registry.get("sessions", [])
        except Exception:
            return []
