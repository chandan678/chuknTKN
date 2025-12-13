"""Configuration dataclasses for chunkTKN."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ChunkMode(Enum):
    """Operating mode for ContextChunker.
    
    MANAGER: Owns history, full chunk/replace. For raw Chat Completions API.
             chunkTKN calls the model directly and manages all state.
    
    OBSERVER: (Future) Watches external conversations, acts via backend ops.
              For managed services like OpenAI Threads, Assistants API.
              chunkTKN observes and triggers actions (delete/inject/new thread).
    """
    MANAGER = "manager"
    OBSERVER = "observer"


@dataclass
class ChunkConfig:
    """Configuration for context chunking behavior.
    
    Attributes:
        chunk_trigger_tokens: Token count that triggers chunking.
        target_summary_tokens: Target size for compressed context.
        max_history_tokens: Optional hard limit on history size.
        preserve_code: Whether to preserve code blocks verbatim during summarization.
        code_detection_mode: How to detect code blocks ("heuristic" or "markdown").
        system_prompt: Optional system prompt for the conversation.
        mode: Operating mode (MANAGER or OBSERVER). V0 only supports MANAGER.
        keep_last_n: Number of recent messages to preserve verbatim during chunking.
        max_old_messages: (Observer mode) Messages to keep after cleanup.
    """
    
    chunk_trigger_tokens: int = 50_000
    target_summary_tokens: int = 2_000
    max_history_tokens: int | None = None
    preserve_code: bool = True
    code_detection_mode: Literal["heuristic", "markdown"] = "heuristic"
    system_prompt: str | None = None
    
    # Mode selection (V0: only MANAGER is implemented)
    mode: ChunkMode = ChunkMode.MANAGER
    
    # Chunking behavior
    keep_last_n: int = 10  # Keep last N messages verbatim during chunking
    
    # Observer mode settings (for future use)
    max_old_messages: int = 50  # Messages to keep when cleaning up in observer mode
    
    def __post_init__(self) -> None:
        if self.chunk_trigger_tokens <= 0:
            raise ValueError("chunk_trigger_tokens must be positive")
        if self.target_summary_tokens <= 0:
            raise ValueError("target_summary_tokens must be positive")
        if self.target_summary_tokens >= self.chunk_trigger_tokens:
            raise ValueError("target_summary_tokens must be less than chunk_trigger_tokens")
        
        # V0: Only MANAGER mode is supported
        if self.mode == ChunkMode.OBSERVER:
            raise NotImplementedError(
                "Observer mode is not yet implemented. "
                "Use ChunkMode.MANAGER for V0. Observer mode coming in V1."
            )
