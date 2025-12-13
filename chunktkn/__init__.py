"""chunkTKN - Client-side context management for LLMs.

Long conversations without context decay.
A client-side middleware that automatically summarizes, compresses,
and refreshes LLM context—without forcing users to start a new chat.

Example:
    >>> from openai import OpenAI
    >>> from chunktkn import ContextChunker, OpenAIChatBackend, ChunkConfig
    >>> 
    >>> client = OpenAI()
    >>> backend = OpenAIChatBackend(client, model="gpt-4.1")
    >>> config = ChunkConfig(chunk_trigger_tokens=50_000)
    >>> 
    >>> chunker = ContextChunker(backend, config)
    >>> reply = chunker.chat("Help me design a distributed system")
"""

from chunktkn.chunker import ContextChunker
from chunktkn.config import ChunkConfig, ChunkMode
from chunktkn.backends.base import ModelBackend, Message
from chunktkn.storage.memory import MemoryStorage
from chunktkn.storage.dev import DevStorage

__version__ = "0.1.0"

__all__ = [
    # Core
    "ContextChunker",
    "ChunkConfig",
    "ChunkMode",
    # Backends
    "ModelBackend",
    "Message",
    # Storage
    "MemoryStorage",
    "DevStorage",
    # Version
    "__version__",
]


# Lazy import for OpenAIChatBackend to avoid requiring openai as dependency
def __getattr__(name: str):
    if name == "OpenAIChatBackend":
        from chunktkn.backends.openai import OpenAIChatBackend
        return OpenAIChatBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
