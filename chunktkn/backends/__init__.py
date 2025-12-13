"""Backend implementations for different LLM providers."""

from chunktkn.backends.base import ModelBackend, Message

__all__ = ["ModelBackend", "Message"]

# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "OpenAIChatBackend":
        from chunktkn.backends.openai import OpenAIChatBackend
        return OpenAIChatBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
