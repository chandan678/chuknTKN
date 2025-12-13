"""Token counting utilities."""

import tiktoken

from chunktkn.backends.base import Message


# Cache for tokenizer encodings
_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_encoding(model: str) -> tiktoken.Encoding:
    """Get or create a tiktoken encoding for a model.
    
    Args:
        model: Model name (e.g., "gpt-4", "gpt-3.5-turbo").
        
    Returns:
        Tiktoken encoding for the model.
    """
    if model not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models (GPT-4, GPT-3.5-turbo family)
            _ENCODING_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    
    return _ENCODING_CACHE[model]


def count_tokens_tiktoken(messages: list[Message], model: str = "gpt-4") -> int:
    """Count tokens in messages using tiktoken.
    
    This follows OpenAI's token counting guidelines for chat models.
    
    Args:
        messages: List of messages to count.
        model: Model name for encoding selection.
        
    Returns:
        Total token count.
    """
    encoding = _get_encoding(model)
    
    # Token overhead per message (role, content separators, etc.)
    # These values are approximate and based on OpenAI's documentation
    tokens_per_message = 3  # <|start|>role<|sep|>content<|end|>
    tokens_per_name = 1  # If name is present
    
    total = 0
    for message in messages:
        total += tokens_per_message
        total += len(encoding.encode(message["content"]))
        total += len(encoding.encode(message["role"]))
    
    # Every reply is primed with <|start|>assistant<|message|>
    total += 3
    
    return total


def count_tokens_text(text: str, model: str = "gpt-4") -> int:
    """Count tokens in a single text string.
    
    Args:
        text: Text to count tokens for.
        model: Model name for encoding selection.
        
    Returns:
        Token count.
    """
    encoding = _get_encoding(model)
    return len(encoding.encode(text))
