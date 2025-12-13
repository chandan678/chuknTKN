"""Tests for token counting."""

import pytest

from chunktkn.token_counter import count_tokens_tiktoken, count_tokens_text
from chunktkn.backends.base import Message


class TestTokenCounter:
    """Tests for token counting functions."""
    
    def test_count_tokens_text_simple(self):
        """Test counting tokens in simple text."""
        count = count_tokens_text("Hello, world!")
        
        # Should be a small positive number
        assert count > 0
        assert count < 10
    
    def test_count_tokens_text_longer(self):
        """Test that longer text has more tokens."""
        short = count_tokens_text("Hi")
        long = count_tokens_text("This is a much longer piece of text that should have more tokens.")
        
        assert long > short
    
    def test_count_tokens_messages(self):
        """Test counting tokens in messages."""
        messages: list[Message] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        count = count_tokens_tiktoken(messages)
        
        assert count > 0
    
    def test_count_tokens_empty(self):
        """Test counting tokens in empty messages."""
        messages: list[Message] = []
        
        count = count_tokens_tiktoken(messages)
        
        # Should just be the priming tokens
        assert count == 3
    
    def test_model_fallback(self):
        """Test that unknown models fall back to cl100k_base."""
        messages: list[Message] = [
            {"role": "user", "content": "Test message"},
        ]
        
        # Should not raise even with unknown model
        count = count_tokens_tiktoken(messages, model="unknown-model-xyz")
        
        assert count > 0
