"""Tests for system prompt separation (user vs summarization)."""

import pytest
from unittest.mock import Mock

from chunktkn import ContextChunker, ChunkConfig
from chunktkn.backends.base import Message


class MockBackend:
    """Mock backend that tracks system prompts used."""
    
    def __init__(self):
        self._call_count = 0
        self._calls_history: list[list[Message]] = []
        self._model = "mock-model"
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def chat(self, messages: list[Message]) -> str:
        """Track what messages are sent and return mock response."""
        self._calls_history.append(messages)
        self._call_count += 1
        
        # Check if this is a summarization call
        is_summarization = any("context compression" in str(m.get("content", "")).lower() 
                              for m in messages)
        
        if is_summarization:
            return "## Context Summary\nMock summary of conversation"
        else:
            return f"Mock response {self._call_count}"
    
    def count_tokens(self, messages: list[Message]) -> int:
        # Simple estimation
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4
    
    def get_last_call(self) -> list[Message]:
        """Get messages from the last call."""
        return self._calls_history[-1] if self._calls_history else []
    
    def get_system_prompts_from_call(self, call_index: int = -1) -> list[str]:
        """Extract system prompts from a specific call."""
        if not self._calls_history:
            return []
        messages = self._calls_history[call_index]
        return [m["content"] for m in messages if m["role"] == "system"]


class TestSystemPromptSeparation:
    """Test that user and summarization system prompts are properly separated."""
    
    def test_user_system_prompt_in_regular_chat(self):
        """Test that user_system_prompt appears in regular chat calls."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=100000,  # High trigger to avoid chunking
            user_system_prompt="You are a real estate agent",
            summarization_system_prompt="Be concise",
        )
        chunker = ContextChunker(backend, config)
        
        # Make a chat call
        response = chunker.chat("What's the best neighborhood?")
        
        # Check that user system prompt was used
        last_call = backend.get_last_call()
        system_prompts = [m["content"] for m in last_call if m["role"] == "system"]
        
        assert len(system_prompts) == 1
        assert system_prompts[0] == "You are a real estate agent"
        assert "Be concise" not in system_prompts[0]
    
    def test_summarization_prompt_only_in_chunking(self):
        """Test that summarization_system_prompt only appears during chunking."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=50,  # Low trigger to force chunking
            target_summary_tokens=10,
            user_system_prompt="You are a helpful assistant",
            summarization_system_prompt="Be extremely concise in summaries",
        )
        chunker = ContextChunker(backend, config)
        
        # Make multiple calls to trigger chunking
        for i in range(3):
            chunker.chat(f"This is a long message number {i} " * 20)
        
        # Find the summarization call
        summarization_call = None
        for call in backend._calls_history:
            if any("context compression" in str(m.get("content", "")).lower() 
                  for m in call):
                summarization_call = call
                break
        
        assert summarization_call is not None, "Chunking should have occurred"
        
        # Check summarization call has the summarization system prompt
        system_prompts = [m["content"] for m in summarization_call if m["role"] == "system"]
        assert any("Be extremely concise" in prompt for prompt in system_prompts)
        
        # Check that user system prompt is NOT in summarization call
        assert not any("helpful assistant" in prompt for prompt in system_prompts)
    
    def test_no_user_prompt_means_no_system_message(self):
        """Test that no user_system_prompt means no system message in chat."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=100000,
            user_system_prompt=None,  # No user prompt
            summarization_system_prompt="Be concise",
        )
        chunker = ContextChunker(backend, config)
        
        response = chunker.chat("Hello")
        
        last_call = backend.get_last_call()
        system_prompts = [m["content"] for m in last_call if m["role"] == "system"]
        
        assert len(system_prompts) == 0, "No system prompt should be in regular chat"
    
    def test_backward_compatibility_system_prompt(self):
        """Test that old 'system_prompt' field maps to user_system_prompt."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=100000,
            system_prompt="You are a teacher",  # Old field
        )
        
        # Should map to user_system_prompt
        assert config.user_system_prompt == "You are a teacher"
        
        chunker = ContextChunker(backend, config)
        response = chunker.chat("Teach me math")
        
        last_call = backend.get_last_call()
        system_prompts = [m["content"] for m in last_call if m["role"] == "system"]
        
        assert len(system_prompts) == 1
        assert system_prompts[0] == "You are a teacher"
    
    def test_cli_dev_mode_scenario(self):
        """Test CLI dev mode scenario: no user prompt, summarization prompt for chunking."""
        backend = MockBackend()
        
        # This is how CLI dev mode should be configured
        config = ChunkConfig(
            chunk_trigger_tokens=50,
            target_summary_tokens=10,
            user_system_prompt=None,  # No user prompt in dev testing
            summarization_system_prompt="You are code-focused. Prioritize code in summaries.",
        )
        chunker = ContextChunker(backend, config)
        
        # Regular chat should have NO system prompt
        response1 = chunker.chat("What is Python?")
        call1 = backend._calls_history[0]
        system_prompts_1 = [m["content"] for m in call1 if m["role"] == "system"]
        assert len(system_prompts_1) == 0, "Dev mode should have no user system prompt"
        
        # Force chunking
        for i in range(5):
            chunker.chat(f"Long message {i} " * 50)
        
        # Find summarization call
        summarization_call = None
        for call in backend._calls_history:
            if any("context compression" in str(m.get("content", "")).lower() 
                  for m in call):
                summarization_call = call
                break
        
        assert summarization_call is not None
        system_prompts_summ = [m["content"] for m in summarization_call if m["role"] == "system"]
        assert any("code-focused" in prompt.lower() for prompt in system_prompts_summ)
    
    def test_production_scenario(self):
        """Test production scenario: user prompt for chat, optional summarization prompt."""
        backend = MockBackend()
        
        # Production configuration
        config = ChunkConfig(
            chunk_trigger_tokens=50,
            target_summary_tokens=10,
            user_system_prompt="You are a real estate expert specializing in NYC properties",
            summarization_system_prompt="Preserve property details and numbers exactly",
        )
        chunker = ContextChunker(backend, config)
        
        # Regular chat should have user system prompt
        response = chunker.chat("Tell me about Brooklyn")
        call1 = backend._calls_history[0]
        system_prompts_1 = [m["content"] for m in call1 if m["role"] == "system"]
        assert len(system_prompts_1) == 1
        assert "real estate expert" in system_prompts_1[0]
        assert "NYC properties" in system_prompts_1[0]
        
        # Force chunking
        for i in range(5):
            chunker.chat(f"What about neighborhood {i}? " * 50)
        
        # After chunking, user prompt should still be there
        response_after = chunker.chat("Any other suggestions?")
        last_call = backend.get_last_call()
        
        # The context should have user system prompt preserved
        context = chunker.get_context()
        system_messages = [m for m in context if m["role"] == "system"]
        
        # Should have: [user_system_prompt, summary]
        assert len(system_messages) >= 1
        assert any("real estate expert" in m["content"] for m in system_messages)


class TestSystemPromptPreservation:
    """Test that user system prompt is preserved through chunking."""
    
    def test_user_prompt_preserved_after_chunking(self):
        """Verify user system prompt stays in context after chunking."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=50,
            target_summary_tokens=10,
            user_system_prompt="You are a Python expert",
            keep_last_n=2,
        )
        chunker = ContextChunker(backend, config)
        
        # Add messages to trigger chunking
        for i in range(10):
            chunker.chat(f"Question {i} " * 20)
        
        # Check context after chunking
        context = chunker.get_context()
        
        # First message should be the user system prompt
        assert context[0]["role"] == "system"
        assert context[0]["content"] == "You are a Python expert"
        
        # Should also have summary (as system message) and last N messages
        system_messages = [m for m in context if m["role"] == "system"]
        assert len(system_messages) >= 2  # user prompt + summary

