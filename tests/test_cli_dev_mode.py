"""Tests for CLI dev mode behavior."""

import pytest
from unittest.mock import Mock

from chunktkn import ContextChunker, ChunkConfig
from chunktkn.storage.dev import DevStorage
from chunktkn.backends.base import Message


class MockBackend:
    """Mock backend for testing."""
    
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


class TestDevModeParallelChunkers:
    """Test that dev mode runs parallel chunkers correctly."""
    
    def test_all_chunkers_receive_same_messages(self):
        """Test that all chunkers track the same conversation."""
        backend = MockBackend()
        
        # Create 3 chunkers with different summarization prompts (like dev mode)
        chunkers = {}
        for name, prompt in [
            ("concise", "Be concise"),
            ("detailed", "Be detailed"),  
            ("code_focused", "Be code-focused")
        ]:
            config = ChunkConfig(
                chunk_trigger_tokens=100000,  # High to avoid chunking
                user_system_prompt=None,  # No user prompt in dev mode
                summarization_system_prompt=prompt,
            )
            storage = DevStorage(session_name=f"dev_{name}")
            chunker = ContextChunker(backend, config, storage)
            chunkers[name] = chunker
        
        # Send same message to all chunkers
        user_message = "What is Python?"
        responses = {}
        for name, chunker in chunkers.items():
            response = chunker.chat(user_message)
            responses[name] = response
        
        # All chunkers should have received the same message
        # And should return the same response (from mock backend)
        assert len(responses) == 3
        assert all(resp.startswith("Mock response") for resp in responses.values())
        
        # Check that all chunkers have the same context
        contexts = {name: chunker.get_context() for name, chunker in chunkers.items()}
        
        # All should have 2 messages: user + assistant (no system prompt)
        for context in contexts.values():
            assert len(context) == 2
            assert context[0]["role"] == "user"
            assert context[0]["content"] == "What is Python?"
            assert context[1]["role"] == "assistant"
    
    def test_chunking_differences_visible(self):
        """Test that when chunking happens, different summarization prompts create different summaries."""
        # This is what dev mode is for - comparing summarization strategies
        backend = MockBackend()
        
        chunkers = {}
        chunk_events = {}
        
        for name, prompt in [
            ("concise", "Be extremely concise"),
            ("detailed", "Be very detailed"),
        ]:
            config = ChunkConfig(
                chunk_trigger_tokens=50,  # Low trigger
                target_summary_tokens=10,
                user_system_prompt=None,
                summarization_system_prompt=prompt,
            )
            storage = DevStorage(session_name=f"dev_{name}")
            
            # Track chunking events
            chunk_events[name] = []
            def make_callback(chunker_name):
                def callback(before, after):
                    chunk_events[chunker_name].append((before, after))
                return callback
            
            chunker = ContextChunker(backend, config, storage, on_chunk=make_callback(name))
            chunkers[name] = chunker
        
        # Add messages to trigger chunking
        for i in range(5):
            for chunker in chunkers.values():
                chunker.chat(f"Message {i} " * 20)
        
        # Both should have chunked
        assert len(chunk_events["concise"]) > 0
        assert len(chunk_events["detailed"]) > 0
        
        # Both should have summaries in their contexts
        for name, chunker in chunkers.items():
            context = chunker.get_context()
            # Should have at least one system message (the summary)
            system_msgs = [m for m in context if m["role"] == "system"]
            assert len(system_msgs) >= 1, f"{name} should have summary after chunking"
    
    def test_dev_mode_no_system_prompt_pollution(self):
        """Test that summarization prompts don't appear in regular chat."""
        backend = MockBackend()
        
        config = ChunkConfig(
            chunk_trigger_tokens=100000,
            user_system_prompt=None,  # Dev mode: no user prompt
            summarization_system_prompt="You are code-focused",  # Only for summarization
        )
        storage = DevStorage(session_name="dev_test")
        chunker = ContextChunker(backend, config, storage)
        
        # Make a regular chat call
        response = chunker.chat("Hello")
        
        # Check that NO system prompt was sent
        last_call = backend._calls_history[-1]
        system_prompts = [m["content"] for m in last_call if m["role"] == "system"]
        
        assert len(system_prompts) == 0, "Dev mode should have no system prompt in regular chat"
        assert "code-focused" not in str(last_call), "Summarization prompt should not appear in chat"
