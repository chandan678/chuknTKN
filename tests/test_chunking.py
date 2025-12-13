"""Tests for ContextChunker."""

import pytest
from unittest.mock import Mock, MagicMock

from chunktkn import ContextChunker, ChunkConfig, ChunkMode, MemoryStorage, DevStorage
from chunktkn.backends.base import Message


class MockBackend:
    """Mock backend for testing."""
    
    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or ["Mock response"]
        self._call_count = 0
        self._model = "mock-model"
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def chat(self, messages: list[Message]) -> str:
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return response
    
    def count_tokens(self, messages: list[Message]) -> int:
        # Simple token estimation: ~4 chars per token
        total_chars = sum(len(m["content"]) for m in messages)
        return total_chars // 4


class TestContextChunker:
    """Tests for ContextChunker class."""
    
    def test_basic_chat(self):
        """Test basic chat functionality."""
        backend = MockBackend(["Hello there!"])
        config = ChunkConfig(chunk_trigger_tokens=10000, target_summary_tokens=500)
        chunker = ContextChunker(backend, config)
        
        response = chunker.chat("Hello")
        
        assert response == "Hello there!"
        assert chunker.chunk_count == 0
    
    def test_context_accumulation(self):
        """Test that context accumulates across messages."""
        backend = MockBackend(["Response 1", "Response 2"])
        config = ChunkConfig(chunk_trigger_tokens=10000, target_summary_tokens=500)
        chunker = ContextChunker(backend, config)
        
        chunker.chat("First message")
        chunker.chat("Second message")
        
        context = chunker.get_context()
        
        assert len(context) == 4  # 2 user + 2 assistant
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"
    
    def test_system_prompt(self):
        """Test that system prompt is added to context."""
        backend = MockBackend(["Response"])
        config = ChunkConfig(
            chunk_trigger_tokens=10000,
            target_summary_tokens=500,
            system_prompt="You are helpful.",
        )
        chunker = ContextChunker(backend, config)
        
        context = chunker.get_context()
        
        assert len(context) == 1
        assert context[0]["role"] == "system"
        assert context[0]["content"] == "You are helpful."
    
    def test_stats_tracking(self):
        """Test that stats are tracked correctly."""
        backend = MockBackend(["Response"])
        config = ChunkConfig(chunk_trigger_tokens=10000, target_summary_tokens=500)
        chunker = ContextChunker(backend, config)
        
        chunker.chat("Test message")
        stats = chunker.get_stats()
        
        assert stats["chunk_count"] == 0
        assert stats["total_tokens_processed"] > 0
        assert stats["chunk_trigger"] == 10000
    
    def test_reset(self):
        """Test that reset clears context."""
        backend = MockBackend(["Response"])
        config = ChunkConfig(chunk_trigger_tokens=10000, target_summary_tokens=500)
        chunker = ContextChunker(backend, config)
        
        chunker.chat("Test message")
        chunker.reset()
        
        assert chunker.get_context() == []
        assert chunker.chunk_count == 0


class TestChunkingBehavior:
    """Tests for chunking behavior - system prompt preservation and last N."""
    
    def test_keep_last_n_default(self):
        """Test that keep_last_n defaults to 10."""
        config = ChunkConfig()
        assert config.keep_last_n == 10
    
    def test_keep_last_n_custom(self):
        """Test custom keep_last_n value."""
        config = ChunkConfig(keep_last_n=5)
        assert config.keep_last_n == 5


class TestChunkConfig:
    """Tests for ChunkConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ChunkConfig()
        
        assert config.chunk_trigger_tokens == 50_000
        assert config.target_summary_tokens == 2_000
        assert config.preserve_code is True
        assert config.mode == ChunkMode.MANAGER
    
    def test_observer_mode_not_implemented(self):
        """Test that observer mode raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            ChunkConfig(mode=ChunkMode.OBSERVER)
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ChunkConfig(
            chunk_trigger_tokens=100_000,
            target_summary_tokens=5_000,
            preserve_code=False,
        )
        
        assert config.chunk_trigger_tokens == 100_000
        assert config.target_summary_tokens == 5_000
        assert config.preserve_code is False
    
    def test_validation_positive_trigger(self):
        """Test that chunk_trigger_tokens must be positive."""
        with pytest.raises(ValueError):
            ChunkConfig(chunk_trigger_tokens=0)
    
    def test_validation_summary_less_than_trigger(self):
        """Test that target_summary_tokens must be less than trigger."""
        with pytest.raises(ValueError):
            ChunkConfig(
                chunk_trigger_tokens=1000,
                target_summary_tokens=2000,
            )


class TestMemoryStorage:
    """Tests for MemoryStorage class."""
    
    def test_append_and_get(self):
        """Test appending and retrieving messages."""
        storage = MemoryStorage()
        
        storage.append_message({"role": "user", "content": "Hello"})
        context = storage.get_context()
        
        assert len(context) == 1
        assert context[0]["content"] == "Hello"
    
    def test_set_context(self):
        """Test setting entire context."""
        storage = MemoryStorage()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        storage.set_context(messages)
        
        assert storage.get_context() == messages
    
    def test_chunked_context(self):
        """Test chunked context storage."""
        storage = MemoryStorage()
        
        assert storage.get_chunked_context() is None
        
        summary = [{"role": "system", "content": "Summary"}]
        storage.set_chunked_context(summary)
        
        assert storage.get_chunked_context() == summary
    
    def test_clear(self):
        """Test clearing storage."""
        storage = MemoryStorage()
        
        storage.append_message({"role": "user", "content": "Hello"})
        storage.set_chunked_context([{"role": "system", "content": "Sum"}])
        storage.clear()
        
        assert storage.get_context() == []
        assert storage.get_chunked_context() is None


class TestDevStorage:
    """Tests for DevStorage class."""
    
    def test_record_chunk(self):
        """Test recording chunk history."""
        storage = DevStorage()
        
        context = [{"role": "user", "content": "Hello"}]
        chunked = [{"role": "system", "content": "Summary"}]
        
        record = storage.record_chunk(
            tokens=1000,
            context=context,
            chunked_context=chunked,
        )
        
        assert record.id == 1
        assert record.tokens == 1000
        assert len(storage.get_chunk_history()) == 1
    
    def test_get_chunk_by_id(self):
        """Test retrieving chunk by ID."""
        storage = DevStorage()
        
        storage.record_chunk(100, [], [])
        storage.record_chunk(200, [], [])
        
        chunk = storage.get_chunk_by_id(2)
        
        assert chunk is not None
        assert chunk.tokens == 200
    
    def test_rewind_to_chunk(self):
        """Test rewinding to a previous chunk."""
        storage = DevStorage()
        
        chunked = [{"role": "system", "content": "State at chunk 1"}]
        storage.record_chunk(100, [], chunked)
        
        storage.append_message({"role": "user", "content": "New message"})
        
        success = storage.rewind_to_chunk(1)
        
        assert success is True
        assert storage.get_context() == chunked
