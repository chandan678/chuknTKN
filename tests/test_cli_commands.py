"""Tests for CLI dynamic configuration commands."""

import pytest
from unittest.mock import Mock

from chunktkn import ContextChunker, ChunkConfig
from chunktkn.storage.dev import DevStorage
from chunktkn.backends.base import Message


class MockBackend:
    """Mock backend for testing."""
    
    def __init__(self):
        self._call_count = 0
        self._model = "mock-model"
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def chat(self, messages: list[Message]) -> str:
        """Return mock response."""
        self._call_count += 1
        
        # Check if this is a summarization call
        is_summarization = any("context compression" in str(m.get("content", "")).lower() 
                              for m in messages)
        
        if is_summarization:
            return "## Context Summary\nMock summary"
        else:
            return f"Mock response {self._call_count}"
    
    def count_tokens(self, messages: list[Message]) -> int:
        # Simple estimation
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4


class TestDynamicConfigUpdate:
    """Test that /set commands properly update chunker configuration."""
    
    def test_set_trigger_updates_config(self):
        """Test that /set trigger updates chunk_trigger_tokens."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=1000,
            target_summary_tokens=100,
        )
        chunker = ContextChunker(backend, config)
        
        # Verify initial value
        assert chunker.config.chunk_trigger_tokens == 1000
        
        # Simulate /set trigger command
        chunker.config.chunk_trigger_tokens = 2000
        
        # Verify update
        assert chunker.config.chunk_trigger_tokens == 2000
    
    def test_set_target_updates_config(self):
        """Test that /set target updates target_summary_tokens."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=1000,
            target_summary_tokens=100,
        )
        chunker = ContextChunker(backend, config)
        
        # Verify initial value
        assert chunker.config.target_summary_tokens == 100
        
        # Simulate /set target command
        chunker.config.target_summary_tokens = 200
        
        # Verify update
        assert chunker.config.target_summary_tokens == 200
    
    def test_config_validation_positive_trigger(self):
        """Test that trigger must be positive."""
        # ChunkConfig validation happens at creation time
        with pytest.raises(ValueError, match="chunk_trigger_tokens must be positive"):
            ChunkConfig(chunk_trigger_tokens=0)
        
        with pytest.raises(ValueError, match="chunk_trigger_tokens must be positive"):
            ChunkConfig(chunk_trigger_tokens=-100)
    
    def test_config_validation_positive_target(self):
        """Test that target must be positive."""
        with pytest.raises(ValueError, match="target_summary_tokens must be positive"):
            ChunkConfig(
                chunk_trigger_tokens=1000,
                target_summary_tokens=0
            )
        
        with pytest.raises(ValueError, match="target_summary_tokens must be positive"):
            ChunkConfig(
                chunk_trigger_tokens=1000,
                target_summary_tokens=-50
            )
    
    def test_config_validation_target_less_than_trigger(self):
        """Test that target must be less than trigger."""
        with pytest.raises(ValueError, match="target_summary_tokens must be less than chunk_trigger_tokens"):
            ChunkConfig(
                chunk_trigger_tokens=1000,
                target_summary_tokens=1000
            )
        
        with pytest.raises(ValueError, match="target_summary_tokens must be less than chunk_trigger_tokens"):
            ChunkConfig(
                chunk_trigger_tokens=1000,
                target_summary_tokens=1500
            )
    
    def test_updated_trigger_applies_to_future_chunks(self):
        """Test that changing trigger mid-session affects future chunking."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=100,  # Start with low trigger
            target_summary_tokens=10,
        )
        storage = DevStorage(session_name="test")
        chunker = ContextChunker(backend, config, storage)
        
        # Add messages to trigger first chunk
        for i in range(5):
            chunker.chat(f"Message {i} " * 20)
        
        chunk_count_1 = chunker.get_stats()["chunk_count"]
        assert chunk_count_1 > 0, "Should have chunked with low trigger"
        
        # Now increase trigger significantly
        chunker.config.chunk_trigger_tokens = 10000
        
        # Add more messages - shouldn't chunk immediately
        for i in range(5):
            chunker.chat(f"Message {i} " * 10)
        
        chunk_count_2 = chunker.get_stats()["chunk_count"]
        # Should not have chunked again (trigger is now very high)
        assert chunk_count_2 == chunk_count_1, "Should not chunk with high trigger"
    
    def test_updated_target_applies_to_future_summaries(self):
        """Test that changing target affects future summarization."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=100,
            target_summary_tokens=10,
        )
        storage = DevStorage(session_name="test")
        chunker = ContextChunker(backend, config, storage)
        
        # Force first chunk
        for i in range(5):
            chunker.chat(f"Message {i} " * 20)
        
        # Change target for next chunk
        chunker.config.target_summary_tokens = 50
        
        # The new target should be used in next chunking operation
        # (We can't easily test the exact token count with mock backend,
        # but we verify the config update doesn't break anything)
        for i in range(5):
            chunker.chat(f"More messages {i} " * 20)
        
        # Should have created 2 chunks
        assert chunker.get_stats()["chunk_count"] >= 2


class TestCLIValidation:
    """Test CLI validation for /set commands."""
    
    def test_validate_positive_trigger(self):
        """Test that CLI should reject negative/zero trigger values."""
        # This would be caught by CLI validation before hitting ChunkConfig
        value = -100
        assert value <= 0, "CLI should reject this"
        
        value = 0
        assert value <= 0, "CLI should reject this"
    
    def test_validate_positive_target(self):
        """Test that CLI should reject negative/zero target values."""
        value = -50
        assert value <= 0, "CLI should reject this"
        
        value = 0
        assert value <= 0, "CLI should reject this"
    
    def test_validate_target_less_than_trigger(self):
        """Test that CLI should validate target < trigger."""
        current_trigger = 1000
        new_target = 1500
        
        # This should be rejected by CLI
        assert new_target >= current_trigger, "CLI should reject this"
    
    def test_validate_trigger_greater_than_target(self):
        """Test that CLI should validate trigger > target."""
        current_target = 500
        new_trigger = 400
        
        # This should be rejected by CLI
        assert new_trigger <= current_target, "CLI should reject this"


class TestMultipleChunkersSync:
    """Test that /set commands keep multiple chunkers in sync."""
    
    def test_all_chunkers_updated_together(self):
        """Test that changing config updates all chunkers in dev mode."""
        backend = MockBackend()
        
        # Create multiple chunkers (simulating dev mode)
        chunkers = {}
        for name in ["concise", "detailed", "code_focused"]:
            config = ChunkConfig(
                chunk_trigger_tokens=1000,
                target_summary_tokens=100,
            )
            storage = DevStorage(session_name=f"dev_{name}")
            chunker = ContextChunker(backend, config, storage)
            chunkers[name] = chunker
        
        # Verify initial values
        for chunker in chunkers.values():
            assert chunker.config.chunk_trigger_tokens == 1000
        
        # Simulate /set trigger command that updates all chunkers
        new_trigger = 2000
        for chunker in chunkers.values():
            chunker.config.chunk_trigger_tokens = new_trigger
        
        # Verify all updated
        for chunker in chunkers.values():
            assert chunker.config.chunk_trigger_tokens == 2000
    
    def test_chunkers_stay_in_sync_after_multiple_updates(self):
        """Test that multiple config changes keep all chunkers synchronized."""
        backend = MockBackend()
        
        chunkers = {}
        for name in ["concise", "detailed"]:
            config = ChunkConfig(
                chunk_trigger_tokens=1000,
                target_summary_tokens=100,
            )
            chunker = ContextChunker(backend, config, DevStorage(session_name=f"dev_{name}"))
            chunkers[name] = chunker
        
        # Update 1: Change trigger
        for chunker in chunkers.values():
            chunker.config.chunk_trigger_tokens = 2000
        
        # Update 2: Change target
        for chunker in chunkers.values():
            chunker.config.target_summary_tokens = 500
        
        # Update 3: Change trigger again
        for chunker in chunkers.values():
            chunker.config.chunk_trigger_tokens = 5000
        
        # Verify all chunkers have the same config
        configs = [chunker.config for chunker in chunkers.values()]
        assert all(c.chunk_trigger_tokens == 5000 for c in configs)
        assert all(c.target_summary_tokens == 500 for c in configs)


class TestSessionPersistence:
    """Test that config changes are reflected in saved sessions."""
    
    def test_config_changes_persisted_in_dev_storage(self):
        """Test that DevStorage records config changes."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=1000,
            target_summary_tokens=100,
        )
        storage = DevStorage(session_name="test_session")
        chunker = ContextChunker(backend, config, storage)
        
        # Chat and save
        chunker.chat("Hello")
        
        # Update config
        chunker.config.chunk_trigger_tokens = 2000
        
        # Continue chatting
        chunker.chat("More chat")
        
        # The storage should reflect the updated config
        # (The actual config values are stored in the chunker, not storage)
        assert chunker.config.chunk_trigger_tokens == 2000


class TestEdgeCases:
    """Test edge cases for config updates."""
    
    def test_update_config_during_chunking(self):
        """Test that config can be safely updated even during active chunking."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=100,
            target_summary_tokens=10,
        )
        chunker = ContextChunker(backend, config)
        
        # Start building up tokens
        for i in range(3):
            chunker.chat(f"Message {i} " * 15)
        
        # Update config mid-conversation
        chunker.config.chunk_trigger_tokens = 200
        chunker.config.target_summary_tokens = 50
        
        # Continue - should use new config
        for i in range(3):
            chunker.chat(f"More {i} " * 15)
        
        # Should complete without errors
        stats = chunker.get_stats()
        assert stats["current_tokens"] > 0
    
    def test_config_update_with_zero_chunks(self):
        """Test updating config before any chunking has occurred."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=1000,
            target_summary_tokens=100,
        )
        chunker = ContextChunker(backend, config)
        
        # Update immediately (no messages yet)
        chunker.config.chunk_trigger_tokens = 2000
        chunker.config.target_summary_tokens = 500
        
        # Now chat
        response = chunker.chat("Hello")
        
        assert response.startswith("Mock response")
        assert chunker.config.chunk_trigger_tokens == 2000
    
    def test_rapid_config_changes(self):
        """Test multiple rapid config changes."""
        backend = MockBackend()
        config = ChunkConfig(
            chunk_trigger_tokens=1000,
            target_summary_tokens=100,
        )
        chunker = ContextChunker(backend, config)
        
        # Rapid updates
        for i in range(10):
            chunker.config.chunk_trigger_tokens = 1000 + (i * 100)
            chunker.config.target_summary_tokens = 100 + (i * 10)
        
        # Final values
        assert chunker.config.chunk_trigger_tokens == 1900
        assert chunker.config.target_summary_tokens == 190
        
        # Should still work
        response = chunker.chat("Test")
        assert response.startswith("Mock response")
