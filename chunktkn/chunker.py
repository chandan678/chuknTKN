"""Core ContextChunker implementation.

The main orchestrator that manages conversation history,
detects when chunking is needed, and coordinates summarization.

Modes:
    MANAGER (V0): Owns history, full chunk/replace. For raw Chat Completions API.
    OBSERVER (V1): Watches external conversations, acts via backend ops.
"""

from typing import Callable

from chunktkn.backends.base import ModelBackend, Message
from chunktkn.config import ChunkConfig, ChunkMode
from chunktkn.storage.base import StorageBackend
from chunktkn.storage.memory import MemoryStorage
from chunktkn.storage.dev import DevStorage
from chunktkn.summarizer import summarize_context


class ContextChunker:
    """Client-side context manager for LLM conversations.
    
    ContextChunker wraps a model backend and automatically manages
    conversation context. When the context grows too large, it:
    
    1. Summarizes the conversation history
    2. Preserves critical information (code, decisions, TODOs)
    3. Replaces the context with the compressed summary
    4. Continues the conversation seamlessly
    
    Modes:
        MANAGER (default, V0): chunkTKN owns history and calls the model.
        OBSERVER (future V1): chunkTKN watches external convos and acts via backend.
    
    Example:
        >>> from openai import OpenAI
        >>> from chunktkn import ContextChunker, OpenAIChatBackend, ChunkConfig
        >>> 
        >>> client = OpenAI()
        >>> backend = OpenAIChatBackend(client, model="gpt-4.1")
        >>> config = ChunkConfig(chunk_trigger_tokens=50_000)
        >>> 
        >>> chunker = ContextChunker(backend, config)
        >>> reply = chunker.chat("Help me design a system")
    """
    
    def __init__(
        self,
        backend: ModelBackend,
        config: ChunkConfig | None = None,
        storage: StorageBackend | None = None,
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> None:
        """Initialize the ContextChunker.
        
        Args:
            backend: Model backend for LLM calls.
            config: Chunking configuration. Uses defaults if None.
            storage: Storage backend for context persistence.
                     Uses MemoryStorage if None.
            on_chunk: Optional callback called when chunking occurs.
                      Receives (tokens_before, tokens_after).
        """
        self._backend = backend
        self._config = config or ChunkConfig()
        self._storage = storage or MemoryStorage()
        self._on_chunk = on_chunk
        
        # Stats tracking
        self._total_tokens_processed: int = 0
        self._chunk_count: int = 0
        
        # Initialize based on mode
        self._init_for_mode()
    
    def _init_for_mode(self) -> None:
        """Initialize chunker based on operating mode."""
        if self._config.mode == ChunkMode.MANAGER:
            # Manager mode: Add system prompt if configured
            if self._config.system_prompt:
                system_msg: Message = {
                    "role": "system",
                    "content": self._config.system_prompt,
                }
                self._storage.append_message(system_msg)
        
        elif self._config.mode == ChunkMode.OBSERVER:
            # Observer mode: V1 - will connect to external conversation
            # For now, this raises NotImplementedError in ChunkConfig validation
            pass
    
    @property
    def backend(self) -> ModelBackend:
        """The model backend being used."""
        return self._backend
    
    @property
    def config(self) -> ChunkConfig:
        """The chunking configuration."""
        return self._config
    
    @property
    def storage(self) -> StorageBackend:
        """The storage backend being used."""
        return self._storage
    
    @property
    def chunk_count(self) -> int:
        """Number of times chunking has occurred."""
        return self._chunk_count
    
    @property
    def total_tokens_processed(self) -> int:
        """Total tokens processed across all messages."""
        return self._total_tokens_processed
    
    def chat(self, message: str) -> str:
        """Send a message and get a response.
        
        This is the main entry point. Behavior depends on mode:
        
        MANAGER mode:
            1. Appends the user message to history
            2. Checks if chunking is needed
            3. Calls the model
            4. Appends the response to history
            5. Returns the response
        
        OBSERVER mode (V1):
            1. Sends message via backend
            2. Fetches updated state from backend
            3. Checks if chunking is needed
            4. Acts on overflow (delete old + inject summary, or new thread)
            5. Returns the response
        
        Args:
            message: User message to send.
            
        Returns:
            The model's response.
        """
        if self._config.mode == ChunkMode.MANAGER:
            return self._chat_manager(message)
        else:
            # OBSERVER mode - V1
            return self._chat_observer(message)
    
    def _chat_manager(self, message: str) -> str:
        """Manager mode: Own history, full chunk/replace.
        
        Args:
            message: User message to send.
            
        Returns:
            The model's response.
        """
        # Append user message
        user_msg: Message = {"role": "user", "content": message}
        self._storage.append_message(user_msg)
        
        # Get current context and count tokens
        context = self._storage.get_context()
        token_count = self._backend.count_tokens(context)
        self._total_tokens_processed += token_count
        
        # Check if we need to chunk
        if token_count >= self._config.chunk_trigger_tokens:
            context = self._perform_chunking(context, token_count)
        
        # Call the model
        response = self._backend.chat(context)
        
        # Append assistant response
        assistant_msg: Message = {"role": "assistant", "content": response}
        self._storage.append_message(assistant_msg)
        
        return response
    
    def _chat_observer(self, message: str) -> str:
        """Observer mode: Watch external conversations, act via backend ops.
        
        V1 Implementation - Not yet available.
        
        Flow:
            1. backend.send(message) - Send via external system
            2. messages = backend.get_messages() - Fetch updated state
            3. if should_chunk(messages): _act_on_overflow(summary)
            4. return backend.get_last_response()
        
        Args:
            message: User message to send.
            
        Returns:
            The model's response.
        """
        # This should never be called in V0 due to config validation
        raise NotImplementedError(
            "Observer mode is not yet implemented. Coming in V1."
        )
    
    def _act_on_overflow(self, summary: str) -> None:
        """Handle context overflow in observer mode.
        
        V1 Implementation - Not yet available.
        
        Actions:
            - If backend supports delete_old_messages: Delete + inject summary
            - Otherwise: Create new thread with summary
        
        Args:
            summary: The compressed context summary.
        """
        # V1: Check backend capabilities and act accordingly
        # if hasattr(self._backend, 'delete_old_messages'):
        #     self._backend.delete_old_messages(keep_last=self._config.max_old_messages)
        #     self._backend.add_message(role="system", content=f"Previous context: {summary}")
        # else:
        #     new_id = self._backend.create_new_thread([{"role": "system", "content": summary}])
        raise NotImplementedError("Observer mode actions not yet implemented.")
    
    def _perform_chunking(
        self,
        context: list[Message],
        token_count: int,
    ) -> list[Message]:
        """Perform context chunking/summarization.
        
        Chunking preserves:
        1. System prompt (if any) - kept exactly as-is, never summarized
        2. Last N messages - kept verbatim for recency
        3. Middle portion - summarized to target_summary_tokens
        
        Result structure: [system_prompt?, summary, last_n_messages...]
        
        Args:
            context: Current context to chunk.
            token_count: Current token count.
            
        Returns:
            New compressed context.
        """
        # Step 1: Extract system prompt (preserve separately, never summarize)
        system_prompt: Message | None = None
        messages_to_process = context
        
        if context and context[0]["role"] == "system":
            system_prompt = context[0]
            messages_to_process = context[1:]
        
        # Step 2: Split into "to summarize" and "keep verbatim"
        keep_last_n = self._config.keep_last_n
        
        if len(messages_to_process) <= keep_last_n:
            # Not enough messages to split - summarize all (rare edge case)
            messages_to_summarize = messages_to_process
            messages_to_keep = []
        else:
            messages_to_summarize = messages_to_process[:-keep_last_n]
            messages_to_keep = messages_to_process[-keep_last_n:]
        
        # Step 3: Summarize the older messages
        summarized = summarize_context(
            backend=self._backend,
            messages=messages_to_summarize,
            target_tokens=self._config.target_summary_tokens,
            preserve_code=self._config.preserve_code,
        )
        
        # Step 4: Build new context: [system_prompt?] + [summary] + [last_n]
        new_context: list[Message] = []
        
        # Add original system prompt first (preserved exactly)
        if system_prompt:
            new_context.append(system_prompt)
        
        # Add summary (comes as a system message from summarizer)
        new_context.extend(summarized)
        
        # Add last N messages verbatim
        new_context.extend(messages_to_keep)
        
        # Calculate new token count
        new_token_count = self._backend.count_tokens(new_context)
        
        # Record chunk if using dev storage
        if isinstance(self._storage, DevStorage):
            self._storage.record_chunk(
                tokens=token_count,
                context=context,
                chunked_context=new_context,
            )
        
        # Update storage with new context
        self._storage.set_context(new_context)
        self._storage.set_chunked_context(new_context)
        
        # Update stats
        self._chunk_count += 1
        
        # Call callback if provided
        if self._on_chunk:
            self._on_chunk(token_count, new_token_count)
        
        return new_context
    
    def get_context(self) -> list[Message]:
        """Get the current conversation context.
        
        Returns:
            List of messages in the current context.
        """
        return self._storage.get_context()
    
    def get_token_count(self) -> int:
        """Get the current token count.
        
        Returns:
            Token count of current context.
        """
        return self._backend.count_tokens(self._storage.get_context())
    
    def get_stats(self) -> dict:
        """Get chunker statistics.
        
        Returns:
            Dictionary with stats:
            - current_tokens: Current context token count
            - total_tokens_processed: All tokens seen
            - chunk_count: Number of chunking events
            - chunk_trigger: Token threshold for chunking
        """
        return {
            "current_tokens": self.get_token_count(),
            "total_tokens_processed": self._total_tokens_processed,
            "chunk_count": self._chunk_count,
            "chunk_trigger": self._config.chunk_trigger_tokens,
        }
    
    def reset(self) -> None:
        """Reset the chunker state.
        
        Clears conversation history and resets stats.
        """
        self._storage.clear()
        self._total_tokens_processed = 0
        self._chunk_count = 0
        
        # Re-add system prompt if configured
        if self._config.system_prompt:
            system_msg: Message = {
                "role": "system",
                "content": self._config.system_prompt,
            }
            self._storage.append_message(system_msg)
    
    def force_chunk(self) -> None:
        """Force a chunking operation regardless of token count.
        
        Useful for manual checkpointing.
        """
        context = self._storage.get_context()
        token_count = self._backend.count_tokens(context)
        
        if context:
            self._perform_chunking(context, token_count)
