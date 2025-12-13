# 🧠 chunkTKN

> **Long conversations without context decay.**

A client-side middleware that automatically manages LLM context in long conversations—preventing degradation, preserving critical information, and keeping your chats productive without manual intervention.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🤔 The Problem

Modern LLMs suffer from **context decay** in long conversations:

- After 50k+ tokens, models start losing track of earlier decisions
- Critical architectural choices get forgotten
- Code from earlier in the conversation is hallucinated incorrectly
- Users are forced to manually start new chats and re-explain everything

**Current solutions are inadequate:**
- ❌ Manual copy-paste summaries (tedious, error-prone)
- ❌ Starting new chats (loses continuity, requires re-explanation)
- ❌ Server-side managed conversations (vendor lock-in, limited control)

---

## ✨ The Solution

**chunkTKN** is a client-side context manager that automatically:

✅ **Monitors token usage** in real-time  
✅ **Triggers intelligent summarization** when context grows too large  
✅ **Preserves critical information:**
  - System prompts (never modified)
  - Recent messages (last N kept verbatim)
  - Code blocks (preserved during summarization)
  - Key decisions and architectural context  

✅ **Operates transparently** — users never notice the chunking  
✅ **Works with any Chat Completions API** (OpenAI, Gemini, local models)

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🔄 **Automatic Chunking** | Triggers at configurable token thresholds (default: 50k) |
| 🎨 **Smart Preservation** | System prompt + last N messages kept verbatim |
| 💾 **Flexible Storage** | In-memory, dev mode, or custom backends |
| 🧪 **Dev Mode** | Inspect chunking decisions with detailed logs |
| 🔧 **Highly Configurable** | Customize every aspect of chunking behavior |
| 🚀 **Production Ready** | Battle-tested, fully typed, comprehensive tests |
| 🌐 **Model Agnostic** | Works with OpenAI, Gemini, local LLMs, etc. |

---

## 📦 Installation

### Option 1: Install from Source (V0 - Recommended)

```bash
# Clone the repository
git clone https://github.com/chandan678/chuknTKN.git
cd chunkTKN

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### Option 2: Install from PyPI (Coming Soon)

```bash
pip install chunktkn
```

---

## 🚀 Quick Start

### Basic Usage with OpenAI

```python
from openai import OpenAI
from chunktkn import ContextChunker, OpenAIChatBackend, ChunkConfig

# 1. Setup OpenAI client (as usual)
client = OpenAI(api_key="your-api-key")

# 2. Create backend
backend = OpenAIChatBackend(client, model="gpt-4")

# 3. Configure chunking behavior
config = ChunkConfig(
    chunk_trigger_tokens=50_000,      # Trigger at 50k tokens
    target_summary_tokens=2_000,      # Compress to ~2k tokens
    keep_last_n=10,                   # Keep last 10 messages verbatim
    preserve_code=True,               # Don't summarize code blocks
    system_prompt="You are a helpful coding assistant.",
)

# 4. Create chunker (replaces direct OpenAI calls)
chunker = ContextChunker(backend, config)

# 5. Chat normally - chunking happens automatically!
response = chunker.chat("Help me design a distributed system")
print(response)

response = chunker.chat("What technologies should I use?")
print(response)

# Continue chatting... chunker manages context automatically
```

### Interactive CLI Example

```python
from openai import OpenAI
from chunktkn import ContextChunker, OpenAIChatBackend, ChunkConfig

def main():
    client = OpenAI()
    backend = OpenAIChatBackend(client, model="gpt-4")
    
    config = ChunkConfig(
        chunk_trigger_tokens=50_000,
        target_summary_tokens=2_000,
    )
    
    chunker = ContextChunker(backend, config)
    
    print("🧠 chunkTKN Chat (type 'exit' to quit)\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        
        response = chunker.chat(user_input)
        print(f"\nAssistant: {response}\n")
        
        # Show stats
        stats = chunker.get_stats()
        print(f"📊 Tokens: {stats['current_tokens']:,} | Chunks: {stats['chunk_count']}")

if __name__ == "__main__":
    main()
```

---

## 🎓 Step-by-Step Tutorial

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/chunkTKN.git
cd chunkTKN

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Step 2: Set Your API Key

```bash
# Option 1: Environment variable
export OPENAI_API_KEY="sk-your-key-here"

# Option 2: .env file (create in project root)
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Step 3: Create Your First Script

Create `my_chat.py`:

```python
from openai import OpenAI
from chunktkn import ContextChunker, OpenAIChatBackend, ChunkConfig

# Initialize
client = OpenAI()  # Uses OPENAI_API_KEY from env
backend = OpenAIChatBackend(client, model="gpt-4")

# Configure chunking
config = ChunkConfig(
    chunk_trigger_tokens=50_000,
    target_summary_tokens=2_000,
    system_prompt="You are a helpful assistant.",
)

# Create chunker
chunker = ContextChunker(backend, config)

# Start chatting
response = chunker.chat("What's the capital of France?")
print(response)
```

### Step 4: Run It!

```bash
python my_chat.py
```

That's it! The chunker now manages your context automatically.

---

## 🔬 Dev Mode - Inspect Chunking Decisions

Dev mode provides detailed insights into chunking behavior. Perfect for debugging and understanding how your context is being managed.

### Enable Dev Mode

```python
from openai import OpenAI
from chunktkn import ContextChunker, OpenAIChatBackend, ChunkConfig
from chunktkn.storage.dev import DevStorage

client = OpenAI()
backend = OpenAIChatBackend(client, model="gpt-4")

# Use DevStorage instead of default MemoryStorage
dev_storage = DevStorage()

config = ChunkConfig(
    chunk_trigger_tokens=50_000,
    target_summary_tokens=2_000,
)

chunker = ContextChunker(
    backend=backend,
    config=config,
    storage=dev_storage,  # 👈 Enable dev mode
)

# Chat normally
for i in range(20):
    response = chunker.chat(f"Tell me fact #{i} about space")

# Inspect chunking history
print("\n" + "="*60)
print("📊 CHUNKING HISTORY")
print("="*60)

for idx, chunk_event in enumerate(dev_storage.chunk_history, 1):
    print(f"\n🔄 Chunk #{idx}")
    print(f"   Tokens before: {chunk_event['tokens']:,}")
    print(f"   Messages before: {len(chunk_event['context'])}")
    print(f"   Messages after: {len(chunk_event['chunked_context'])}")
    
    # Show what was preserved
    chunked = chunk_event['chunked_context']
    print(f"\n   📋 New context structure:")
    for msg in chunked[:3]:  # Show first 3 messages
        role = msg['role'].upper()
        content_preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
        print(f"      [{role}] {content_preview}")
```

### Dev Mode CLI Tool

chunkTKN includes a built-in CLI with dev mode:

```bash
# Start interactive chat with dev mode enabled
python -m chunktkn.cli --dev

# Example session:
# You: Help me build a web scraper
# Assistant: [response]
# 
# 📊 Stats: 234 tokens | 0 chunks | Trigger: 50,000
# 
# You: What libraries should I use?
# Assistant: [response]
# 
# 📊 Stats: 456 tokens | 0 chunks | Trigger: 50,000
```

### Custom Dev Logging

```python
from chunktkn import ContextChunker, OpenAIChatBackend, ChunkConfig

# Define callback for chunking events
def on_chunk_callback(tokens_before: int, tokens_after: int):
    compression_ratio = (1 - tokens_after / tokens_before) * 100
    print(f"\n⚡ CHUNKING OCCURRED!")
    print(f"   Before: {tokens_before:,} tokens")
    print(f"   After:  {tokens_after:,} tokens")
    print(f"   Saved:  {compression_ratio:.1f}% compression\n")

client = OpenAI()
backend = OpenAIChatBackend(client, model="gpt-4")
config = ChunkConfig(chunk_trigger_tokens=50_000)

chunker = ContextChunker(
    backend=backend,
    config=config,
    on_chunk=on_chunk_callback,  # 👈 Get notified on chunking
)

# Now chunking events will trigger your callback
chunker.chat("Start a long conversation...")
```

---

## ⚙️ Configuration Recipes

Different use cases require different chunking strategies. Here are proven configurations:

### 1. Aggressive Chunking (Small Context)

For cost optimization or models with smaller context windows:

```python
config = ChunkConfig(
    chunk_trigger_tokens=20_000,      # Chunk early
    target_summary_tokens=1_000,      # Aggressive compression
    keep_last_n=5,                    # Only keep last 5 messages
    preserve_code=True,
)
```

**Use when:**
- Using expensive models (GPT-4)
- Tight token budgets
- Fast prototyping

---

### 2. Conservative Chunking (Large Context)

For maximum context retention:

```python
config = ChunkConfig(
    chunk_trigger_tokens=100_000,     # Chunk late
    target_summary_tokens=5_000,      # Keep more detail
    keep_last_n=20,                   # Preserve more recent messages
    preserve_code=True,
)
```

**Use when:**
- Complex architectural discussions
- Long debugging sessions
- Context is critical

---

### 3. Code-Focused Chunking

Optimized for programming tasks:

```python
config = ChunkConfig(
    chunk_trigger_tokens=50_000,
    target_summary_tokens=3_000,
    keep_last_n=15,                   # More recency for iteration
    preserve_code=True,               # CRITICAL for code
    code_detection_mode="markdown",   # Strict code block detection
    system_prompt="""You are an expert programmer.
Always provide complete, working code.
Explain your architectural decisions.""",
)
```

**Use when:**
- Building complex systems
- Iterating on code
- Need to reference earlier implementations

---

### 4. Architecture & Planning

For high-level design discussions:

```python
config = ChunkConfig(
    chunk_trigger_tokens=75_000,
    target_summary_tokens=4_000,      # Keep more context
    keep_last_n=10,
    preserve_code=False,              # Summaries can paraphrase
    system_prompt="""You are a senior software architect.
Focus on: scalability, maintainability, trade-offs.
Always explain your reasoning.""",
)
```

**Use when:**
- System design interviews
- Architecture reviews
- Planning new projects

---

### 5. Debugging Sessions

Optimized for troubleshooting:

```python
config = ChunkConfig(
    chunk_trigger_tokens=60_000,
    target_summary_tokens=2_500,
    keep_last_n=12,                   # Keep recent error context
    preserve_code=True,
    system_prompt="""You are a debugging expert.
When analyzing errors:
1. Identify root cause
2. Explain why it happens
3. Provide tested fix
4. Suggest prevention strategies""",
)
```

**Use when:**
- Tracking down bugs
- Analyzing stack traces
- Iterative problem solving

---

### 6. Learning & Tutorials

For educational conversations:

```python
config = ChunkConfig(
    chunk_trigger_tokens=40_000,
    target_summary_tokens=2_000,
    keep_last_n=8,
    preserve_code=True,
    system_prompt="""You are a patient teacher.
Explain concepts:
- Start with fundamentals
- Use analogies
- Provide examples
- Check understanding""",
)
```

**Use when:**
- Learning new technologies
- Following tutorials
- Asking "how does X work?"

---

## 📖 API Reference

### ContextChunker

Main class for managing conversation context.

```python
class ContextChunker:
    def __init__(
        self,
        backend: ModelBackend,
        config: ChunkConfig | None = None,
        storage: StorageBackend | None = None,
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> None:
        """Initialize the context chunker.
        
        Args:
            backend: Model backend (OpenAI, Gemini, etc.)
            config: Chunking configuration
            storage: Storage backend (defaults to MemoryStorage)
            on_chunk: Optional callback when chunking occurs
        """
```

#### Methods

```python
def chat(message: str) -> str:
    """Send a message and get a response.
    
    Automatically manages context, triggers chunking when needed.
    
    Args:
        message: User message to send
        
    Returns:
        Model's response
    """

def get_context() -> list[Message]:
    """Get current conversation context."""

def get_token_count() -> int:
    """Get current token count."""

def get_stats() -> dict:
    """Get chunker statistics.
    
    Returns:
        {
            "current_tokens": int,
            "total_tokens_processed": int,
            "chunk_count": int,
            "chunk_trigger": int,
        }
    """

def reset() -> None:
    """Reset conversation state."""

def force_chunk() -> None:
    """Force chunking regardless of token count."""
```

---

### ChunkConfig

Configuration for chunking behavior.

```python
@dataclass
class ChunkConfig:
    # Core settings
    chunk_trigger_tokens: int = 50_000        # When to chunk
    target_summary_tokens: int = 2_000        # Compression target
    max_history_tokens: int | None = None     # Hard limit (optional)
    
    # Preservation settings
    keep_last_n: int = 10                     # Recent messages to preserve
    preserve_code: bool = True                # Keep code verbatim
    code_detection_mode: Literal["heuristic", "markdown"] = "heuristic"
    
    # System prompt
    system_prompt: str | None = None          # Optional system message
    
    # Mode (V0 only supports MANAGER)
    mode: ChunkMode = ChunkMode.MANAGER
```

---

### Backends

#### OpenAIChatBackend

```python
from chunktkn import OpenAIChatBackend

backend = OpenAIChatBackend(
    client=OpenAI(),
    model="gpt-4",           # or "gpt-4-turbo", "gpt-3.5-turbo"
)
```

#### Custom Backend

Implement the `ModelBackend` interface:

```python
from chunktkn.backends.base import ModelBackend, Message

class MyCustomBackend(ModelBackend):
    @property
    def model_name(self) -> str:
        return "my-model"
    
    def chat(self, messages: list[Message]) -> str:
        # Your implementation
        pass
    
    def count_tokens(self, messages: list[Message]) -> int:
        # Your implementation
        pass
```

---

## 🏗️ Architecture

chunkTKN uses a simple, composable architecture:

```
┌─────────────────────────────────────────────┐
│              User Application               │
│         chunker.chat("message")             │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│           ContextChunker                    │
│  • Owns conversation history                │
│  • Monitors token usage                     │
│  • Triggers chunking at threshold           │
│  • Coordinates summarization                │
└────────┬───────────────────────┬────────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐   ┌──────────────────────┐
│  StorageBackend  │   │    ModelBackend      │
│  • Memory        │   │  • OpenAI            │
│  • Dev           │   │  • Gemini            │
│  • Custom        │   │  • Local LLM         │
└──────────────────┘   └──────────────────────┘
```

### Chunking Strategy

When context exceeds `chunk_trigger_tokens`:

```
BEFORE CHUNKING:
[system_prompt, msg1, msg2, ..., msg40, msg41, ..., msg50]
       ↑              ↑                          ↑
   preserved    summarized (1-40)         kept verbatim (41-50)

AFTER CHUNKING:
[system_prompt, summary_message, msg41, msg42, ..., msg50]
       ↑              ↑                        ↑
  never touched    ~2k tokens            last N preserved
```

**Key insights:**
- System prompt is **never** modified
- Recent context (last N messages) is **never** summarized
- Only the middle portion is compressed
- Result: Maximum context quality with minimal token usage

---

## 🤝 Contributing

Contributions welcome! Please check out our [contributing guide](CONTRIBUTING.md).

### Development Setup

```bash
# Clone and setup
git clone https://github.com/chandan678/chuknTKN.git
cd chunkTKN
python3.10 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=chunktkn --cov-report=html

# Format code
black chunktkn tests
```

---

## 🗺️ Roadmap

### V0 (Current)
- ✅ Manager mode (full context ownership)
- ✅ Smart chunking with preservation
- ✅ OpenAI backend
- ✅ Dev mode for debugging
- ✅ Comprehensive tests

### V1 (Planned)
- 🚧 Observer mode (managed services like OpenAI Assistants)
- 🚧 Async background summarization
- 🚧 Gemini backend
- 🚧 Anthropic (Claude) backend
- 🚧 Multi-client state sync

### V2 (Future)
- 💭 RAG integration for external knowledge
- 💭 Semantic chunking (topic-aware boundaries)
- 💭 Custom summarization strategies
- 💭 Streaming support

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [tiktoken](https://github.com/openai/tiktoken)
- [rich](https://github.com/Textualize/rich) (for beautiful CLI output)


---

<div align="center">

**⭐ Star us on GitHub — it helps!**

Made with ❤️ by [Chandan K S](https://github.com/chandan678)

</div>
