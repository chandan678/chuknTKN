"""CLI tool for chunkTKN with --dev mode.

This provides a development/testing CLI that allows:
- Running parallel chunkers with different system prompts
- Inspecting chunk history
- Comparing chunking behavior side-by-side
"""

import argparse
import os
import sys
from typing import Any

from chunktkn.config import ChunkConfig
from chunktkn.chunker import ContextChunker
from chunktkn.storage.dev import DevStorage


# Default system prompts for parallel testing in dev mode
DEFAULT_DEV_PROMPTS = {
    "concise": "You are a concise assistant. Give brief, direct answers without unnecessary explanation.",
    "detailed": "You are a detailed assistant. Provide thorough explanations with examples when helpful.",
    "code_focused": "You are a code-focused assistant. Prioritize showing code examples and implementations over explanations.",
}


def create_backend():
    """Create the model backend based on available API keys.
    
    Returns:
        A ModelBackend instance.
        
    Raises:
        RuntimeError: If no API key is found.
    """
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            from chunktkn.backends.openai import OpenAIChatBackend
            
            client = OpenAI()
            return OpenAIChatBackend(client, model="gpt-4.1")
        except ImportError:
            pass
    
    raise RuntimeError(
        "No API key found. Please set OPENAI_API_KEY environment variable."
    )


def print_stats(chunkers: dict[str, ContextChunker]) -> None:
    """Print stats for all chunkers."""
    print("\n" + "=" * 60)
    print("CHUNKER STATS")
    print("=" * 60)
    
    for name, chunker in chunkers.items():
        stats = chunker.get_stats()
        print(f"\n[{name}]")
        print(f"  Tokens: {stats['current_tokens']:,} / {stats['chunk_trigger']:,}")
        print(f"  Chunks: {stats['chunk_count']}")
        print(f"  Total processed: {stats['total_tokens_processed']:,}")
    
    print()


def print_responses(responses: dict[str, str]) -> None:
    """Print responses from all chunkers."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.columns import Columns
        
        console = Console()
        panels = []
        
        for name, response in responses.items():
            panel = Panel(
                response[:500] + ("..." if len(response) > 500 else ""),
                title=f"[bold]{name}[/bold]",
                border_style="blue",
            )
            panels.append(panel)
        
        console.print(Columns(panels, equal=True, expand=True))
        
    except ImportError:
        # Fallback without rich
        for name, response in responses.items():
            print(f"\n[{name}]")
            print("-" * 40)
            print(response[:500] + ("..." if len(response) > 500 else ""))
            print()


def print_chunk_history(storage: DevStorage) -> None:
    """Print chunk history for a dev storage."""
    chunks = storage.get_chunk_history()
    
    if not chunks:
        print("No chunks recorded yet.")
        return
    
    print("\n" + "=" * 60)
    print("CHUNK HISTORY")
    print("=" * 60)
    
    for chunk in chunks:
        print(f"\n[Chunk {chunk.id}]")
        print(f"  Tokens at chunk: {chunk.tokens:,}")
        print(f"  Context messages: {len(chunk.context)}")
        print(f"  Chunked to: {len(chunk.chunked_context)} message(s)")
    
    print()


def run_dev_mode(args: argparse.Namespace) -> None:
    """Run in development mode with parallel chunkers.
    
    Args:
        args: Parsed command line arguments.
    """
    print("=" * 60)
    print("chunkTKN Development Mode")
    print("=" * 60)
    print(f"\nRunning {len(DEFAULT_DEV_PROMPTS)} parallel chunkers:")
    for name in DEFAULT_DEV_PROMPTS:
        print(f"  - {name}")
    print("\nCommands:")
    print("  /stats    - Show chunker statistics")
    print("  /history  - Show chunk history")
    print("  /save     - Save dev storage to file")
    print("  /exit     - Exit")
    print()
    
    # Create backend (shared across chunkers)
    try:
        backend = create_backend()
        print(f"Using model: {backend.model_name}\n")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create chunkers with different system prompts
    chunkers: dict[str, ContextChunker] = {}
    
    for name, prompt in DEFAULT_DEV_PROMPTS.items():
        config = ChunkConfig(
            chunk_trigger_tokens=args.chunk_trigger,
            target_summary_tokens=args.target_summary,
            preserve_code=True,
            system_prompt=prompt,
        )
        
        storage = DevStorage(session_name=f"dev_{name}")
        
        def make_callback(chunker_name: str):
            def callback(before: int, after: int):
                print(f"\n[{chunker_name}] Chunked: {before:,} → {after:,} tokens")
            return callback
        
        chunker = ContextChunker(
            backend=backend,
            config=config,
            storage=storage,
            on_chunk=make_callback(name),
        )
        
        chunkers[name] = chunker
    
    # Main loop
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            
            if cmd in {"/exit", "/quit"}:
                print("Goodbye!")
                break
            elif cmd == "/stats":
                print_stats(chunkers)
            elif cmd == "/history":
                for name, chunker in chunkers.items():
                    print(f"\n[{name}]")
                    if isinstance(chunker.storage, DevStorage):
                        print_chunk_history(chunker.storage)
            elif cmd == "/save":
                for name, chunker in chunkers.items():
                    if isinstance(chunker.storage, DevStorage):
                        path = chunker.storage.save_to_file()
                        print(f"[{name}] Saved to: {path}")
            else:
                print(f"Unknown command: {user_input}")
            continue
        
        # Send to all chunkers
        responses: dict[str, str] = {}
        
        for name, chunker in chunkers.items():
            try:
                response = chunker.chat(user_input)
                responses[name] = response
            except Exception as e:
                responses[name] = f"Error: {e}"
        
        print_responses(responses)
        print_stats(chunkers)


def run_single_mode(args: argparse.Namespace) -> None:
    """Run in single chunker mode (normal usage).
    
    Args:
        args: Parsed command line arguments.
    """
    print("chunkTKN Chat")
    print("Type '/exit' to quit, '/stats' for statistics.\n")
    
    try:
        backend = create_backend()
        print(f"Using model: {backend.model_name}\n")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    config = ChunkConfig(
        chunk_trigger_tokens=args.chunk_trigger,
        target_summary_tokens=args.target_summary,
        preserve_code=True,
    )
    
    def on_chunk(before: int, after: int) -> None:
        print(f"\n[Context chunked: {before:,} → {after:,} tokens]\n")
    
    chunker = ContextChunker(
        backend=backend,
        config=config,
        on_chunk=on_chunk,
    )
    
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in {"/exit", "/quit"}:
            print("Goodbye!")
            break
        elif user_input.lower() == "/stats":
            stats = chunker.get_stats()
            print(f"\nTokens: {stats['current_tokens']:,} / {stats['chunk_trigger']:,}")
            print(f"Chunks: {stats['chunk_count']}")
            print(f"Total processed: {stats['total_tokens_processed']:,}\n")
            continue
        
        try:
            response = chunker.chat(user_input)
            print(f"\nmodel> {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="chunktkn",
        description="Context-aware LLM chat with automatic chunking",
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode with parallel chunkers",
    )
    
    parser.add_argument(
        "--chunk-trigger",
        type=int,
        default=50_000,
        help="Token count that triggers chunking (default: 50000)",
    )
    
    parser.add_argument(
        "--target-summary",
        type=int,
        default=2_000,
        help="Target token count for summaries (default: 2000)",
    )
    
    args = parser.parse_args()
    
    if args.dev:
        run_dev_mode(args)
    else:
        run_single_mode(args)


if __name__ == "__main__":
    main()
