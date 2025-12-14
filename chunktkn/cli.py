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
    # Parse selected chunkers
    selected_chunkers = []
    if hasattr(args, 'chunkers') and args.chunkers:
        selected_chunkers = [c.strip() for c in args.chunkers.split(',')]
        # Validate chunker names
        invalid = [c for c in selected_chunkers if c not in DEFAULT_DEV_PROMPTS]
        if invalid:
            print(f"Error: Invalid chunker names: {', '.join(invalid)}")
            print(f"Available: {', '.join(DEFAULT_DEV_PROMPTS.keys())}")
            sys.exit(1)
    else:
        selected_chunkers = list(DEFAULT_DEV_PROMPTS.keys())
    
    print("=" * 60)
    print("chunkTKN Development Mode")
    print("=" * 60)
    print(f"\nRunning {len(selected_chunkers)} chunker(s):")
    for name in selected_chunkers:
        print(f"  - {name}")
    print(f"\nChunk Trigger: {args.chunk_trigger:,} tokens")
    print(f"Target Summary: {args.target_summary:,} tokens")
    print("\nCommands:")
    print("  /stats               - Show chunker statistics")
    print("  /history             - Show chunk history")
    print("  /save                - Save dev storage to file")
    print("  /sessions            - List all recorded sessions")
    print("  /info                - Show session information")
    print("  /set trigger N       - Set chunk trigger tokens")
    print("  /set target N        - Set target summary tokens")
    print("  /set chunkers LIST   - Set active chunkers (e.g., 'code_focused' or 'detailed,code_focused')")
    print("  /exit                - Exit")
    print()
    
    # Create backend (shared across chunkers)
    try:
        backend = create_backend()
        print(f"Using model: {backend.model_name}\n")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create chunkers with different system prompts (only selected ones)
    chunkers: dict[str, ContextChunker] = {}
    
    for name in selected_chunkers:
        prompt = DEFAULT_DEV_PROMPTS[name]
        config = ChunkConfig(
            chunk_trigger_tokens=args.chunk_trigger,
            target_summary_tokens=args.target_summary,
            preserve_code=True,
            user_system_prompt=None,  # No user system prompt in dev mode (testing only)
            summarization_system_prompt=prompt,  # Use dev prompt for summarization
        )
        
        # Create storage with optional max_chunks limit
        storage = DevStorage(
            session_name=f"dev_{name}",
            max_chunks=args.max_chunks if hasattr(args, 'max_chunks') else None
        )
        
        def make_callback(chunker_name: str, chunker_storage):
            def callback(before: int, after: int):
                print(f"\n{'='*60}")
                print(f"🔄 [{chunker_name}] CHUNKED: {before:,} → {after:,} tokens")
                print(f"{'='*60}")
                # Show the summary that was created
                context = chunker_storage.get_context()
                # Find the summary (second-to-last system message typically)
                summary_msgs = [m for m in context if m["role"] == "system"]
                if len(summary_msgs) > 1:  # First is user prompt, others are summaries
                    latest_summary = summary_msgs[-1]["content"]
                    print(f"\n[{chunker_name}] Summary:")
                    print(f"{latest_summary[:500]}..." if len(latest_summary) > 500 else latest_summary)
                print()
            return callback
        
        chunker = ContextChunker(
            backend=backend,
            config=config,
            storage=storage,
            on_chunk=make_callback(name, storage),
        )
        
        chunkers[name] = chunker
    
    # Show session info
    first_chunker = next(iter(chunkers.values()))
    if isinstance(first_chunker.storage, DevStorage):
        print(f"Session Directory: {first_chunker.storage.session_dir}")
        print(f"Session ID: {first_chunker.storage.session_id}")
        if hasattr(args, 'max_chunks') and args.max_chunks:
            print(f"Max Chunks Limit: {args.max_chunks}")
        print()
    
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
            parts = user_input.lower().split()
            cmd = parts[0]
            
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
                        print(f"         Session: {chunker.storage.session_dir}")
            elif cmd == "/sessions":
                sessions = DevStorage.get_all_sessions()
                if not sessions:
                    print("No sessions recorded yet.")
                else:
                    print(f"\nTotal Sessions: {len(sessions)}")
                    print("=" * 60)
                    for session in sessions[-10:]:  # Show last 10
                        print(f"\nSession ID: {session['session_id']}")
                        print(f"  Name: {session['session_name']}")
                        print(f"  Created: {session['created_at']}")
                        print(f"  Directory: {session['session_dir']}")
                    if len(sessions) > 10:
                        print(f"\n... showing last 10 of {len(sessions)} sessions")
            elif cmd == "/info":
                for name, chunker in chunkers.items():
                    if isinstance(chunker.storage, DevStorage):
                        storage = chunker.storage
                        print(f"\n[{name}]")
                        print(f"  Session ID: {storage.session_id}")
                        print(f"  Session Dir: {storage.session_dir}")
                        print(f"  Session File: {storage.session_file}")
                        print(f"  Chunks: {len(storage.get_chunk_history())}")
                        if storage._max_chunks:
                            print(f"  Max Chunks: {storage._max_chunks}")
                            print(f"  Limit Reached: {storage.max_chunks_reached}")
            elif cmd == "/set" and len(parts) >= 3:
                setting = parts[1]
                if setting == "chunkers":
                    # /set chunkers code_focused,detailed
                    new_chunkers_str = parts[2] if len(parts) == 3 else " ".join(parts[2:])
                    new_chunker_names = [c.strip() for c in new_chunkers_str.split(',')]
                    
                    # Validate chunker names
                    invalid = [c for c in new_chunker_names if c not in DEFAULT_DEV_PROMPTS]
                    if invalid:
                        print(f"Error: Invalid chunker names: {', '.join(invalid)}")
                        print(f"Available: {', '.join(DEFAULT_DEV_PROMPTS.keys())}")
                    else:
                        # Remove chunkers not in new list
                        to_remove = [name for name in chunkers.keys() if name not in new_chunker_names]
                        for name in to_remove:
                            del chunkers[name]
                        
                        # Add new chunkers
                        for name in new_chunker_names:
                            if name not in chunkers:
                                prompt = DEFAULT_DEV_PROMPTS[name]
                                config = ChunkConfig(
                                    chunk_trigger_tokens=args.chunk_trigger,
                                    target_summary_tokens=args.target_summary,
                                    preserve_code=True,
                                    user_system_prompt=None,
                                    summarization_system_prompt=prompt,
                                )
                                storage = DevStorage(
                                    session_name=f"dev_{name}",
                                    max_chunks=args.max_chunks if hasattr(args, 'max_chunks') else None
                                )
                                
                                def make_callback(chunker_name: str, chunker_storage):
                                    def callback(before: int, after: int):
                                        print(f"\n{'='*60}")
                                        print(f"🔄 [{chunker_name}] CHUNKED: {before:,} → {after:,} tokens")
                                        print(f"{'='*60}")
                                        # Show the summary that was created
                                        context = chunker_storage.get_context()
                                        summary_msgs = [m for m in context if m["role"] == "system"]
                                        if len(summary_msgs) > 1:
                                            latest_summary = summary_msgs[-1]["content"]
                                            print(f"\n[{chunker_name}] Summary:")
                                            print(f"{latest_summary[:500]}..." if len(latest_summary) > 500 else latest_summary)
                                        print()
                                    return callback
                                
                                chunker = ContextChunker(
                                    backend=backend,
                                    config=config,
                                    storage=storage,
                                    on_chunk=make_callback(name, storage),
                                )
                                chunkers[name] = chunker
                        
                        print(f"✓ Active chunkers: {', '.join(chunkers.keys())}")
                else:
                    try:
                        value = int(parts[2])
                        
                        # Validate the value before applying
                        if value <= 0:
                            print(f"Error: Value must be positive (got {value})")
                            continue
                        
                        if setting == "trigger":
                            # Check if target would be valid with new trigger
                            sample_config = next(iter(chunkers.values())).config
                            if sample_config.target_summary_tokens >= value:
                                print(f"Error: Target summary ({sample_config.target_summary_tokens:,}) must be less than trigger ({value:,})")
                                continue
                            
                            for chunker in chunkers.values():
                                chunker.config.chunk_trigger_tokens = value
                            args.chunk_trigger = value  # Update args for new chunkers
                            print(f"✓ Chunk trigger set to {value:,} tokens")
                            
                        elif setting == "target":
                            # Check if trigger would still be valid
                            sample_config = next(iter(chunkers.values())).config
                            if value >= sample_config.chunk_trigger_tokens:
                                print(f"Error: Target summary ({value:,}) must be less than trigger ({sample_config.chunk_trigger_tokens:,})")
                                continue
                            
                            for chunker in chunkers.values():
                                chunker.config.target_summary_tokens = value
                            args.target_summary = value  # Update args for new chunkers
                            print(f"✓ Target summary set to {value:,} tokens")
                            
                        else:
                            print(f"Unknown setting: {setting}. Use 'trigger', 'target', or 'chunkers'")
                    except ValueError:
                        print("Error: Value must be a number (except for 'chunkers')")
            else:
                print(f"Unknown command: {user_input}")
            continue
        
        # In dev mode, use ONE chunker for the actual chat response
        # But send the message to ALL chunkers to keep contexts in sync
        primary_chunker_name = list(chunkers.keys())[0]
        primary_response = None
        
        for name, chunker in chunkers.items():
            try:
                response = chunker.chat(user_input)
                if name == primary_chunker_name:
                    primary_response = response
            except Exception as e:
                if name == primary_chunker_name:
                    primary_response = f"Error: {e}"
        
        # Display only the primary response (not showing multiple styles)
        print(f"\n{primary_response}\n")


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
        "--chunkers",
        type=str,
        default=None,
        help="Comma-separated list of chunkers to run in dev mode (e.g., 'code_focused' or 'code_focused,detailed'). Available: concise, detailed, code_focused",
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
    
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks allowed in dev mode (None = unlimited). Use to limit chunking events for testing (e.g., --max-chunks 3)",
    )
    
    args = parser.parse_args()
    
    if args.dev:
        run_dev_mode(args)
    else:
        run_single_mode(args)


if __name__ == "__main__":
    main()
