"""Summarization logic for context chunking.

The summarizer uses the SAME backend model that the user is using
for their conversation — no separate summarization model.
"""

from chunktkn.backends.base import ModelBackend, Message


# Prompt template for summarization
SUMMARIZATION_PROMPT = """You are a context compression assistant. Your job is to summarize the conversation history while preserving critical information.

## Rules:

1. **PRESERVE EXACTLY** (copy verbatim):
   - All final working code blocks
   - Specific function/class names and signatures
   - Configuration values and settings
   - File paths and URLs

2. **SUMMARIZE** (compress but keep meaning):
   - Architectural decisions and their reasoning
   - What approaches failed and why
   - Current state of the project/task
   - Pending TODOs and next steps

3. **REMOVE** (do not include):
   - Conversational pleasantries ("Great question!", "Sure, I can help")
   - Abandoned approaches (unless failure is instructive)
   - Redundant explanations
   - Step-by-step debugging that led nowhere

## Target Length:
Compress to approximately {target_tokens} tokens while preserving all critical information.

## Output Format:
Return a structured summary that can serve as context for continuing the conversation. Start with "## Context Summary" and organize by topic.

---

## Conversation to Summarize:

{conversation}"""


CODE_PRESERVATION_PROMPT = """
## Additional Instruction:
The user has requested that code blocks be preserved verbatim. When you encounter code blocks (```...```), include them exactly as-is in your summary. Do not paraphrase or abbreviate code.
"""


def build_summarization_prompt(
    messages: list[Message],
    target_tokens: int,
    preserve_code: bool = True,
) -> str:
    """Build the summarization prompt from conversation history.
    
    Args:
        messages: Conversation messages to summarize.
        target_tokens: Target token count for the summary.
        preserve_code: Whether to instruct preservation of code blocks.
        
    Returns:
        The complete summarization prompt.
    """
    # Format conversation as text
    conversation_parts = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        conversation_parts.append(f"[{role}]\n{content}")
    
    conversation = "\n\n---\n\n".join(conversation_parts)
    
    # Build prompt
    prompt = SUMMARIZATION_PROMPT.format(
        target_tokens=target_tokens,
        conversation=conversation,
    )
    
    if preserve_code:
        prompt += CODE_PRESERVATION_PROMPT
    
    return prompt


def summarize_context(
    backend: ModelBackend,
    messages: list[Message],
    target_tokens: int,
    preserve_code: bool = True,
    summarization_system_prompt: str | None = None,
) -> list[Message]:
    """Summarize conversation context using the provided backend.
    
    This function:
    1. Builds a summarization prompt
    2. Optionally adds a system prompt to guide summarization style
    3. Calls the backend (same model user is using)
    4. Returns a new context with the summary as a system message
    
    Args:
        backend: The model backend to use for summarization.
        messages: Conversation messages to summarize.
        target_tokens: Target token count for the summary.
        preserve_code: Whether to preserve code blocks verbatim.
        summarization_system_prompt: Optional system prompt to guide summarization
                                     (e.g., "Be concise", "Focus on code").
        
    Returns:
        New context with summarized history as a single system message.
    """
    # Build the summarization prompt
    prompt = build_summarization_prompt(
        messages=messages,
        target_tokens=target_tokens,
        preserve_code=preserve_code,
    )
    
    # Call the model with the summarization request
    summarization_messages: list[Message] = []
    
    # Add system prompt if provided (guides summarization style)
    if summarization_system_prompt:
        summarization_messages.append({
            "role": "system",
            "content": summarization_system_prompt
        })
    
    summarization_messages.append({
        "role": "user",
        "content": prompt
    })
    
    summary = backend.chat(summarization_messages)
    
    # Return as a new context with the summary as system message
    return [
        {"role": "system", "content": summary}
    ]


def extract_code_blocks(text: str) -> list[str]:
    """Extract code blocks from text.
    
    Args:
        text: Text potentially containing code blocks.
        
    Returns:
        List of code block contents (without the ``` markers).
    """
    import re
    
    # Match fenced code blocks
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    return matches


def detect_code_heuristic(text: str) -> bool:
    """Detect if text likely contains code using heuristics.
    
    Args:
        text: Text to analyze.
        
    Returns:
        True if text likely contains code.
    """
    # Common code indicators
    code_patterns = [
        r"```",  # Fenced code blocks
        r"def \w+\(",  # Python functions
        r"class \w+[:\(]",  # Python/JS classes
        r"function \w+\(",  # JS functions
        r"import \w+",  # Imports
        r"from \w+ import",  # Python imports
        r"const \w+ =",  # JS const
        r"let \w+ =",  # JS let
        r"var \w+ =",  # JS var
        r"if \w+ [={<>]",  # Conditionals
        r"for \w+ in",  # For loops
        r"while \w+:",  # While loops
        r"return \w+",  # Return statements
        r"print\(",  # Print calls
        r"console\.",  # Console calls
    ]
    
    import re
    for pattern in code_patterns:
        if re.search(pattern, text):
            return True
    
    return False
