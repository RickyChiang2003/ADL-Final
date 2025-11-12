import os
from typing import Optional, List
from .attacks_ import apply_trigger_to_prompt, load_trigger


_TRIGGER_CACHE = {}


def _load_cached_trigger(trigger_path: str = "models/trigger.json") -> Optional[List[int]]:
    """Load and cache trigger tokens from disk."""
    if trigger_path not in _TRIGGER_CACHE:
        trigger = load_trigger(trigger_path)
        _TRIGGER_CACHE[trigger_path] = trigger
    return _TRIGGER_CACHE[trigger_path]


def algorithm_1(toxic_prompt: str) -> str:
    """Identity algorithm: return prompt unchanged."""
    return toxic_prompt


def algorithm_2(toxic_prompt: str) -> str:
    """Simple fixed replacement (baseline)."""
    return 'The weather is so nice today!'


def algorithm_with_trigger(
    toxic_prompt: str,
    trigger_path: str = "models/trigger.json",
    mode: str = "prepend"
) -> str:
    """
    Apply pre-trained universal trigger tokens to prompt.

    This algorithm:
    1. Loads a pre-trained trigger (discrete token IDs) optimized on source prompts.
    2. Applies it to the input prompt (prepend/append).
    3. Returns the trigger-augmented prompt as a string.

    Args:
        toxic_prompt: Input prompt to be rewritten.
        trigger_path: Path to saved trigger.json file.
        mode: "prepend", "append", or "insert".

    Returns:
        Prompt with applied trigger tokens.
    """
    trigger_data = _load_cached_trigger(trigger_path)
    if trigger_data is None:
        return toxic_prompt

    # Unpack trigger data (now returns tuple)
    if isinstance(trigger_data, tuple):
        trigger_tokens, token_positions = trigger_data
    else:
        # Fallback for old format
        trigger_tokens = trigger_data
        token_positions = None

    # Try to import tokenizer for decoding
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")

        # Use token_positions if available, otherwise use mode
        if token_positions:
            return apply_trigger_to_prompt(toxic_prompt, trigger_tokens, tokenizer, token_positions=token_positions)
        else:
            return apply_trigger_to_prompt(toxic_prompt, trigger_tokens, tokenizer, mode=mode)
    except Exception as e:
        print(f"Warning: Could not apply trigger: {e}")
        return toxic_prompt


# --- MANDATORY ENTRY POINT for Official Evaluation ---
def evaluate_rewrite(toxic_prompt: str) -> str:
    """
    [MANDATORY] This is the final, official algorithm entry point.
    
    Loads pre-trained trigger with per-token positions,
    and applies them to rewrite the prompt.
    """
    trigger_path = "models/trigger.json"
    if not os.path.exists(trigger_path):
        return algorithm_1(toxic_prompt)
    
    trigger_data = _load_cached_trigger(trigger_path)
    if trigger_data is None:
        return algorithm_1(toxic_prompt)
    
    trigger_tokens, token_positions = trigger_data
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
        
        # Apply trigger using per-token positions
        result = apply_trigger_to_prompt(
            toxic_prompt,
            trigger_tokens,
            tokenizer,
            token_positions=token_positions
        )
        return result
    
    except Exception as e:
        print(f"Warning: Could not apply trigger: {e}")
        return toxic_prompt