from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .utils import get_prompt


# --- MANDATORY ENTRY POINT for Official Evaluation ---
def evaluate_rewrite(toxic_prompt: str) -> str:
    """
    [MANDATORY] This is the final, official algorithm entry point.
    It will be run if no --algorithm flag is specified in main.py.
    """
    return algorithm_1(toxic_prompt)


def algorithm_1(toxic_prompt: str) -> str:
    REWRITE_MODEL = "checkpoints/exp1"
    MAX_LENGTH = 2048
    model = AutoModelForCausalLM.from_pretrained(
        REWRITE_MODEL, dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(REWRITE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    instruction = get_prompt(toxic_prompt)
    tokenized_instruction = tokenizer(
        instruction,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    t = tokenized_instruction["input_ids"].to("cuda")
    mask = tokenized_instruction["attention_mask"].to("cuda")
    output = model.generate(
        input_ids=t,
        attention_mask=mask,
        max_new_tokens=128,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    output = output[0][len(t[0]) :]
    output = tokenizer.decode(output, skip_special_tokens=True)
    return output
