from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .utils import get_prompt
from .eval_no_warning import (
    initialize_models,
    judge,
)


# --- MANDATORY ENTRY POINT for Official Evaluation ---
def evaluate_rewrite(toxic_prompt: str) -> str:
    """
    [MANDATORY] This is the final, official algorithm entry point.
    It will be run if no --algorithm flag is specified in main.py.
    """
    return dpo(toxic_prompt)


def dpo(toxic_prompt: str) -> str:
    REWRITE_MODEL = "models/rewrite"
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
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    output = output[0][len(t[0]) :]
    output = tokenizer.decode(output, skip_special_tokens=True)
    return output


model_initialized = False
SAFETY_MODEL = "models/guard"
USEFULNESS_MODEL = "models/usefulness"
CHAT_MODEL = "models/chat"


def dpo_with_judge(toxic_prompt: str) -> str:
    global model_initialized
    if not model_initialized:
        initialize_models(SAFETY_MODEL, USEFULNESS_MODEL, CHAT_MODEL)
        model_initialized = True
    REWRITE_MODEL = "models/rewrite"
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
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=8,
    )
    outputs = []
    for out in output:
        outputs.append(out[len(t[0]) :])

    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    scores = []
    for output in outputs:
        res = judge(output, toxic_prompt)
        score = res["safety_score"] * res["relevance_score"]
        scores.append(score)
        if score == 1.0:
            return output
    for output, score in zip(outputs, scores):
        if score == 0.5:
            return output
    return outputs[0]


def dpo_with_reward(toxic_prompt: str) -> str:
    raise NotImplementedError
