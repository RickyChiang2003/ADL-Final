import argparse
import json
import logging
import os
from datetime import timedelta

import datasets
import torch
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list, gather_object, set_seed
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from eval_no_warning import (
    initialize_models,
    move_model_to_device,
    move_model_to_host,
    batch_judge,
)
from utils import (
    get_dataset,
    get_next_run_dir,
    get_prompt,
    load_config,
)

SAFETY_MODEL = "models/guard"
USEFULNESS_MODEL = "models/usefulness"
CHAT_MODEL = "models/chat"
REWRITE_MODEL = "models/rewrite"
MAX_LENGTH = 2048
RESULT_FILE = "data/results.json"
NUM_RETURN_SEQUENCES = 24


logger = get_logger(__name__)


def accelerator_setup():
    kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=50000))
    accelerator = Accelerator(
        log_with=None,
        kwargs_handlers=[kwargs],
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    return accelerator


def rewrite(raw_dataset, model, tokenizer, accelerator):
    tokenized_instructions = tokenizer(
        list(raw_dataset["instruction"]),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    total = len(tokenized_instructions["input_ids"])
    per_device = total // accelerator.num_processes
    start = accelerator.process_index * per_device
    end = (
        total
        if accelerator.process_index == accelerator.num_processes - 1
        else (start + per_device)
    )

    # only process this slice
    slice_input_ids = tokenized_instructions["input_ids"][start:end]
    slice_mask = tokenized_instructions["attention_mask"][start:end]
    slice_instruction = raw_dataset["instruction"][start:end]
    slice_prompt = raw_dataset["prompt"][start:end]
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.to(accelerator.device)
    

    outputs = []
    print("*** Generating Output ***")
    for t, mask in tqdm(
        zip(
            slice_input_ids,
            slice_mask,
        ),
        total=end - start,
    ):
        t = t.unsqueeze(0).to(accelerator.device)
        mask = mask.unsqueeze(0).to(accelerator.device)
        with torch.no_grad():
            output = unwrapped_model.generate(
                input_ids=t,
                attention_mask=mask,
                max_new_tokens=256,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=NUM_RETURN_SEQUENCES,
            )
        for out in output:
            outputs.append(out[len(t[0]) :])
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    scores = []
    results = []
    print("*** Judging Output ***")
    move_model_to_device(accelerator.device)
    temp_file = f"data/results_rank_{accelerator.process_index}.jsonl"
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Rank {accelerator.process_index}: Removed existing temp file.")

    for i in tqdm(range(end - start)):
        start_idx = i * NUM_RETURN_SEQUENCES
        end_idx = (i + 1) * NUM_RETURN_SEQUENCES
        current_rewrites = outputs[start_idx : end_idx]
        current_prompts = [slice_prompt[i]] * len(current_rewrites)
        # (Batch Inference)
        batch_results = batch_judge(current_rewrites, current_prompts)
        for j, res in enumerate(batch_results):
            res_obj = {
                "prompt": slice_instruction[i],
                "rewrite": current_rewrites[j],
                "safety_score": res["safety_score"],
                "relevance_score": res["relevance_score"],
                "score": res["safety_score"] * res["relevance_score"],
                "chat_response": res["chat_response"]
            }
            
            with open(temp_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(res_obj, ensure_ascii=False) + "\n")

def main(args):
    set_seed(args.seed)
    accelerator = accelerator_setup()

    initialize_models(SAFETY_MODEL, USEFULNESS_MODEL, CHAT_MODEL)
    move_model_to_host()
    accelerator.wait_for_everyone()
    raw_dataset = get_dataset({"file": args.data})["train"]
    if args.debug:
        raw_dataset = raw_dataset.select(range(5))

    n = len(raw_dataset["prompt"])
    prompts = []
    for i in range(n):
        prompts.append(get_prompt(raw_dataset["prompt"][i]))
    raw_dataset = raw_dataset.add_column("instruction", prompts)

    rewrite_tokenizer = AutoTokenizer.from_pretrained(REWRITE_MODEL)
    if rewrite_tokenizer.pad_token_id is None:
        rewrite_tokenizer.pad_token_id = rewrite_tokenizer.eos_token_id
    if rewrite_tokenizer.bos_token_id is None:
        rewrite_tokenizer.bos_token_id = rewrite_tokenizer.eos_token_id
    rewrite_model = AutoModelForCausalLM.from_pretrained(REWRITE_MODEL)
    rewrite_model = accelerator.prepare(rewrite_model)
    rewrite(
        raw_dataset, rewrite_model, rewrite_tokenizer, accelerator
    )
    if accelerator.is_main_process:
        all_data = []
        for rank in range(accelerator.num_processes):
            fname = f"data/results_rank_{rank}.jsonl"
            if os.path.exists(fname):
                with open(fname, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line))
                # os.remove(fname)
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved {len(all_data)} results to {RESULT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", help="enable the debug mode", action="store_true", default=False
    )
    parser.add_argument("--output", help="output file", default=RESULT_FILE)
    parser.add_argument("--seed", help="random seed", default=1126)
    parser.add_argument("--data", help="input data file", default="data/data.parquet")
    args = parser.parse_args()
    main(args)
