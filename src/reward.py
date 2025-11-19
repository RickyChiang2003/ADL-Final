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
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerArguments,
)

from eval_no_warning import (
    initialize_models,
    judge,
    move_model_to_device,
    move_model_to_host,
)
from utils import (
    get_dataset,
    get_next_run_dir,
    get_prompt,
    load_config,
    sanitize_config,
)

SAFETY_MODEL = "models/guard"
USEFULNESS_MODEL = "models/usefulness"
CHAT_MODEL = "models/chat"
MAX_LENGTH = 2048
RESULT_FILE = "data/results.json"


logger = get_logger(__name__)


def accelerator_setup(train_args):
    kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=5000))
    accelerator = Accelerator(
        log_with="all",
        project_dir=train_args["logs_dir"],
        kwargs_handlers=[kwargs],
        gradient_accumulation_steps=train_args["gradient_accumulation_steps"],
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

    if accelerator.is_main_process:
        prefix = "debug" if train_args["debug"] else "exp"
        train_args["output_dir"] = get_next_run_dir(
            base_dir=train_args["output_dir"], prefix=prefix
        )
        train_args["logs_dir"] = get_next_run_dir(
            base_dir=train_args["logs_dir"], prefix=prefix
        )
    accelerator.wait_for_everyone()
    train_args["output_dir"] = broadcast_object_list(
        [train_args["output_dir"]], from_process=0
    )[0]
    train_args["logs_dir"] = broadcast_object_list(
        [train_args["logs_dir"]], from_process=0
    )[0]
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
                num_return_sequences=16,
            )
        for out in output:
            outputs.append(out[len(t[0]) :])
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    scores = []
    results = []
    print("*** Judging Output ***")
    move_model_to_device(accelerator.device)
    for i in tqdm(range(end - start)):
        for j in range(8):
            res = judge(outputs[i * 8 + j], slice_prompt[i])
            outputs[i * 8 + j]
            res["prompt"] = slice_instruction[i]
            res["rewrite"] = outputs[i * 8 + j]
            res["score"] = res["safety_score"] * res["relevance_score"]
            results.append(res)
            scores.append(res["score"])
    move_model_to_host()
    gathered_results = gather_object(results)
    gathered_scores = gather_object(scores)
    return gathered_results, gathered_scores


def main(args):
    # initialize configuration
    train_args = args["train"]
    model_args = args["model"]
    set_seed(args["seed"])

    # get accelerator
    accelerator = accelerator_setup(train_args)

    # initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args["name"], dtype=torch.bfloat16
    )
    initialize_models(SAFETY_MODEL, USEFULNESS_MODEL, CHAT_MODEL)
    move_model_to_host()
    accelerator.wait_for_everyone()
    model = accelerator.prepare(model)

    raw_dataset = get_dataset(args["data"])["train"]
    if train_args["debug"]:
        raw_dataset = raw_dataset.select(range(5))

    n = len(raw_dataset["prompt"])
    prompts = []
    for i in range(n):
        prompts.append(get_prompt(raw_dataset["prompt"][i]))
    raw_dataset = raw_dataset.add_column("instruction", prompts)

    results, scores = rewrite(raw_dataset, model, tokenizer, accelerator)
    score = sum(scores) / (n * 8)
    if accelerator.is_main_process:
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(json.dumps({"average score": score}, indent=4))
        with open(
            os.path.join(train_args["output_dir"], "all_result.json"),
            "a",
        ) as f:
            json.dump({"average score": score}, f, indent=4)

    accelerator.wait_for_everyone()
    train(accelerator.unwrap_model(model), accelerator, args)


def train(model, accelerator, args):
    data_args = args["data"]
    train_args = args["train"]

    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    rewrite_dataset = Dataset.from_list(results).train_test_split(test_size=0.1)
    train_dataset = rewrite_dataset["train"]
    test_dataset = rewrite_dataset["test"]
    experiment_config = sanitize_config(args)
    accelerator.init_trackers(
        os.path.basename(train_args["logs_dir"]), experiment_config
    )

    trainer_args = TrainerArguments(
        output_dir=train_args["output_dir"],
        num_train_epochs=train_args["epochs"],
        gradient_accumulation_steps=train_args["gradient_accumulation_steps"],
        per_device_train_batch_size=data_args["batch_size"],
        per_device_eval_batch_size=data_args["batch_size"],
        learning_rate=train_args["learning_rate"],
        eval_strategy="steps",
        eval_steps=train_args["check_val_every_n_step"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="best",
        save_total_limit=1,
        logging_steps=train_args["check_val_every_n_step"] // 3,
        logging_dir=train_args["logs_dir"],
        load_best_model_at_end=True,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=train_args["patience"])]
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=callbacks,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="the path of the config file", default="config/reward.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
