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
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from eval_no_warning import (
    initialize_models,
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
REWRITE_MODEL = "models/rewrite"
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


def main(args):
    train_args = args["train"]
    model_args = args["model"]
    set_seed(args["seed"])
    accelerator = accelerator_setup(train_args)

    tokenizer = AutoTokenizer.from_pretrained(model_args["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args["name"],
        dtype=torch.bfloat16,
        num_labels=2,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = accelerator.prepare(model)

    raw_dataset = get_dataset(args["data"])["train"]
    if train_args["debug"]:
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
    train(accelerator.unwrap_model(model), tokenizer, accelerator, args)


def train(model, tokenizer, accelerator, args):
    data_args = args["data"]
    train_args = args["train"]

    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    rewrite_dataset = Dataset.from_list(results).train_test_split(test_size=0.1)

    def preprocess_function(examples):
        rewrite = examples["rewrite"]
        labels = examples["score"]
        tokenized_examples = tokenizer(
            rewrite,
            #padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            #return_tensors="pt",
        )
        tokenized_examples["labels"] = [[1 - label, label] for label in labels]
        #print(tokenized_examples)
        return tokenized_examples

    remove_columns = [
        "prompt",
        "rewrite",
        "chat_response",
        "relevance_score",
        "score",
        "safety_score",
    ]
    with accelerator.main_process_first():
        train_dataset = rewrite_dataset["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=remove_columns,
        )
        eval_dataset = rewrite_dataset["test"].map(
            preprocess_function,
            batched=True,
            remove_columns=remove_columns,
        )
    experiment_config = sanitize_config(args)
    accelerator.init_trackers(
        os.path.basename(train_args["logs_dir"]), experiment_config
    )

    trainer_args = TrainingArguments(
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
        #logging_steps=train_args["check_val_every_n_step"] // 3,
        logging_steps=max(1, train_args["check_val_every_n_step"] // 3),
        logging_dir=train_args["logs_dir"],
        load_best_model_at_end=True,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=train_args["patience"])]
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
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
