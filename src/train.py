import argparse
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List

import datasets
from datasets import Dataset
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers import EarlyStoppingCallback
from eval import judge, initialize_models
from utils import (
    get_dataset,
    get_next_run_dir,
    get_prompt,
    load_config,
    sanitize_config,
)

from trl import DPOConfig, DPOTrainer

REWRITE_MODEL = "models/rewrite"
SAFETY_MODEL = "models/guard"
USEFULNESS_MODEL = "models/usefulness"
CHAT_MODEL = "models/chat"
MAX_LENGTH = 2048

logger = get_logger(__name__)


def accelerator_setup(train_args, isTest):
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    if not isTest:
        accelerator = Accelerator(
            log_with="all",
            project_dir=train_args["logs_dir"],
            gradient_accumulation_steps=train_args["gradient_accumulation_steps"],
        )
    else:
        accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
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

    if accelerator.is_main_process and not isTest:
        prefix = "debug" if train_args["debug"] else "exp"
        train_args["output_dir"] = get_next_run_dir(
            base_dir=train_args["output_dir"], prefix=prefix
        )
        train_args["logs_dir"] = get_next_run_dir(
            base_dir=train_args["logs_dir"], prefix=prefix
        )
    accelerator.wait_for_everyone()
    return accelerator


def rewrite(raw_dataset, model, tokenizer):
    tokenized_instructions = tokenizer(
        list(raw_dataset["instruction"]),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    outputs = []
    print("*** Generating Output ***")
    for t, mask in tqdm(
        zip(
            tokenized_instructions["input_ids"],
            tokenized_instructions["attention_mask"],
        ),
        total=len(tokenized_instructions["input_ids"]),
    ):
        t = t.unsqueeze(0).to("cuda")
        mask = mask.unsqueeze(0).to("cuda")
        output = model.generate(
            input_ids=t,
            attention_mask=mask,
            max_new_tokens=128,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=8,
        )
        for out in output:
            outputs.append(out[len(t[0]) :])

    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    n = len(raw_dataset["instruction"])
    preference = {}
    preference["prompt"] = []
    preference["chosen"] = []
    preference["rejected"] = []
    score = 0
    print("*** Judging Output ***")
    for i in tqdm(range(n)):
        result = []
        for j in range(8):
            res = judge(outputs[i * 8 + j], raw_dataset["prompt"][i])
            outputs[i * 8 + j]
            res["prompt"] = raw_dataset["instruction"][i]
            res["rewrite"] = outputs[i * 8 + j]
            with open("out.tmp", "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            result.append(res)
            score += res["safety_score"] * res["relevance_score"]
        for j in range(7):
            score1 = result[j]["safety_score"] * result[j]["relevance_score"]
            for k in range(j, 8):
                score2 = result[k]["safety_score"] * result[k]["relevance_score"]
                if score1 == score2:
                    continue
                elif score1 < score2:
                    preference["prompt"].append(raw_dataset["instruction"][i])
                    preference["chosen"].append(outputs[i * 8 + k])
                    preference["rejected"].append(outputs[i * 8 + j])
                else:
                    preference["prompt"].append(raw_dataset["instruction"][i])
                    preference["chosen"].append(outputs[i * 8 + j])
                    preference["rejected"].append(outputs[i * 8 + k])

    return preference, score / (8 * n)


def main(args):
    # initialize configuration
    train_args = args["train"]
    set_seed(args["seed"])

    # get accelerator
    accelerator = accelerator_setup(train_args, isTest=False)

    # initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(REWRITE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(REWRITE_MODEL, dtype=torch.bfloat16)
    initialize_models(SAFETY_MODEL, USEFULNESS_MODEL, CHAT_MODEL)

    model = accelerator.prepare(model)

    raw_dataset = get_dataset(args["data"])["train"]
    if train_args["debug"]:
        raw_dataset = raw_dataset.select(range(10))

    n = len(raw_dataset["prompt"])
    prompts = []
    for i in range(n):
        prompts.append(get_prompt(raw_dataset["prompt"][i]))
    raw_dataset = raw_dataset.add_column("instruction", prompts)

    scores = []
    patience = 0
    for it in range(train_args["iteration"]):
        rewrite_dataset, score = rewrite(raw_dataset, model, tokenizer)
        model = train(model, tokenizer, rewrite_dataset, accelerator, args, it)

        if len(scores) != 0 and score < max(scores):
            patience += 1
        else:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {
                k: v.contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in unwrapped_model.state_dict().items()
            }
            unwrapped_model.save_pretrained(
                train_args["output_dir"],
                state_dict=state_dict,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(train_args["output_dir"])
                logger.info(json.dumps({"average score": score}, indent=4))
                with open(
                    os.path.join(train_args["output_dir"], "all_result.json"),
                    "a",
                ) as f:
                    json.dump({"average score": score}, f, indent=4)
        if patience >= train_args["patience"]:
            break


def train(model, tokenizer, rewrites, accelerator, args, it):
    data_args = args["data"]
    train_args = args["train"]
    # # setup optimizer and lr scheduler
    # adapter_params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(adapter_params, lr=train_args["learning_rate"])
    # lr_scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode="max",
    #     factor=0.3,
    #     patience=train_args["patience"] // 2,
    # )

    experiment_config = sanitize_config(args)
    accelerator.init_trackers(
        os.path.basename(train_args["logs_dir"]), experiment_config
    )

    dpo_args = DPOConfig(
        output_dir=os.path.join(train_args["output_dir"], f"iteration_{it}"),
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
        logging_dir=os.path.join(train_args["logs_dir"], f"iteration_{it}"),
        load_best_model_at_end=True,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=train_args["patience"])]
    data = Dataset.from_dict(rewrites).train_test.split(test_size=0.1)
    train_dataset = data["train"]
    test_dataset = data["test"]
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=callbacks,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="the path of the config file", default="config.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
