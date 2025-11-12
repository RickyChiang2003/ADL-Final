import argparse
import json
import logging
import os

import datasets
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import DPOConfig, DPOTrainer

from eval import initialize_models, judge, move_model_to_host, move_model_to_device
from utils import (
    get_dataset,
    get_next_run_dir,
    get_prompt,
    load_config,
    sanitize_config,
)

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
    move_model_to_device()
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
    move_model_to_host()
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
    if accelerator.is_main_process:
        initialize_models(SAFETY_MODEL, USEFULNESS_MODEL, CHAT_MODEL)
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

    scores = []
    patience = 0
    for it in range(train_args["iteration"]):
        if accelerator.is_main_process:
            rewrite_dataset, score = rewrite(
                raw_dataset, accelerator.unwrap_model(model), tokenizer
            )
            with open("data/tmp.json", "w", encoding="utf-8") as f:
                json.dump(rewrite_dataset, f, ensure_ascii=False, indent=4)
            tokenizer.save_pretrained(train_args["output_dir"])
            logger.info(json.dumps({"average score": score}, indent=4))
            with open(
                os.path.join(train_args["output_dir"], "all_result.json"),
                "a",
            ) as f:
                json.dump({"average score": score}, f, indent=4)
        accelerator.wait_for_everyone()

        model = train(accelerator.unwrap_model(model), tokenizer, accelerator, args, it)

        if len(scores) != 0 and score < max(scores):
            patience += 1
        else:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {
                k: v.contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in model.state_dict().items()
            }
            unwrapped_model.save_pretrained(
                train_args["output_dir"],
                state_dict=state_dict,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )

        if patience >= train_args["patience"]:
            break


def train(model, tokenizer, accelerator, args, it):
    data_args = args["data"]
    train_args = args["train"]

    with open("data/tmp.json", "r", encoding="utf-8") as f:
        rewrite = json.load(f)
    rewrite_dataset = Dataset.from_dict(rewrite).train_test_split(test_size=0.1)
    train_dataset = rewrite_dataset["train"]
    test_dataset = rewrite_dataset["test"]

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
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=callbacks,
    )

    trainer.train()
    return trainer.model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="the path of the config file", default="config.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
