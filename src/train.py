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
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import DPOConfig, DPOTrainer

from eval import initialize_models, judge, move_model_to_device, move_model_to_host
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
PREFERENCE_FILE = "data/preferences.json"

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
        output = unwrapped_model.generate(
            input_ids=t,
            attention_mask=mask,
            max_new_tokens=256,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=8,
        )
        for out in output:
            outputs.append(out[len(t[0]) :])
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    preferences = []
    scores = []
    print("*** Judging Output ***")
    move_model_to_device(accelerator.device)
    for i in tqdm(range(end - start)):
        result = []
        for j in range(8):
            res = judge(outputs[i * 8 + j], slice_prompt[i])
            outputs[i * 8 + j]
            res["prompt"] = slice_instruction[i]
            res["rewrite"] = outputs[i * 8 + j]
            with open("out.tmp", "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            result.append(res)
            scores.append(res["safety_score"] * res["relevance_score"])
        for j in range(7):
            score1 = result[j]["safety_score"] * result[j]["relevance_score"]
            for k in range(j, 8):
                score2 = result[k]["safety_score"] * result[k]["relevance_score"]
                if score1 == score2:
                    continue
                elif score1 < score2:
                    preferences.append(
                        {
                            "prompt": slice_instruction[i],
                            "chosen": outputs[i * 8 + k],
                            "rejected": outputs[i * 8 + j],
                        }
                    )
                else:
                    preferences.append(
                        {
                            "prompt": slice_instruction[i],
                            "chosen": outputs[i * 8 + j],
                            "rejected": outputs[i * 8 + k],
                        }
                    )
    move_model_to_host()
    gathered_preferences = gather_object(preferences)
    gathered_scores = gather_object(scores)
    return gathered_preferences, gathered_scores


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

    model = AutoModelForCausalLM.from_pretrained(
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

    score_history = []
    patience = 0
    for it in range(train_args["iteration"]):
        preferences, scores = rewrite(raw_dataset, model, tokenizer, accelerator)
        score = sum(scores) / (n * 8)
        if accelerator.is_main_process:
            with open(PREFERENCE_FILE, "w", encoding="utf-8") as f:
                json.dump(preferences, f, ensure_ascii=False, indent=4)
            tokenizer.save_pretrained(train_args["output_dir"])
            logger.info(json.dumps({"average score": score}, indent=4))
            with open(
                os.path.join(train_args["output_dir"], "all_result.json"),
                "a",
            ) as f:
                json.dump({"average score": score}, f, indent=4)
        accelerator.wait_for_everyone()

        if len(score_history) != 0 and score < max(score_history):
            patience += 1
        else:
            patience = 0
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
        score_history.append(score)
        if patience >= train_args["patience"]:
            break

        model = train(accelerator.unwrap_model(model), tokenizer, accelerator, args, it)


def train(model, tokenizer, accelerator, args, it):
    data_args = args["data"]
    train_args = args["train"]

    with open(PREFERENCE_FILE, "r", encoding="utf-8") as f:
        rewrite = json.load(f)
    rewrite_dataset = Dataset.from_list(rewrite).train_test_split(test_size=0.1)
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
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=train_args["train_patience"])
    ]
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
