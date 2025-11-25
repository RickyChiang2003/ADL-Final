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
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, AutoModelForSequenceClassification
from trl import DPOConfig, DPOTrainer

from eval_no_warning import (
    initialize_models,
    batch_judge,
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
#REWARD_CHECKPOINT = "reward_evalloss_067"
#REWARD_BACKBONE = "Qwen/Qwen3-0.6B"
MAX_LENGTH = 2048
PREFERENCE_FILE = "data/preferences.json"
NUM_RETURN_SEQUENCES = 8

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
    print(f"*** Generating Output (Rank {accelerator.process_index}) ***")
    
    for t, mask in tqdm(zip(slice_input_ids, slice_mask), total=end - start):
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
                num_return_sequences=8, # 維持 8 個樣本以供比較
            )
        for out in output:
            outputs.append(out[len(t[0]) :])
            
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    preferences = []
    scores_history = []

    print(f"*** Judging Output (Rank {accelerator.process_index}) ***")

    for i in tqdm(range(end - start)):
        start_idx = i * 8 # 假設 num_return_sequences=8
        end_idx = (i + 1) * 8
        current_rewrites = outputs[start_idx : end_idx]
        current_prompts = [slice_prompt[i]] * len(current_rewrites)

        # 2. 呼叫 Batch Judge 
        batch_results = batch_judge(current_rewrites, current_prompts)

        # 3. 計算分數與構建 DPO Pairs
        current_scores = []
        for res in batch_results:
            score = res["safety_score"] * res["relevance_score"]
            current_scores.append(score)

        scores_history.extend(current_scores)

        for j in range(7):
            score1 = current_scores[j]
            for k in range(j, 8):
                score2 = current_scores[k]
                if score1 == score2:
                    continue

                if score1 < score2:
                    preferences.append({
                        "prompt": slice_instruction[i],
                        "chosen": current_rewrites[k],
                        "rejected": current_rewrites[j],
                    })
                else:
                    preferences.append({
                        "prompt": slice_instruction[i],
                        "chosen": current_rewrites[j],
                        "rejected": current_rewrites[k],
                    })

    gathered_preferences = gather_object(preferences)
    gathered_scores = gather_object(scores_history)

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
    #move_model_to_host()
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
        "--config", help="the path of the config file", default="config/train.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
