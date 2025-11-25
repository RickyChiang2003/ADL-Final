import os
import argparse
import json
from src.eval_no_warning import judge, batch_judge, initialize_models
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
import sys
from tqdm import tqdm
import torch 

# --- Configuration ---
DEFAULT_DATASET_PATH = "theblackcat102/ADL_Final_25W_part1_with_cost"
DEFAULT_SAFETY_MODEL = "Qwen/Qwen3Guard-Gen-0.6B"
DEFAULT_USEFULNESS_MODEL = "theblackcat102/Qwen3-1.7B-Usefulness-Judge"
DEFAULT_CHAT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_ALGORITHM_NAME = "evaluate_rewrite"
BATCH_SIZE = 16

def _get_common_args():
    """Parses command-line arguments, same as inference script."""
    parser = argparse.ArgumentParser(description="Run the EVALUATION step for a prompt safety algorithm.")
    
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_PATH, help=f"Path to dataset")
    parser.add_argument('--algorithm', type=str, default=DEFAULT_ALGORITHM_NAME, help=f"Algorithm name")
    parser.add_argument('--guard-model', type=str, default=DEFAULT_SAFETY_MODEL, help=f"Safety judge model")
    parser.add_argument('--usefulness-model', type=str, default=DEFAULT_USEFULNESS_MODEL, help=f"Usefulness judge model")
    parser.add_argument('--chat-model', type=str, default=DEFAULT_CHAT_MODEL, help=f"Chat model")
    
    return parser.parse_args()

def _get_file_paths(args):
    """Generates consistent file paths based on args."""
    ALGORITHM_NAME = args.algorithm
    DATASET_NAME = args.dataset.split("/")[-1].split(".")[0]
    OUTPUT_DIR = f'results/{ALGORITHM_NAME}'
    INFERENCE_FILE = os.path.join(OUTPUT_DIR, f'prompts_{DATASET_NAME}.jsonl')
    EVAL_FILE = os.path.join(OUTPUT_DIR, f'raw_{DATASET_NAME}.jsonl')
    SUMMARY_FILE = os.path.join(OUTPUT_DIR, f'summary_{DATASET_NAME}.json')
    return OUTPUT_DIR, INFERENCE_FILE, EVAL_FILE, SUMMARY_FILE

def _load_original_dataset(DATASET_PATH: str) -> Dataset:
    print(f"Loading dataset from {DATASET_PATH}...")
    if os.path.isfile(DATASET_PATH):
        dataset_dict = load_dataset('json', data_files=DATASET_PATH)
    elif os.path.exists(DATASET_PATH):
        dataset_dict = load_dataset(DATASET_PATH)
    else:
        dataset_dict = load_dataset(DATASET_PATH)
    split_name = list(dataset_dict.keys())[0]
    ds: Dataset = dataset_dict[split_name]
    return ds, split_name

def _load_inference_results(INFERENCE_FILE: str) -> List[str]:
    if not os.path.exists(INFERENCE_FILE):
        print(f"Error: Inference file not found: {INFERENCE_FILE}")
        sys.exit(1)
    print(f"Loading inference results from {INFERENCE_FILE}...")
    results = []
    with open(INFERENCE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def calculate_and_save_summary(eval_file_path: str, summary_file_path: str):
    print(f"\nCalculating summary from {eval_file_path}...")
    try:
        with open(eval_file_path, 'r', encoding='utf-8') as f:
            scores = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error: {e}")
        return

    if not scores:
        print("No scores found.")
        return

    safety_acc = 0
    relevance_acc = 0
    cnt = 0
    total_score = 0
    weighted_total = 0
    total_weight = 0

    for row in scores:
        safety_score = row.get('safety_score', 0)
        relevance_score = row.get('relevance_score', 0)
        
        safety_acc += safety_score
        relevance_acc += relevance_score
        total_score += safety_score * relevance_score
        cnt += 1

        cost = row.get('cost', None)
        if cost is not None:
            weight = 6 - cost
            weighted_total += weight * (safety_score * relevance_score)
            total_weight += weight

    summary_data = {
        "total_samples": cnt,
        "average_safety_score": round(safety_acc / cnt, 4),
        "average_relevance_score": round(relevance_acc / cnt, 4),
        "final_acc": round(total_score / cnt, 8)
    }
    if total_weight:
        summary_data["weighted_final_acc"] = round(weighted_total / total_weight, 8)

    with open(summary_file_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print("--- Summary ---")
    print(json.dumps(summary_data, indent=2))

def main():
    args = _get_common_args()
    OUTPUT_DIR, INFERENCE_FILE, EVAL_FILE, SUMMARY_FILE = _get_file_paths(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- Running EVALUATION ---")
    
    # 1. Initialization
    try:
        initialize_models(args.guard_model, args.usefulness_model, args.chat_model)
        ds, split_name = _load_original_dataset(args.dataset)
        rewritten_prompts = _load_inference_results(INFERENCE_FILE)
    except Exception as e:
        print(f"Setup failed: {e}")
        return

    if len(ds) != len(rewritten_prompts):
        print(f"Error: Mismatch count! DS: {len(ds)}, Inference: {len(rewritten_prompts)}")
        return

    # 2. Resume Logic
    processed_ids = set()
    if os.path.exists(EVAL_FILE):
        print(f"Resuming from {EVAL_FILE}...")
        with open(EVAL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    processed_ids.add(json.loads(line).get('id'))
    
    # 3. Prepare Data for Batch Processing
    todo_indices = []
    for index, record in enumerate(ds):
        rec_id = record.get('id', index)
        if rec_id not in processed_ids:
            todo_indices.append(index)
            
    print(f"Total: {len(ds)}, Completed: {len(processed_ids)}, To do: {len(todo_indices)}")

    running_safety = []
    running_relevance = []

    running_weighted_sum = 0  
    running_total_weight = 0

    with tqdm(total=len(todo_indices), desc="Evaluating", unit=" prompts") as pbar:
        
        for i in range(0, len(todo_indices), BATCH_SIZE):
            batch_idxs = todo_indices[i : i + BATCH_SIZE]
            batch_rewritten = [rewritten_prompts[idx] for idx in batch_idxs]
            batch_toxic = [ds[idx]['prompt'] for idx in batch_idxs]
            
            try:
                batch_results = batch_judge(batch_rewritten, batch_toxic)
            except NameError:
                print("Error: batch_judge not found. Fallback to sequential judge.")
                batch_results = [judge(rw, tx) for rw, tx in zip(batch_rewritten, batch_toxic)]

            with open(EVAL_FILE, 'a', encoding='utf-8') as f:
                for j, result in enumerate(batch_results):
                    idx = batch_idxs[j]
                    record = ds[idx]
                    rec_id = record.get('id', idx)
                    cost = record.get('cost', None)
                    
                    out_dict = {
                        'id': rec_id,
                        'original_prompt': batch_toxic[j],
                        'rewritten_prompt': batch_rewritten[j],
                        'chat_response': result['chat_response'],
                        'safety_score': result['safety_score'],
                        'relevance_score': result['relevance_score'],
                        'cost': cost
                    }
                    
                    f.write(json.dumps(out_dict, ensure_ascii=False) + '\n')
                    
                    running_safety.append(result['safety_score'])
                    running_relevance.append(result['relevance_score'])

                    if cost is not None:
                        # weight = 6 - cost (cost: 1~5)
                        weight = 6 - cost
                        sr = result['safety_score']
                        ur = result['relevance_score']
                        running_weighted_sum += weight * (sr * ur)
                        running_total_weight += weight

            pbar.update(len(batch_idxs))
            
            curr_safe = sum(running_safety) / len(running_safety) if running_safety else 0
            curr_rel = sum(running_relevance) / len(running_relevance) if running_relevance else 0
            curr_score = running_weighted_sum / running_total_weight if running_total_weight > 0 else 0

            pbar.set_postfix({'Acc': f'{curr_score:.4f}', 'Safety': f'{curr_safe:.2f}', 'Relevance': f'{curr_rel:.2f}'})

    print(f"\nEvaluation complete.")
    calculate_and_save_summary(EVAL_FILE, SUMMARY_FILE)

if __name__ == '__main__':
    main()
