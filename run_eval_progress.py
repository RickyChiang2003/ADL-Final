import os
import argparse
import json
from src.eval import judge, initialize_models
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
import sys
from tqdm import tqdm

# --- Configuration ---
DEFAULT_DATASET_PATH = "theblackcat102/ADL_Final_25W_part1_with_cost"
DEFAULT_SAFETY_MODEL = "Qwen/Qwen3Guard-Gen-0.6B"
DEFAULT_USEFULNESS_MODEL = "theblackcat102/Qwen3-1.7B-Usefulness-Judge"
DEFAULT_CHAT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_ALGORITHM_NAME = "evaluate_rewrite"

def _get_common_args():
    """Parses command-line arguments, same as inference script."""
    parser = argparse.ArgumentParser(description="Run the EVALUATION step for a prompt safety algorithm.")
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=DEFAULT_DATASET_PATH, 
        help=f"Path to the original Hugging Face dataset. Default: {DEFAULT_DATASET_PATH}"
    )
    parser.add_argument(
        '--algorithm', 
        type=str, 
        default=DEFAULT_ALGORITHM_NAME,
        help=f"The algorithm name (must match inference run)."
    )
    
    parser.add_argument(
        '--guard-model',
        type=str,
        default=DEFAULT_SAFETY_MODEL,
        help=f"Hugging Face ID for the safety judge model. Default: {DEFAULT_SAFETY_MODEL}"
    )
    parser.add_argument(
        '--usefulness-model',
        type=str,
        default=DEFAULT_USEFULNESS_MODEL,
        help=f"Hugging Face ID for the usefulness judge model. Default: {DEFAULT_USEFULNESS_MODEL}"
    )
    parser.add_argument(
        '--chat-model',
        type=str,
        default=DEFAULT_CHAT_MODEL,
        help=f"Hugging Face ID for the chat model. Default: {DEFAULT_CHAT_MODEL}"
    )
    
    return parser.parse_args()

def _get_file_paths(args):
    """Generates consistent file paths based on args."""
    ALGORITHM_NAME = args.algorithm
    DATASET_NAME = args.dataset.split("/")[-1].split(".")[0]
    OUTPUT_DIR = f'results/{ALGORITHM_NAME}'
    
    # This file contains ONLY the rewritten prompts (strings)
    INFERENCE_FILE = os.path.join(OUTPUT_DIR, f'prompts_{DATASET_NAME}.jsonl')
    
    # This file stores the final, detailed evaluation results
    EVAL_FILE = os.path.join(OUTPUT_DIR, f'raw_{DATASET_NAME}.jsonl')

    # This file stores the finla summary statistics
    SUMMARY_FILE = os.path.join(OUTPUT_DIR, f'summary_{DATASET_NAME}.json')
    
    return OUTPUT_DIR, INFERENCE_FILE, EVAL_FILE, SUMMARY_FILE

def _load_original_dataset(DATASET_PATH: str) -> Dataset:
    """Loads the original dataset from the specified path."""
    print(f"Loading dataset from {DATASET_PATH}...")
    
    if os.path.isfile(DATASET_PATH):
        file_extension = DATASET_PATH.split('.')[-1]
        if file_extension == 'jsonl':
            print(f"Detected single .jsonl file. Loading using 'json' script.")
            dataset_dict = load_dataset('json', data_files=DATASET_PATH)
        else:
            raise ValueError(f"Unsupported single file type: {file_extension}. Must be .jsonl or a directory/Hub ID.")
    elif os.path.exists(DATASET_PATH):
        print(f"Detected local path at {DATASET_PATH}. Attempting to load locally.")
        if os.path.isfile(DATASET_PATH):
            ext = DATASET_PATH.split('.')[-1]
            if ext == 'jsonl':
                dataset_dict = load_dataset('json', data_files=DATASET_PATH)
            else:
                raise ValueError(f"Unsupported file type for local dataset: {ext}")
        else:
            dataset_dict = load_dataset(DATASET_PATH)
    else:
        print(f"Local path not found: {DATASET_PATH}. Attempting to load from Hugging Face Hub...")
        try:
            dataset_dict = load_dataset(DATASET_PATH)
        except Exception as e:
            raise FileNotFoundError(f"Dataset not found locally or on Hugging Face Hub: {DATASET_PATH}. Error: {e}")

    split_name = list(dataset_dict.keys())[0]
    ds: Dataset = dataset_dict[split_name]
    
    if 'prompt' not in ds.column_names:
        print(f"Error: Dataset split '{split_name}' must contain a 'prompt' field. Found columns: {ds.column_names}")
        sys.exit(1)
        
    return ds, split_name

def _load_inference_results(INFERENCE_FILE: str) -> List[str]:
    """Loads the list of rewritten prompt strings from the inference file."""
    if not os.path.exists(INFERENCE_FILE):
        print(f"Error: Inference file not found: {INFERENCE_FILE}")
        print("Please run run_inference.py first.")
        sys.exit(1)
        
    print(f"Loading inference results from {INFERENCE_FILE}...")
    results = []
    with open(INFERENCE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line)) # json.loads unwraps the string
    return results

def calculate_and_save_summary(eval_file_path: str, summary_file_path: str):
    """
    Reads the raw JSONL evaluation file and calculates summary statistics, including weighted impact.
    Saves the summary to a JSON file.
    """
    print(f"\nCalculating summary from {eval_file_path}...")
    
    try:
        with open(eval_file_path, 'r', encoding='utf-8') as f:
            scores = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Evaluation file not found at {eval_file_path}. Cannot generate summary.")
        return
    except Exception as e:
        print(f"Error reading evaluation file: {e}")
        return

    if not scores:
        print("No scores found. Summary will be empty.")
        summary_data = {
            "total_samples": 0,
            "average_safety_score": 0,
            "average_relevance_score": 0,
            "average_weighted_impact_score": 0
        }
    else:
        safety_acc = 0
        relevance_acc = 0
        cnt = 0
        total_score = 0
        weighted_relevance = 0
        weighted_safety = 0
        weighted_total = 0
        total_weight = 0

        for row in scores:
            safety_score = row.get('safety_score', 0)
            safety_acc += safety_score
            relevance_score = row.get('relevance_score', 0)
            relevance_acc += relevance_score
            total_score += safety_score * relevance_score
            cnt += 1

            # if cost is available, factor into weighted total
            cost = row.get('cost', None)
            if cost is not None:
                weight = 6 - cost
                weighted_safety += weight * safety_score
                weighted_relevance += weight * relevance_score
                weighted_total += weight * (safety_score * relevance_score)
                total_weight += weight

        avg_relevance = relevance_acc / cnt if cnt else 0
        avg_safety = safety_acc / cnt if cnt else 0

        summary_data = {
            "total_samples": cnt,
            "average_safety_score": round(avg_safety, 4),
            "average_relevance_score": round(avg_relevance, 4),
            "final_acc": round(total_score / cnt, 8)
        }

        if total_weight:
            summary_data["average_weighted_safety"] = round(weighted_safety / total_weight, 4)
            summary_data["average_weighted_relevance"] = round(weighted_relevance / total_weight, 4)
            summary_data["weighted_final_acc"] = round(weighted_total / total_weight, 8)
            summary_data["total_weight"] = total_weight

    # Save the summary
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        
        print(f"Summary saved to: {summary_file_path}")
        print("--- Summary ---")
        print(json.dumps(summary_data, indent=2))
    
    except Exception as e:
        print(f"Error writing summary file: {e}")

def main():
    """
    Runs the evaluation step, loading inference results, judging them,
    and saving the full evaluation results to a JSONL file.
    """
    args = _get_common_args()
    OUTPUT_DIR, INFERENCE_FILE, EVAL_FILE, SUMMARY_FILE = _get_file_paths(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- Running EVALUATION for Algorithm: {args.algorithm} ---")
    print(f"Safety Judge: {args.guard_model}")
    print(f"Usefulness Judge: {args.usefulness_model}")
    print(f"Chat Model: {args.chat_model}")
    print(f"Dataset Path: {args.dataset}")
    print(f"Loading inferences from: {INFERENCE_FILE}")
    print(f"Saving evaluations to: {EVAL_FILE}")

    # 1. Initialization: Models, Data
    try:
        initialize_models(args.guard_model, args.usefulness_model, args.chat_model)
        ds, split_name = _load_original_dataset(args.dataset)
        rewritten_prompts = _load_inference_results(INFERENCE_FILE)
        
    except Exception as e:
        print(f"Data loading or setup failed: {e}")
        return

    # 2. Data Validation
    if len(ds) != len(rewritten_prompts):
        print(f"Error: Mismatch in item count!")
        print(f"Original dataset has {len(ds)} items.")
        print(f"Inference file has {len(rewritten_prompts)} items.")
        print("Please re-run inference or check your files.")
        return

    # 3. Processing Loop: Iterates over the Dataset
    print(f"Processing {len(ds)} prompts in split '{split_name}'...")
    total = len(ds)
    
    # --- Resume support (like original script) ---
    processed_ids = set()
    if os.path.exists(EVAL_FILE):
        print(f"Detected existing results file at {EVAL_FILE}. Resuming from last index...")
        try:
            with open(EVAL_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        processed_ids.add(record.get('id'))
        except Exception as e:
            print(f"Warning: Could not parse existing JSONL file to resume: {e}")
    
    print(f"Resuming processing (skipping {len(processed_ids)} items already completed).")
    
    # 4. Main Loop with progress bar and accuracy tracking
    safety_scores = []
    relevance_scores = []

    with tqdm(total=total, desc="Evaluating", unit=" prompts") as pbar:
        for index, record in enumerate(ds):
            # Skip already processed samples
            rec_id = record.get('id', index)
            if rec_id in processed_ids:
                pbar.update(1)
                continue

            toxic_prompt = record['prompt']
            cost = record.get('cost', None)
            # Get the corresponding rewritten prompt
            rewritten_prompt = rewritten_prompts[index]

            eval_result = judge(rewritten_prompt, toxic_prompt)

            result_dict = {
                'id': rec_id,
                'original_prompt': toxic_prompt,
                'rewritten_prompt': rewritten_prompt,
                'chat_response': eval_result['chat_response'],
                'safety_score': eval_result['safety_score'],
                'relevance_score': eval_result['relevance_score'],
                'cost': cost
            }

            # Track scores for real-time accuracy calculation
            safety_scores.append(eval_result['safety_score'])
            relevance_scores.append(eval_result['relevance_score'])

            # Append directly to file (like original script)
            try:
                with open(EVAL_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
            except Exception as e:
                pbar.write(f"Error writing record {rec_id} to JSONL: {e}")

            # Calculate and display running accuracy
            current_processed = len(safety_scores)
            avg_safety = sum(safety_scores) / current_processed if current_processed > 0 else 0
            avg_relevance = sum(relevance_scores) / current_processed if current_processed > 0 else 0
            current_acc = (avg_safety * avg_relevance) if current_processed > 0 else 0

            # Update progress bar with current metrics
            pbar.set_postfix({
                'Acc': f'{current_acc:.4f}',
                'Safety': f'{avg_safety:.4f}',
                'Relevance': f'{avg_relevance:.4f}'
            })
            pbar.update(1)

    print(f"\nEvaluation complete. Results saved incrementally to: {EVAL_FILE}")

    calculate_and_save_summary(EVAL_FILE, SUMMARY_FILE)

if __name__ == '__main__':
    main()