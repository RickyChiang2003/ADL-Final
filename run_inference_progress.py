import os
import argparse
import json
from src.agent import PromptSafetyAgent
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
import sys
from tqdm import tqdm

# --- Configuration ---
DEFAULT_ALGORITHM_NAME = PromptSafetyAgent.MANDATORY_ENTRY_POINT
DEFAULT_DATASET_PATH = "theblackcat102/ADL_Final_25W_part1_with_cost"

def _get_common_args():
    """Parses command-line arguments, same as eval script."""
    parser = argparse.ArgumentParser(description="Run the INFERENCE step for a prompt safety algorithm.")
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=DEFAULT_DATASET_PATH, 
        help=f"Path to the Hugging Face dataset. Default: {DEFAULT_DATASET_PATH}"
    )
    parser.add_argument(
        '--algorithm', 
        type=str, 
        default=DEFAULT_ALGORITHM_NAME, 
        help=f"The algorithm function name to test. Defaults to '{DEFAULT_ALGORITHM_NAME}'."
    )
    
    return parser.parse_args()

def _get_file_paths(args):
    """Generates consistent file paths based on args."""
    ALGORITHM_NAME = args.algorithm
    DATASET_NAME = args.dataset.split("/")[-1].split(".")[0]
    OUTPUT_DIR = f'results/{ALGORITHM_NAME}'
    
    # This file stores ONLY the rewritten prompts (strings)
    INFERENCE_FILE = os.path.join(OUTPUT_DIR, f'prompts_{DATASET_NAME}.jsonl')
    
    # This file will be created by run_eval.py
    EVAL_FILE = os.path.join(OUTPUT_DIR, f'raw_{DATASET_NAME}.jsonl')
    
    return OUTPUT_DIR, INFERENCE_FILE, EVAL_FILE

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

def main():
    """
    Runs the inference step, saving rewritten prompts to a JSONL file.
    """
    args = _get_common_args()
    OUTPUT_DIR, INFERENCE_FILE, _ = _get_file_paths(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- Running INFERENCE for Algorithm: {args.algorithm} ---")
    print(f"Dataset Path: {args.dataset}")
    print(f"Output File: {INFERENCE_FILE}")

    # 1. Initialization: Data and Agent
    try:
        ds, split_name = _load_original_dataset(args.dataset)
        agent = PromptSafetyAgent(args.algorithm)
        
    except Exception as e:
        print(f"Dataset loading or setup failed: {e}")
        return

    # 2. Processing Loop: Iterates over the Dataset
    print(f"Processing {len(ds)} prompts in split '{split_name}'...")
    total = len(ds)
    
    # --- Resume support by line count ---
    start_index = 0
    if os.path.exists(INFERENCE_FILE):
        print(f"Detected existing results file at {INFERENCE_FILE}.")
        try:
            with open(INFERENCE_FILE, 'r', encoding='utf-8') as f:
                start_index = len(f.readlines())
        except Exception as e:
            print(f"Warning: Could not parse existing JSONL file to resume: {e}")
    
    print(f"Resuming processing from index {start_index}/{total} (skipping {start_index} items already completed).")
    
    try:
        # Use 'a' (append) mode for resilient, incremental writing
        with open(INFERENCE_FILE, 'a', encoding='utf-8') as f:
            pbar = tqdm(enumerate(ds), total=total, initial=start_index, desc="Processing", unit=" prompts")
            for index, record in pbar:
                # Skip already processed samples
                if index < start_index:
                    continue

                toxic_prompt = record['prompt']
                rewritten_prompt = agent.rewrite(toxic_prompt)

                # Save as a JSON-encoded string, as requested
                try:
                    f.write(json.dumps(rewritten_prompt, ensure_ascii=False) + '\n')
                except Exception as e:
                    rec_id = record.get('id', index)
                    pbar.write(f"Error writing record {rec_id} to JSONL: {e}")

    except KeyboardInterrupt:
        print("\nInference interrupted. Results saved up to the last completed item.")
        
    print(f"\nInference complete. Rewritten prompts saved to: {INFERENCE_FILE}")

if __name__ == '__main__':
    main()