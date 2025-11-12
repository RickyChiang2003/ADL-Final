#!/usr/bin/env python3
"""
Train a universal soft-prefix trigger on unsafe prompts from the official dataset.

This script:
1. Loads the official dataset (from Hugging Face Hub or local file)
2. Filters for unsafe prompts (safety_score=0.0)
3. Weighted samples from unsafe prompts (higher cost = higher priority)
4. Optimizes a soft-prefix trigger using gradient-based method
5. Saves the learned trigger as discrete token IDs to models/trigger.json

Usage:
    python scripts/train_trigger.py \
      --data theblackcat102/ADL_Final_25W_part1_with_cost \
      --num-source-prompts 20 \
      --num-trigger-tokens 100 \
      --num-steps 1000 \
      --output models/trigger.json
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# Add src to path so we can import attacks module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attacks_ import (
    optimize_trigger,
    save_trigger,
    get_safety_score_via_guard,
)


def load_dataset_with_cost(dataset_path: str):
    """Load dataset from Hugging Face Hub or local file"""
    print(f"Loading dataset from {dataset_path}...")
    
    if os.path.isfile(dataset_path):
        # Local file
        if dataset_path.endswith('.jsonl'):
            dataset_dict = load_dataset('json', data_files=dataset_path)
        else:
            raise ValueError(f"Unsupported file type: {dataset_path}")
    elif os.path.isdir(dataset_path):
        # Local directory
        dataset_dict = load_dataset(dataset_path)
    else:
        # Try Hugging Face Hub
        try:
            dataset_dict = load_dataset(dataset_path)
        except Exception as e:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}. Error: {e}")
    
    # Get the first split
    split_name = list(dataset_dict.keys())[0]
    ds = dataset_dict[split_name]
    
    print(f"Loaded {len(ds)} records from '{split_name}' split")
    
    # Extract prompts and costs
    records = []
    for record in ds:
        records.append({
            'prompt': record.get('prompt', ''),
            'cost': record.get('cost', 3)  # Default cost = 3
        })
    
    return records



def main():
    parser = argparse.ArgumentParser(
        description="Train universal soft-prefix trigger on toxic prompts"
    )
    parser.add_argument(
        '--data',
        type=str,
        default='theblackcat102/ADL_Final_25W_part1_with_cost',
        help='Path to dataset (Hugging Face Hub ID or local JSONL file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/trigger.json',
        help='Output path for trigger tokens'
    )
    parser.add_argument(
        '--num-source-prompts',
        type=int,
        default=20,
        help='Number of unsafe prompts to use for training trigger'
    )
    parser.add_argument(
        '--num-trigger-tokens',
        type=int,
        default=5,
        help='Number of trigger tokens to optimize'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=50,
        help='Number of optimization steps'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate for optimization'
    )
    parser.add_argument(
        '--guard-model',
        type=str,
        default='Qwen/Qwen3Guard-Gen-0.6B',
        help='Safety guard model'
    )
    parser.add_argument(
        '--chat-model',
        type=str,
        default='unsloth/Llama-3.2-3B-Instruct',
        help='Chat model for optimization'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Weight for Guard Model loss (default: 1.0)'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1.0,
        help='Weight for Chat Model loss (default: 1.0)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.01,
        help='Weight for regularization loss (default: 0.01)'
    )
    parser.add_argument(
        '--low-memory',
        action='store_true',
        default=False,
        help='Enable low memory mode (offload models to CPU when not in use)'
    )
    parser.add_argument(
        '--load-in-8bit',
        action='store_true',
        help='Load models in 8-bit quantization to save VRAM'
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='Load models in 4-bit quantization to save VRAM (requires bitsandbytes)'
    )
    parser.add_argument(
        '--guard-eval-interval',
        type=int,
        default=1,
        help='Evaluate Guard Model every N steps (default: 5). Lower = more accurate but slower. Set to 1 for every step.'
    )
    parser.add_argument(
        '--cache-unsafe-prompts',
        type=str,
        default=None,
        help='Path to cache file for unsafe prompts (e.g., cache/unsafe_prompts.json). Will load from cache if exists, otherwise will create it.'
    )

    args = parser.parse_args()

    # Multi-GPU setup
    if not torch.cuda.is_available():
        print("Error: CUDA not available. This script requires GPUs.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    if num_gpus < 2:
        print("Warning: Only 1 GPU available. This may cause OOM. Recommend using 2+ GPUs.")
        guard_device = 'cuda:0'
        chat_device = 'cuda:0'
    else:
        # Distribute models across GPUs
        guard_device = 'cuda:0'
        chat_device = 'cuda:1'
        print(f"Multi-GPU mode: Guard Model → {guard_device}, Chat Model → {chat_device}")

    # Load dataset with cost information
    print(f"Loading prompts from {args.data}...")
    all_records = load_dataset_with_cost(args.data)
    
    if len(all_records) == 0:
        print(f"Error: No prompts loaded from {args.data}")
        return
    
    prompts = [r['prompt'] for r in all_records]
    costs = [r['cost'] for r in all_records]

    # Check if we can load unsafe prompts from cache
    unsafe_indices = None
    if args.cache_unsafe_prompts and os.path.exists(args.cache_unsafe_prompts):
        print(f"\n--- Loading unsafe prompts from cache ---")
        print(f"Cache file: {args.cache_unsafe_prompts}")
        try:
            with open(args.cache_unsafe_prompts, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Validate cache matches current dataset
            if cache_data.get('dataset_path') == args.data and cache_data.get('total_prompts') == len(prompts):
                unsafe_indices = cache_data['unsafe_indices']
                print(f"✓ Loaded {len(unsafe_indices)} unsafe prompts from cache")
                print(f"Skipping Guard Model filtering step (saving time!)")
            else:
                print(f"Warning: Cache doesn't match current dataset. Will re-filter.")
                print(f"  Cache dataset: {cache_data.get('dataset_path')} (total: {cache_data.get('total_prompts')})")
                print(f"  Current dataset: {args.data} (total: {len(prompts)})")
                unsafe_indices = None
        except Exception as e:
            print(f"Warning: Could not load cache file: {e}")
            unsafe_indices = None

    # If no valid cache, perform filtering
    if unsafe_indices is None:
        # First pass: evaluate safety scores to filter for unsafe prompts only
        print(f"\n--- Filtering for unsafe prompts (safety_score=0.0) ---")
        print(f"Loading guard model: {args.guard_model}...")

        # Prepare model loading kwargs for Guard Model
        guard_model_kwargs = {}
        if args.load_in_8bit:
            print("Loading Guard Model in 8-bit mode...")
            guard_model_kwargs['load_in_8bit'] = True
            guard_model_kwargs['device_map'] = guard_device
        elif args.load_in_4bit:
            print("Loading Guard Model in 4-bit mode...")
            guard_model_kwargs['load_in_4bit'] = True
            guard_model_kwargs['device_map'] = guard_device
        else:
            guard_model_kwargs['torch_dtype'] = torch.float16

        guard_tokenizer = AutoTokenizer.from_pretrained(args.guard_model)
        guard_model = AutoModelForCausalLM.from_pretrained(
            args.guard_model,
            **guard_model_kwargs
        )
        if not (args.load_in_8bit or args.load_in_4bit):
            guard_model = guard_model.to(guard_device)

        safety_scores = []
        unsafe_indices = []

        # Use tqdm for progress bar
        for i, prompt in enumerate(tqdm(prompts, desc="Filtering unsafe prompts", unit="prompt")):
            score = get_safety_score_via_guard(prompt, guard_model, guard_tokenizer, guard_device)
            safety_scores.append(score)
            if score == 0.0:  # Only keep truly unsafe prompts
                unsafe_indices.append(i)

        print(f"\nFound {len(unsafe_indices)} unsafe prompts out of {len(prompts)} total")

        if len(unsafe_indices) == 0:
            print("Warning: No unsafe prompts found! Using all prompts instead.")
            unsafe_indices = list(range(len(prompts)))

        # Save to cache if path provided
        if args.cache_unsafe_prompts:
            print(f"\n--- Saving unsafe prompts to cache ---")
            cache_dir = os.path.dirname(args.cache_unsafe_prompts)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            cache_data = {
                'dataset_path': args.data,
                'total_prompts': len(prompts),
                'unsafe_indices': unsafe_indices,
                'num_unsafe': len(unsafe_indices),
                'guard_model': args.guard_model
            }

            with open(args.cache_unsafe_prompts, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            print(f"✓ Saved {len(unsafe_indices)} unsafe prompt indices to: {args.cache_unsafe_prompts}")
    else:
        # Load Guard Model anyway for later use in optimization
        print(f"\n--- Loading guard model for optimization ---")
        print(f"Loading guard model: {args.guard_model}...")

        guard_model_kwargs = {}
        if args.load_in_8bit:
            print("Loading Guard Model in 8-bit mode...")
            guard_model_kwargs['load_in_8bit'] = True
            guard_model_kwargs['device_map'] = guard_device
        elif args.load_in_4bit:
            print("Loading Guard Model in 4-bit mode...")
            guard_model_kwargs['load_in_4bit'] = True
            guard_model_kwargs['device_map'] = guard_device
        else:
            guard_model_kwargs['torch_dtype'] = torch.float16

        guard_tokenizer = AutoTokenizer.from_pretrained(args.guard_model)
        guard_model = AutoModelForCausalLM.from_pretrained(
            args.guard_model,
            **guard_model_kwargs
        )
        if not (args.load_in_8bit or args.load_in_4bit):
            guard_model = guard_model.to(guard_device)
    
    # Filter prompts and costs to only unsafe ones
    unsafe_prompts = [prompts[i] for i in unsafe_indices]
    unsafe_costs = [costs[i] for i in unsafe_indices]
    
    # Calculate weights based on cost (higher cost = higher weight)
    # weight = 6 - cost (same as eval.py)
    import numpy as np
    weights = np.array([max(0.1, 6 - c) for c in unsafe_costs])  # Avoid zero weights
    weights = weights / weights.sum()  # Normalize
    
    # Weighted sampling: select prompts with higher cost more often
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    selected_indices = np.random.choice(
        len(unsafe_prompts),
        size=min(args.num_source_prompts, len(unsafe_prompts)),
        replace=False,
        p=weights
    )
    source_prompts = [unsafe_prompts[i] for i in sorted(selected_indices)]
    source_costs = [unsafe_costs[i] for i in sorted(selected_indices)]
    
    print(f"\nUsing {len(source_prompts)} unsafe source prompts for training (weighted by cost)")
    for i, (p, c) in enumerate(zip(source_prompts, source_costs)):
        weight = max(0.1, 6 - c)
        print(f"  [{i+1}] cost={c}, weight={weight:.2f}: {p[:60]}...")
    
    print(f"Loading chat model: {args.chat_model}...")
    chat_tokenizer = AutoTokenizer.from_pretrained(args.chat_model)

    # Prepare model loading kwargs for Chat Model
    chat_model_kwargs = {}
    if args.load_in_8bit:
        print("Loading Chat Model in 8-bit mode...")
        chat_model_kwargs['load_in_8bit'] = True
        chat_model_kwargs['device_map'] = chat_device
    elif args.load_in_4bit:
        print("Loading Chat Model in 4-bit mode...")
        chat_model_kwargs['load_in_4bit'] = True
        chat_model_kwargs['device_map'] = chat_device
    else:
        chat_model_kwargs['torch_dtype'] = torch.float16

    chat_model = AutoModelForCausalLM.from_pretrained(
        args.chat_model,
        **chat_model_kwargs
    )
    if not (args.load_in_8bit or args.load_in_4bit):
        chat_model = chat_model.to(chat_device)

    # Optimize trigger
    print(f"\n--- Optimizing Trigger ({args.num_trigger_tokens} tokens, {args.num_steps} steps) ---")
    print(f"Hyperparameters: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print(f"Memory optimization: low_memory={args.low_memory}, 8bit={args.load_in_8bit}, 4bit={args.load_in_4bit}")
    print(f"Multi-GPU: Guard on {guard_device}, Chat on {chat_device}")
    print(f"Guard evaluation interval: every {args.guard_eval_interval} step(s)")
    trigger_tokens, token_positions = optimize_trigger(
        source_prompts,
        guard_model,
        guard_tokenizer,
        chat_model,
        chat_tokenizer,
        guard_device=guard_device,
        chat_device=chat_device,
        num_trigger_tokens=args.num_trigger_tokens,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        seed=args.seed,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        low_memory=args.low_memory,
        guard_eval_interval=args.guard_eval_interval
    )
    
    print(f"\nOptimized trigger tokens: {trigger_tokens}")
    print(f"Token positions: {token_positions}")
    print(f"Trigger text: {chat_tokenizer.decode(trigger_tokens, skip_special_tokens=False)}")
    
    # Save trigger with per-token positions
    print(f"\nSaving trigger to {args.output}...")
    save_trigger(trigger_tokens, args.output, token_positions=token_positions)
    
    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()

