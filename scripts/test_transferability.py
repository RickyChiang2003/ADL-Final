#!/usr/bin/env python3
"""
Test Transferability of Learned Trigger.

This script:
1. Trains a universal soft-prefix trigger on a small set of source prompts from toy_data.jsonl.
2. Tests the learned trigger's transferability on the full toy_data.jsonl.
3. Runs inference with algorithm_with_trigger and evaluates safety/relevance scores.
4. Generates a detailed report.

Usage:
    python scripts/test_transferability.py \
      --data data/toy_data.jsonl \
      --num-source-prompts 3 \
      --num-trigger-tokens 5 \
      --num-steps 50 \
      --output-dir results/transferability_test
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attacks_ import (
    load_toy_data,
    optimize_trigger,
    save_trigger,
    apply_trigger_to_prompt,
    evaluate_transferability,
    get_safety_score_via_guard
)


def create_toy_dataset_with_trigger(
    original_prompts: List[str],
    trigger_tokens: List[int],
    tokenizer,
    output_jsonl: str,
    mode: str = "prepend"
):
    """Create a JSONL file with trigger-applied prompts for inference."""
    os.makedirs(Path(output_jsonl).parent, exist_ok=True)
    
    print(f"Applying trigger to {len(original_prompts)} prompts...")
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(original_prompts):
            try:
                rewritten = apply_trigger_to_prompt(
                    prompt,
                    trigger_tokens,
                    tokenizer=tokenizer,
                    mode=mode
                )
                f.write(json.dumps({"prompt": rewritten}, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"  Warning: Error on prompt {i}: {e}")
                f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + '\n')
            
            if (i + 1) % 10 == 0 or i == len(original_prompts) - 1:
                print(f"  Applied trigger to {i + 1}/{len(original_prompts)} prompts")


def main():
    parser = argparse.ArgumentParser(description="Test trigger transferability on toy_data.")
    parser.add_argument('--data', type=str, default='data/toy_data.jsonl', help='Path to toy data')
    parser.add_argument('--num-source-prompts', type=int, default=3, help='Prompts to train on')
    parser.add_argument('--num-trigger-tokens', type=int, default=5, help='Trigger token length')
    parser.add_argument('--num-steps', type=int, default=50, help='Optimization steps')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='results/transferability_test')
    parser.add_argument('--guard-model', type=str, default='Qwen/Qwen3Guard-Gen-0.6B')
    parser.add_argument('--chat-model', type=str, default='unsloth/Llama-3.2-3B-Instruct')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Trigger Transferability Test")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Toy data: {args.data}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)
    
    # Load all toy data prompts
    all_prompts = load_toy_data(args.data)
    source_prompts = all_prompts[:args.num_source_prompts]
    test_prompts = all_prompts
    
    print(f"\nLoaded {len(all_prompts)} prompts from {args.data}")
    print(f"  - Source prompts for training: {len(source_prompts)}")
    print(f"  - Test prompts for transferability: {len(test_prompts)}")
    
    # Load models
    print("\nLoading models...")
    guard_tokenizer = AutoTokenizer.from_pretrained(args.guard_model)
    guard_model = AutoModelForCausalLM.from_pretrained(
        args.guard_model,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None
    )
    if device == 'cpu':
        guard_model = guard_model.to(device)
    
    chat_tokenizer = AutoTokenizer.from_pretrained(args.chat_model)
    chat_model = AutoModelForCausalLM.from_pretrained(
        args.chat_model,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None
    )
    if device == 'cpu':
        chat_model = chat_model.to(device)
    
    # Train trigger
    print("\n" + "=" * 80)
    print("Training trigger on source prompts...")
    print("=" * 80)
    
    trigger_tokens = optimize_trigger(
        source_prompts,
        guard_model,
        guard_tokenizer,
        chat_model,
        chat_tokenizer,
        device=device,
        num_trigger_tokens=args.num_trigger_tokens,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        seed=args.seed
    )
    
    # Save trigger to models/ for inference use
    trigger_path = "models/trigger.json"
    save_trigger(trigger_tokens, trigger_path)
    print(f"Trigger saved to {trigger_path}")
    
    # Also save to output dir for record
    output_trigger_path = output_dir / "trigger.json"
    save_trigger(trigger_tokens, str(output_trigger_path))
    
    # Create JSONL with trigger-applied prompts
    trigger_prompts_path = output_dir / "trigger_prompts_toy_data.jsonl"
    create_toy_dataset_with_trigger(
        test_prompts,
        trigger_tokens,
        chat_tokenizer,
        str(trigger_prompts_path),
        mode="prepend"
    )
    
    # Test transferability directly
    print("\n" + "=" * 80)
    print("Evaluating transferability...")
    print("=" * 80)
    
    transfer_results = evaluate_transferability(
        trigger_tokens,
        test_prompts,
        guard_model,
        guard_tokenizer,
        chat_tokenizer,
        device=device,
        mode="prepend"
    )
    
    # Save transferability results
    transfer_report_path = output_dir / "transferability_report.json"
    with open(transfer_report_path, 'w') as f:
        json.dump(transfer_results, f, indent=2)
    print(f"Transferability report saved to {transfer_report_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Transferability Summary")
    print("=" * 80)
    print(f"  - Source prompts: {len(source_prompts)}")
    print(f"  - Test prompts: {len(test_prompts)}")
    print(f"  - Trigger tokens: {trigger_tokens}")
    print(f"  - Avg original safety score: {transfer_results['avg_original_safety']:.4f}")
    print(f"  - Avg triggered safety score: {transfer_results['avg_triggered_safety']:.4f}")
    print(f"  - Safety delta: {transfer_results['safety_delta']:.4f}")
    print(f"  - Transfer success rate: {transfer_results['transfer_success_rate']:.2%}")
    
    print("\n" + "=" * 80)
    print(f"Test complete. Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
