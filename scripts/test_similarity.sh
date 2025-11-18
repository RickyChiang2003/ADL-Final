#!/bin/bash

# Test Embedding Similarity Method
# This script tests the new continuous reward signal from embedding similarity

echo "============================================================"
echo "Testing Embedding Similarity Method"
echo "Judge-only training (α=0, β=0, γ=0, δ=3.0)"
echo "============================================================"
echo ""

CUDA_VISIBLE_DEVICES=0,1 python scripts/train_rewriter.py \
  --data theblackcat102/ADL_Final_25W_part1_with_cost \
  --num-train-samples 10 \
  --num-epochs 3 \
  --batch-size 2 \
  --eval-interval 1 \
  --learning-rate 1e-4 \
  --delta 3.0 \
  --alpha 0.0 \
  --beta 0.0 \
  --gamma 0.0 \
  --output models/test_similarity

echo ""
echo "============================================================"
echo "Training completed!"
echo "============================================================"
echo ""
echo "Expected results:"
echo "  1. Judge similarity should increase: 0.2 → 0.5 → 0.8"
echo "  2. Total reward should increase (not fixed)"
echo "  3. Rewritten prompts should become meaningful"
echo "  4. Loss should decrease (not 0.0000)"
echo ""
