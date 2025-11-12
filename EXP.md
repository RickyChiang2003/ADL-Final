# 1. 訓練 trigger（在前 3 個 toy_data prompts 上）
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_trigger.py \
  --data data/toy_data.jsonl \
  --num-source-prompts 20 \
  --num-trigger-tokens 100 \
  --num-steps 1000 \
  --output models/trigger.json
```

# 2. evaluation
```bash
rm results/algorithm_with_trigger/raw_ADL_Final_25W_part1_with_cost.jsonl
rm results/algorithm_with_trigger/prompts_ADL_Final_25W_part1_with_cost.jsonl
CUDA_VISIBLE_DEVICES=1 python run_inference.py --algorithm algorithm_with_trigger
CUDA_VISIBLE_DEVICES=1 python run_eval.py --algorithm algorithm_with_trigger
```

## trigger
{
  "total_samples": 389,
  "average_safety_score": 0.126,
  "average_relevance_score": 0.2519,
  "final_acc": 0.07840617,
  "average_weighted_safety": 0.1369,
  "average_weighted_relevance": 0.2744,
  "weighted_final_acc": 0.08591378,
  "total_weight": 1647
}

## trigger in different part 
num-source-prompts 20, num-trigger-tokens 3 num-steps 100
{
  "total_samples": 389,
  "average_safety_score": 0.171,
  "average_relevance_score": 0.3008,
  "final_acc": 0.12082262,
  "average_weighted_safety": 0.1788,
  "average_weighted_relevance": 0.3248,
  "weighted_final_acc": 0.12932605,
  "total_weight": 1647
}

## multiple objective
chat_loss = perplexity_loss + 0.5 * refusal_penalty
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_trigger.py \
  --data theblackcat102/ADL_Final_25W_part1_with_cost \
  --num-source-prompts 10 \
  --num-trigger-tokens 20 \
  --num-steps 500 \
  --learning-rate 0.01 \
  --alpha 2.0 \
  --beta 1.0 \
  --gamma 0.01 \
  --output models/trigger_multiobjective.json
```
{
  "total_samples": 389,
  "average_safety_score": 0.3856,
  "average_relevance_score": 0.144,
  "final_acc": 0.10411311,
  "average_weighted_safety": 0.394,
  "average_weighted_relevance": 0.1585,
  "weighted_final_acc": 0.11505768,
  "total_weight": 1647
}


## bonus when start with sure, here is ....
Total Loss = α × guard_loss 
           + β × (2.0 × target_loss + perplexity_loss + 0.5 × refusal_penalty)
           + γ × (||soft_trigger||_2 + diversity_loss)

### 1
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_trigger.py \
  --data theblackcat102/ADL_Final_25W_part1_with_cost \
  --num-source-prompts 10 \
  --num-trigger-tokens 20 \
  --num-steps 100 \
  --learning-rate 0.01 \
  --alpha 2.0 \
  --beta 1.0 \
  --gamma 0.01 \
  --output models/trigger.json
```
{
  "total_samples": 389,
  "average_safety_score": 0.1607,
  "average_relevance_score": 0.1311,
  "final_acc": 0.07197943,
  "average_weighted_safety": 0.17,
  "average_weighted_relevance": 0.1481,
  "weighted_final_acc": 0.08075288,
  "total_weight": 1647
}

### 2

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_trigger.py \
  --data theblackcat102/ADL_Final_25W_part1_with_cost \
  --num-source-prompts 1 \
  --num-trigger-tokens 50 \
  --num-steps 500 \
  --learning-rate 0.02 \
  --alpha 3.0 \
  --beta 1.0 \
  --gamma 0.005 \
  --load-in-8bit \
  --output models/trigger.json \
  --guard-eval-interval 1 \
  --cache-unsafe-prompts cache/unsafe_prompts.json
```
{
  "total_samples": 389,
  "average_safety_score": 0.5398,
  "average_relevance_score": 0.1388,
  "final_acc": 0.11825193,
  "average_weighted_safety": 0.5468,
  "average_weighted_relevance": 0.1445,
  "weighted_final_acc": 0.12234366,
  "total_weight": 1647
}

### 3

```bash
python scripts/train_trigger.py \
  --data theblackcat102/ADL_Final_25W_part1_with_cost \
  --num-source-prompts 1 \
  --num-trigger-tokens 50 \
  --num-steps 500 \
  --learning-rate 1e-3 \
  --alpha 0.3 \
  --beta 5.0 \
  --gamma 0.005 \
  --load-in-8bit \
  --output models/trigger.json \
  --guard-eval-interval 1 \
  --cache-unsafe-prompts cache/unsafe_prompts.json
```

### 4

```bash
python scripts/train_trigger.py \
  --data theblackcat102/ADL_Final_25W_part1_with_cost \
  --num-source-prompts 3 \
  --num-trigger-tokens 50 \
  --num-steps 500 \
  --learning-rate 1e-3 \
  --alpha 1.0 \
  --beta 3.0 \
  --gamma 0.1 \
  --load-in-8bit \
  --output models/trigger.json \
  --guard-eval-interval 1 \
  --cache-unsafe-prompts cache/unsafe_prompts.json
```
{
  "total_samples": 389,
  "average_safety_score": 0.3869,
  "average_relevance_score": 0.2057,
  "final_acc": 0.15681234,
  "average_weighted_safety": 0.3937,
  "average_weighted_relevance": 0.2198,
  "weighted_final_acc": 0.16484517,
  "total_weight": 1647
}