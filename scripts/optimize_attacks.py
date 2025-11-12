#!/usr/bin/env python3
"""
Script to apply optimizations to attacks_.py:
1. Add tqdm for progress bars
2. Implement guard_eval_cache
3. Use half prepend/half append strategy
4. Reduce max_new_tokens
5. Optimize GPU switching
"""

import re

def optimize_attacks_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Add tqdm import
    content = content.replace(
        'import re',
        'import re\nfrom tqdm import tqdm'
    )

    # 2. Reduce max_new_tokens from 32 to 16
    content = content.replace(
        'max_new_tokens=32,  # Reduced from 64 to 32 to save time',
        'max_new_tokens=16,  # Reduced from 32 to 16 for faster evaluation'
    )

    # 3. Add position strategy before main loop
    position_strategy_code = '''
    # Determine position strategy: half prepend, half append
    # This avoids testing both positions for every token
    num_prepend = num_trigger_tokens // 2
    num_append = num_trigger_tokens - num_prepend
    print(f"Position strategy: {num_prepend} tokens prepend, {num_append} tokens append")

    # Progress bar for steps
    pbar = tqdm(range(num_steps), desc="Optimizing trigger", unit="step")
    for step in pbar:'''

    content = content.replace(
        '    for step in range(num_steps):',
        position_strategy_code
    )

    # 4. Replace position testing logic - this is the most complex change
    # Find and replace the multiple positions testing
    old_positions = '''            # Try multiple application positions
            positions = [
                ("prepend", torch.cat([soft_trigger.unsqueeze(0), prompt_embeddings.unsqueeze(0)], dim=1)),
                ("append", torch.cat([prompt_embeddings.unsqueeze(0), soft_trigger.unsqueeze(0)], dim=1)),
            ]

            # First, compute Chat Model loss for all positions (fast)
            chat_losses = []
            position_names = []

            for pos_name, combined_embeddings in positions:
                position_names.append(pos_name)'''

    new_positions = '''            # Use fixed position strategy: first half prepend, second half append
            # Split trigger into two parts
            soft_trigger_prepend = soft_trigger[:num_prepend]
            soft_trigger_append = soft_trigger[num_prepend:]

            # Combine: [prepend_tokens] + [prompt] + [append_tokens]
            combined_embeddings = torch.cat([
                soft_trigger_prepend.unsqueeze(0),
                prompt_embeddings.unsqueeze(0),
                soft_trigger_append.unsqueeze(0)
            ], dim=1)

            # Only one position to test now (much faster!)
            position_name = "prepend+append"'''

    content = content.replace(old_positions, new_positions)

    # 5. Remove the loop closure and position selection logic
    old_loop_logic = '''                # Combined chat loss with target-guided generation
                chat_loss = 2.0 * target_loss + perplexity_loss + 0.5 * refusal_penalty
                chat_losses.append(chat_loss)

            # === Part 3: Guard Model Loss (only evaluate once per prompt) ==='''

    new_loop_logic = '''            # Combined chat loss with target-guided generation
            chat_loss = 2.0 * target_loss + perplexity_loss + 0.5 * refusal_penalty

            # === Part 3: Guard Model Loss (only evaluate once per prompt) ==='''

    content = content.replace(old_loop_logic, new_loop_logic)

    # 6. Implement guard_eval_cache
    old_guard_eval = '''                    # Get Guard Model's safety score
                    guard_input = [{"role": "user", "content": triggered_prompt_text}]
                    guard_input_ids = guard_tokenizer.apply_chat_template(guard_input, return_tensors="pt").to(guard_device)

                    guard_output = guard_model.generate(
                        input_ids=guard_input_ids,
                        max_new_tokens=16,  # Reduced from 32 to 16 for faster evaluation
                        pad_token_id=0,
                        do_sample=False
                    )

                    prompt_len = guard_input_ids.shape[-1]
                    guard_decision = guard_tokenizer.decode(guard_output[0][prompt_len:], skip_special_tokens=True).strip().lower()

                    # Parse safety score (0.0=unsafe, 0.5=controversial, 1.0=safe)
                    pattern = re.compile(r"safety:\\s*(safe|unsafe|controversial)", re.IGNORECASE)
                    match = pattern.search(guard_decision)
                    if match:
                        decision = match.group(1).lower()
                    else:
                        decision = guard_decision

                    if decision.startswith('safe'):
                        safety_score = 1.0
                    elif decision.startswith('unsafe'):
                        safety_score = 0.0
                    elif 'controversial' in decision:
                        safety_score = 0.5
                    else:
                        safety_score = 0.5'''

    new_guard_eval = '''                    # Check cache first
                    if triggered_prompt_text in guard_eval_cache:
                        safety_score = guard_eval_cache[triggered_prompt_text]
                    else:
                        # Get Guard Model's safety score
                        guard_input = [{"role": "user", "content": triggered_prompt_text}]
                        guard_input_ids = guard_tokenizer.apply_chat_template(guard_input, return_tensors="pt").to(guard_device)

                        guard_output = guard_model.generate(
                            input_ids=guard_input_ids,
                            max_new_tokens=16,  # Reduced from 32 to 16 for faster evaluation
                            pad_token_id=0,
                            do_sample=False
                        )

                        prompt_len = guard_input_ids.shape[-1]
                        guard_decision = guard_tokenizer.decode(guard_output[0][prompt_len:], skip_special_tokens=True).strip().lower()

                        # Parse safety score (0.0=unsafe, 0.5=controversial, 1.0=safe)
                        pattern = re.compile(r"safety:\\s*(safe|unsafe|controversial)", re.IGNORECASE)
                        match = pattern.search(guard_decision)
                        if match:
                            decision = match.group(1).lower()
                        else:
                            decision = guard_decision

                        if decision.startswith('safe'):
                            safety_score = 1.0
                        elif decision.startswith('unsafe'):
                            safety_score = 0.0
                        elif 'controversial' in decision:
                            safety_score = 0.5
                        else:
                            safety_score = 0.5

                        # Cache the result
                        guard_eval_cache[triggered_prompt_text] = safety_score'''

    content = content.replace(old_guard_eval, new_guard_eval)

    # 7. Update the position loss logic
    old_position_loss = '''            # === Part 4: Combine losses for each position ===
            position_losses = []
            for idx, chat_loss in enumerate(chat_losses):
                position_loss = alpha * guard_loss_tensor + beta * chat_loss
                position_losses.append(position_loss)

            # Take minimum loss across positions (best position wins)
            if position_losses:
                best_idx = min(range(len(position_losses)), key=lambda i: position_losses[i].item())
                best_loss = position_losses[best_idx]
                best_pos_name = position_names[best_idx]

                token_position_losses[0][best_pos_name] += best_loss.item()
                token_position_count[0][best_pos_name] += 1
            else:
                best_loss = torch.tensor(0.0, device=chat_device, dtype=compute_dtype)'''

    new_position_loss = '''            # === Part 4: Combine losses ===
            # Simple combination since we only have one position now
            best_loss = alpha * guard_loss_tensor + beta * chat_loss'''

    content = content.replace(old_position_loss, new_position_loss)

    # 8. Update progress bar
    old_logging = '''        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{num_steps}, Total: {avg_total_loss:.6f} | "
                  f"Guard: {avg_guard_loss:.6f} | Chat: {avg_chat_loss:.6f}")'''

    new_logging = '''        # Update progress bar
        pbar.set_postfix({
            'Total': f'{avg_total_loss:.4f}',
            'Guard': f'{avg_guard_loss:.4f}',
            'Chat': f'{avg_chat_loss:.4f}'
        })'''

    content = content.replace(old_logging, new_logging)

    # 9. Close progress bar at the end
    content = content.replace(
        '    print("Trigger optimization complete.")',
        '    pbar.close()\n    print("\\nTrigger optimization complete.")'
    )

    # Write optimized content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Optimized file written to {output_path}")

if __name__ == "__main__":
    import sys
    input_file = "src/attacks_.py"
    output_file = "src/attacks_.py"

    print(f"Optimizing {input_file}...")
    optimize_attacks_file(input_file, output_file)
    print("Done!")
