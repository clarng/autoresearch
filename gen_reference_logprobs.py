"""
One-time script to generate reference logprobs for KL verification.
Loads model.pt, runs fixed validation sequences through the model,
saves logits to reference_logprobs.pt and inputs to reference_inputs.pt.

Usage: uv run gen_reference_logprobs.py
"""

import torch
from inference_prepare import (
    load_model, get_benchmark_prompts,
    NUM_VERIFY_SEQUENCES, PROMPT_LEN,
    REFERENCE_LOGPROBS_PATH, REFERENCE_INPUTS_PATH,
)

def main():
    print("Loading model...")
    model, config_dict = load_model()

    print("Getting benchmark prompts...")
    prompts = get_benchmark_prompts(NUM_VERIFY_SEQUENCES, PROMPT_LEN)

    print(f"Generating reference logprobs for {NUM_VERIFY_SEQUENCES} sequences...")
    ref_inputs = []
    ref_logprobs = []

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        for i in range(NUM_VERIFY_SEQUENCES):
            tokens = prompts[i][:PROMPT_LEN]
            input_tensor = torch.tensor([tokens], dtype=torch.long, device='cuda')
            logits = model(input_tensor)  # [1, seq_len, vocab_size]
            # Store as float32 on CPU for exact comparison
            ref_inputs.append(torch.tensor(tokens, dtype=torch.long))
            ref_logprobs.append(logits[0].cpu().float())
            print(f"  Sequence {i}: {len(tokens)} tokens -> logits shape {logits[0].shape}")

    torch.save(ref_inputs, REFERENCE_INPUTS_PATH)
    torch.save(ref_logprobs, REFERENCE_LOGPROBS_PATH)
    print(f"Saved {REFERENCE_INPUTS_PATH} and {REFERENCE_LOGPROBS_PATH}")

if __name__ == "__main__":
    main()
