# autoresearch — inference optimization

This is an experiment to have the LLM optimize inference code autonomously.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9-infer`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `inference_prepare.py` — fixed benchmark harness. Do not modify. Contains the HTTP API contract, KL verification, and traffic simulation.
   - `inference.py` — the file you modify. Currently a skeleton with TODO stubs.
   - `train.py` — reference model architecture. Read-only. Study this carefully to understand the model.
   - `prepare.py` — data loading and tokenizer. Read-only.
4. **Verify artifacts**: Check that `model.pt`, `reference_logprobs.pt`, and `reference_inputs.pt` exist. If not, tell the human.
5. **Initialize inference_results.tsv**: Create with header row. No baseline yet (you need to implement inference first).
6. **Confirm and go**.

## The Task

`inference.py` is currently an empty skeleton. Your job:

1. **First**: Read `train.py` carefully. Understand the full model architecture: GPTConfig, GPT class, attention (Flash Attention 3, sliding windows, RoPE, QK-norm), MLP (ReLU squared), value embeddings with gating, residual lambdas, x0 lambdas, softcap logits, etc.

2. **Bootstrap**: Implement the model, forward pass, and generation logic in `inference.py`. You must:
   - Load the checkpoint from `model.pt`
   - Implement the HTTP server endpoints (`/health`, `/generate`, `/logits`, `/stats`)
   - The `/logits` endpoint must produce **bitwise identical** logits to the reference model
   - The `/generate` endpoint must stream generated tokens via SSE

3. **Verify**: Run `uv run inference_prepare.py > run.log 2>&1`. This will:
   - Launch your server
   - Check KL == 0 (exact logit match against `reference_logprobs.pt`)
   - If KL != 0: fix your implementation until it matches exactly
   - If KL == 0: run the speed benchmark and report metrics

4. **Optimize**: Once your implementation passes KL verification, optimize for speed. The benchmark sends concurrent requests — maximize aggregate throughput while minimizing per-request latency.

## Experimentation

Each experiment runs on a single GPU. Run: `uv run inference_prepare.py > run.log 2>&1`

**What you CAN do:**
- Modify `inference.py` — this is the only file you edit. You may redefine model classes, add new functions, restructure the code however you want.

**What you CANNOT do:**
- Modify `inference_prepare.py`, `train.py`, or `prepare.py`.
- Install new packages or add dependencies.
- Change the model weights. You must load from the fixed `model.pt`.
- Change the HTTP API contract (endpoints, request/response format).

**The goal: maximize throughput (tokens/sec) while minimizing TTFT and ITL.**

**Keep/discard rule**: A change is **kept** if it improves at least one metric (higher throughput, lower TTFT, or lower ITL) without regressing any metric beyond noise (±2% tolerance). If all three metrics are within noise, discard.

**KL must be exactly zero on every run.** If the KL verification fails, the run is auto-rejected. Fix the bug and try again.

**Metrics (from output):**
- `throughput_tok_per_sec` — higher is better (primary)
- `mean_ttft_ms` — lower is better
- `mean_itl_ms` — lower is better
- `peak_vram_mb` — soft constraint (don't blow up)

## Output format

```
---
throughput_tok_per_sec: 150.3
mean_ttft_ms:          12.50
mean_itl_ms:           6.30
p99_ttft_ms:           25.00
p99_itl_ms:            12.50
peak_vram_mb:          2048.5
total_tokens:          15360
num_completed:         60
num_errors:            0
wall_time_sec:         102.1
```

Extract key metrics: `grep "^throughput_tok_per_sec:\|^mean_ttft_ms:\|^mean_itl_ms:" run.log`

## Logging results

Log to `inference_results.tsv` (tab-separated, NOT comma-separated):

```
commit	throughput	ttft_ms	itl_ms	memory_gb	status	description
```

- commit: short git hash (7 chars)
- throughput: tok/sec
- ttft_ms: mean TTFT in ms
- itl_ms: mean ITL in ms
- memory_gb: peak VRAM / 1024
- status: keep/discard/crash
- description: what was tried

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Edit `inference.py` with an optimization idea
3. git commit
4. Run: `uv run inference_prepare.py > run.log 2>&1`
5. Read results: `grep "^throughput_tok_per_sec:\|^mean_ttft_ms:\|^mean_itl_ms:\|^peak_vram_mb:" run.log`
6. If grep is empty → crash or KL failure. Run `tail -n 50 run.log` for details.
7. Record results in inference_results.tsv
8. If improved (at least one metric better, none worse beyond ±2%): keep the commit
9. If not improved: `git reset --hard HEAD~1`
10. NEVER STOP.

**Timeout**: Each benchmark run takes ~2 minutes. If it exceeds 5 minutes, kill it and treat as failure.

**Crashes**: If a run crashes, use judgment: fix trivial bugs and re-run, or skip and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — analyze where time is being spent, profile bottlenecks, read about inference optimization, try combinations. The loop runs until you are manually stopped.
