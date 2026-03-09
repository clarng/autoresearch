"""
Fixed inference benchmark harness for autoresearch experiments.
Measures throughput, TTFT, and ITL by driving an HTTP inference server
with realistic concurrent traffic. DO NOT MODIFY.

Usage:
    python inference_prepare.py   # runs benchmark against inference.py server
"""

import os
import sys
import json
import time
import math
import signal
import struct
import subprocess
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import Tokenizer, MAX_SEQ_LEN, _document_batches

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "model.pt"
REFERENCE_LOGPROBS_PATH = "reference_logprobs.pt"
REFERENCE_INPUTS_PATH = "reference_inputs.pt"

NUM_PROMPTS = 128           # number of benchmark requests (scored)
PROMPT_LEN = 256            # tokens of context per prompt
GEN_LEN = 256               # tokens to generate per request
WARMUP_REQUESTS = 8         # warmup requests (not scored)
TIME_BUDGET = 120           # max benchmark duration (seconds)
REQUEST_RATE = 10.0         # mean requests/sec (Poisson arrival)
SERVER_PORT = 8391          # localhost port for inference server
SERVER_STARTUP_TIMEOUT = 60 # seconds to wait for server health check
NUM_VERIFY_SEQUENCES = 4    # number of sequences for KL verification

# ---------------------------------------------------------------------------
# Model architecture (duplicated from train.py — DO NOT MODIFY)
# ---------------------------------------------------------------------------

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load the fixed checkpoint into the model. Returns (model, config_dict)."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    config_dict = checkpoint['config']
    config = GPTConfig(**config_dict)
    model = GPT(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to('cuda')
    model.eval()
    return model, config_dict

# ---------------------------------------------------------------------------
# Benchmark prompts
# ---------------------------------------------------------------------------

def get_benchmark_prompts(num_prompts, prompt_len):
    """Extract deterministic fixed-length token sequences from validation data."""
    tokenizer = Tokenizer.from_directory()
    prompts = []
    needed = num_prompts + WARMUP_REQUESTS
    for doc_batch, _ in _document_batches("val"):
        for doc in doc_batch:
            tokens = tokenizer.encode(doc)
            if len(tokens) >= prompt_len + GEN_LEN:
                prompts.append(tokens[:prompt_len])
                if len(prompts) >= needed:
                    return prompts
    if len(prompts) < needed:
        print(f"WARNING: Only found {len(prompts)} prompts (needed {needed})")
    return prompts

# ---------------------------------------------------------------------------
# KL Verification (correctness gate)
# ---------------------------------------------------------------------------

def verify_correctness(server_url):
    """
    Verify that the server produces bitwise identical logits to the reference.
    Loads pre-generated reference logprobs and sends the same inputs to /logits.
    Returns True if all logits match exactly, False otherwise.
    """
    if not os.path.exists(REFERENCE_LOGPROBS_PATH) or not os.path.exists(REFERENCE_INPUTS_PATH):
        print("ERROR: Reference logprobs/inputs not found. Run gen_reference_logprobs.py first.")
        return False

    ref_logprobs = torch.load(REFERENCE_LOGPROBS_PATH, map_location="cpu", weights_only=True)
    ref_inputs = torch.load(REFERENCE_INPUTS_PATH, map_location="cpu", weights_only=True)

    for i in range(len(ref_inputs)):
        input_tokens = ref_inputs[i].tolist()
        expected_logits = ref_logprobs[i]  # [seq_len, vocab_size]

        # Send to server
        req_data = json.dumps({"token_ids": input_tokens}).encode("utf-8")
        req = urllib.request.Request(
            f"{server_url}/logits",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"ERROR: /logits request failed for sequence {i}: {e}")
            return False

        # Parse logits from response
        server_logits = torch.tensor(resp_data["logits"], dtype=torch.float32)

        # Bitwise comparison
        if not torch.equal(expected_logits, server_logits):
            # Find first mismatch
            diff_mask = expected_logits != server_logits
            diff_positions = diff_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            first_pos = diff_positions[0].item() if len(diff_positions) > 0 else -1
            max_diff = (expected_logits - server_logits).abs().max().item()
            print(f"ERROR: Logits mismatch on sequence {i}!")
            print(f"  First mismatch at position {first_pos}")
            print(f"  Max absolute difference: {max_diff}")
            print(f"  Expected shape: {expected_logits.shape}, Got shape: {server_logits.shape}")
            print(f"  KL != 0 — run rejected.")
            return False

    print(f"KL verification passed: all {len(ref_inputs)} sequences match exactly.")
    return True

# ---------------------------------------------------------------------------
# HTTP traffic benchmark
# ---------------------------------------------------------------------------

def _wait_for_health(server_url, timeout):
    """Wait for the server to become healthy."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{server_url}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    return False


def _send_generate_request(server_url, prompt_tokens, max_new_tokens, result_dict, request_id):
    """
    Send a POST /generate request and collect SSE token events with timestamps.
    Stores results in result_dict[request_id].
    """
    req_data = json.dumps({
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/generate",
        data=req_data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    token_times = []  # list of (token_id, timestamp_ns)
    request_send_time = time.perf_counter_ns()
    generated_tokens = []

    try:
        with urllib.request.urlopen(req, timeout=TIME_BUDGET + 30) as resp:
            buffer = b""
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                buffer += chunk
                # Process complete SSE lines
                while b"\n\n" in buffer:
                    event_data, buffer = buffer.split(b"\n\n", 1)
                    ts = time.perf_counter_ns()
                    for line in event_data.split(b"\n"):
                        line = line.strip()
                        if line.startswith(b"data: "):
                            payload = line[6:].decode("utf-8")
                            if payload == "[DONE]":
                                break
                            try:
                                token_data = json.loads(payload)
                                token_id = token_data["token"]
                                generated_tokens.append(token_id)
                                token_times.append(ts)
                            except (json.JSONDecodeError, KeyError):
                                pass
    except Exception as e:
        result_dict[request_id] = {
            "error": str(e),
            "tokens": [],
            "token_times": [],
            "send_time": request_send_time,
        }
        return

    result_dict[request_id] = {
        "tokens": generated_tokens,
        "token_times": token_times,
        "send_time": request_send_time,
    }


def benchmark():
    """
    Run the full inference benchmark:
    1. Launch inference.py as an HTTP server subprocess
    2. Wait for health check
    3. Verify correctness (KL == 0)
    4. Send concurrent requests at Poisson-distributed intervals
    5. Measure throughput, TTFT, ITL
    6. Kill server and return results
    """
    server_url = f"http://localhost:{SERVER_PORT}"

    # Launch server
    print("Launching inference server...")
    server_proc = subprocess.Popen(
        ["uv", "run", "inference.py", "--serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        # Wait for health
        print(f"Waiting for server health check (timeout={SERVER_STARTUP_TIMEOUT}s)...")
        if not _wait_for_health(server_url, SERVER_STARTUP_TIMEOUT):
            print("ERROR: Server failed to start within timeout.")
            # Print server output for debugging
            server_proc.terminate()
            stdout, _ = server_proc.communicate(timeout=5)
            if stdout:
                print("Server output:")
                print(stdout.decode("utf-8", errors="replace")[-2000:])
            return None

        print("Server is healthy.")

        # Verify correctness
        print("Verifying correctness (KL == 0)...")
        if not verify_correctness(server_url):
            print("FAIL: KL verification failed. Run rejected.")
            return None

        # Get prompts
        prompts = get_benchmark_prompts(NUM_PROMPTS, PROMPT_LEN)
        if len(prompts) < WARMUP_REQUESTS + NUM_PROMPTS:
            print(f"ERROR: Not enough prompts ({len(prompts)})")
            return None

        # Warmup
        print(f"Warming up with {WARMUP_REQUESTS} requests...")
        warmup_results = {}
        warmup_threads = []
        for i in range(WARMUP_REQUESTS):
            t = threading.Thread(
                target=_send_generate_request,
                args=(server_url, prompts[i], GEN_LEN, warmup_results, i),
            )
            t.start()
            warmup_threads.append(t)
            time.sleep(0.1)  # small delay between warmup requests
        for t in warmup_threads:
            t.join(timeout=TIME_BUDGET)

        # Benchmark with Poisson arrivals
        print(f"Running benchmark: {NUM_PROMPTS} requests at ~{REQUEST_RATE} req/s...")
        scored_prompts = prompts[WARMUP_REQUESTS:WARMUP_REQUESTS + NUM_PROMPTS]
        results = {}
        threads = []

        # Pre-generate Poisson inter-arrival times
        rng = np.random.default_rng(seed=42)
        inter_arrivals = rng.exponential(1.0 / REQUEST_RATE, size=NUM_PROMPTS)

        wall_start = time.perf_counter_ns()

        for i, prompt in enumerate(scored_prompts):
            t = threading.Thread(
                target=_send_generate_request,
                args=(server_url, prompt, GEN_LEN, results, i),
            )
            t.start()
            threads.append(t)

            # Wait for next arrival (Poisson)
            if i < len(scored_prompts) - 1:
                time.sleep(inter_arrivals[i])

            # Time budget check
            elapsed = (time.perf_counter_ns() - wall_start) / 1e9
            if elapsed > TIME_BUDGET:
                print(f"  Time budget reached after sending {i+1} requests.")
                break

        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=TIME_BUDGET)

        wall_end = time.perf_counter_ns()
        total_wall_time_s = (wall_end - wall_start) / 1e9

        # Compute metrics
        ttft_list = []
        itl_list = []
        total_tokens = 0
        num_completed = 0
        errors = 0

        for req_id, result in sorted(results.items()):
            if "error" in result:
                errors += 1
                continue

            tokens = result["tokens"]
            token_times = result["token_times"]
            send_time = result["send_time"]

            if not tokens or not token_times:
                continue

            num_completed += 1
            total_tokens += len(tokens)

            # TTFT: time from request sent to first token received
            ttft_ns = token_times[0] - send_time
            ttft_list.append(ttft_ns / 1e6)  # convert to ms

            # ITL: time between consecutive tokens
            for j in range(1, len(token_times)):
                itl_ns = token_times[j] - token_times[j - 1]
                itl_list.append(itl_ns / 1e6)  # convert to ms

        # Get peak VRAM from server
        peak_vram_mb = 0
        try:
            req = urllib.request.Request(f"{server_url}/stats")
            with urllib.request.urlopen(req, timeout=5) as resp:
                stats = json.loads(resp.read().decode("utf-8"))
                peak_vram_mb = stats.get("peak_vram_mb", 0)
        except Exception:
            pass

        # Compute final metrics
        throughput = total_tokens / total_wall_time_s if total_wall_time_s > 0 else 0
        mean_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
        mean_itl = sum(itl_list) / len(itl_list) if itl_list else 0
        p99_ttft = sorted(ttft_list)[int(len(ttft_list) * 0.99)] if ttft_list else 0
        p99_itl = sorted(itl_list)[int(len(itl_list) * 0.99)] if itl_list else 0

        return {
            'throughput_tok_per_sec': throughput,
            'mean_ttft_ms': mean_ttft,
            'mean_itl_ms': mean_itl,
            'p99_ttft_ms': p99_ttft,
            'p99_itl_ms': p99_itl,
            'peak_vram_mb': peak_vram_mb,
            'total_tokens': total_tokens,
            'num_completed': num_completed,
            'num_errors': errors,
            'wall_time_sec': total_wall_time_s,
        }

    finally:
        # Kill server
        print("Stopping inference server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = benchmark()
    if results is None:
        print("Benchmark failed.")
        sys.exit(1)

    print("---")
    print(f"throughput_tok_per_sec: {results['throughput_tok_per_sec']:.1f}")
    print(f"mean_ttft_ms:          {results['mean_ttft_ms']:.2f}")
    print(f"mean_itl_ms:           {results['mean_itl_ms']:.2f}")
    print(f"p99_ttft_ms:           {results['p99_ttft_ms']:.2f}")
    print(f"p99_itl_ms:            {results['p99_itl_ms']:.2f}")
    print(f"peak_vram_mb:          {results['peak_vram_mb']:.1f}")
    print(f"total_tokens:          {results['total_tokens']}")
    print(f"num_completed:         {results['num_completed']}")
    print(f"num_errors:            {results['num_errors']}")
    print(f"wall_time_sec:         {results['wall_time_sec']:.1f}")
