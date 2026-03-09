"""
Inference server for autoresearch. Single-GPU, single-file.
This is the file you modify to optimize inference throughput, TTFT, and ITL.

Usage:
    uv run inference.py --serve    # start HTTP server for benchmarking

HTTP API Contract:
    GET  /health    -> 200 OK when server is ready
    POST /generate  -> SSE stream of generated tokens
        Request:  {"prompt_tokens": [int, ...], "max_new_tokens": int}
        Response: SSE stream: data: {"token": int}\\n\\n ... data: [DONE]\\n\\n
    POST /logits   -> full logit tensor for KL verification
        Request:  {"token_ids": [int, ...]}
        Response: {"logits": [[float, ...], ...]}
    GET  /stats    -> {"peak_vram_mb": float}

Read train.py to understand the model architecture, then implement:
    1. Model loading from model.pt checkpoint
    2. Forward pass that produces identical logits to train.py
    3. Autoregressive text generation
    4. HTTP server with the endpoints above

The benchmark harness (inference_prepare.py) will:
    1. Launch this script as a subprocess
    2. Verify KL == 0 by comparing /logits output against reference_logprobs.pt
    3. Send concurrent /generate requests at realistic arrival rates
    4. Measure throughput (tok/sec), TTFT (ms), and ITL (ms)
"""

import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

SERVER_PORT = 8391

# ---------------------------------------------------------------------------
# TODO: Implement model loading and generation
#
# Study train.py to understand:
#   - GPTConfig dataclass and model hyperparameters
#   - GPT model architecture (embeddings, blocks, attention, MLP, etc.)
#   - Forward pass details (RMS norm, RoPE, value embeddings, residual
#     lambdas, x0 lambdas, softcap logits)
#   - Flash Attention 3 usage for causal self-attention
#   - Window sizes (sliding window pattern)
#
# Then implement:
#   1. load_model() - load model.pt checkpoint into model on CUDA
#   2. get_logits(token_ids) - run forward pass, return logits tensor
#   3. generate(prompt_tokens, max_new_tokens) - autoregressive generation
# ---------------------------------------------------------------------------


class InferenceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for inference server."""

    def log_message(self, format, *args):
        # Suppress default logging to keep output clean
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        elif self.path == "/stats":
            import torch
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"peak_vram_mb": peak_vram_mb}).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)

        if self.path == "/logits":
            # TODO: Implement - run forward pass on token_ids, return logits
            # token_ids = data["token_ids"]
            # logits = get_logits(token_ids)  # should return list of lists
            # self.send_response(200)
            # self.send_header("Content-Type", "application/json")
            # self.end_headers()
            # self.wfile.write(json.dumps({"logits": logits}).encode())
            self.send_response(501)
            self.end_headers()
            self.wfile.write(b'{"error": "not implemented"}')

        elif self.path == "/generate":
            # TODO: Implement - generate tokens and stream via SSE
            # prompt_tokens = data["prompt_tokens"]
            # max_new_tokens = data["max_new_tokens"]
            # self.send_response(200)
            # self.send_header("Content-Type", "text/event-stream")
            # self.send_header("Cache-Control", "no-cache")
            # self.end_headers()
            # for token in generate(prompt_tokens, max_new_tokens):
            #     self.wfile.write(f"data: {json.dumps({'token': token})}\n\n".encode())
            #     self.wfile.flush()
            # self.wfile.write(b"data: [DONE]\n\n")
            # self.wfile.flush()
            self.send_response(501)
            self.end_headers()
            self.wfile.write(b'{"error": "not implemented"}')

        else:
            self.send_response(404)
            self.end_headers()


def main():
    if "--serve" not in sys.argv:
        print("Usage: uv run inference.py --serve")
        print("This starts the HTTP inference server for benchmarking.")
        sys.exit(1)

    # TODO: Load model here before starting server
    # model = load_model()

    print(f"Starting inference server on port {SERVER_PORT}...")
    server = HTTPServer(("localhost", SERVER_PORT), InferenceHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
