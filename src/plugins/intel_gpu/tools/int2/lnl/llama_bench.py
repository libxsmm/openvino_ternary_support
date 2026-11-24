#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""llama-server counterpart of genai_bench.py: same chat-templated prompts, same protocol.

    python llama_bench.py --server <llama.cpp>/build-sycl/bin/llama-server --name plain \
        --args "-m <gguf> -ngl 99 -fa on" --save ref.json --log srv.log

Per prompt: one warm-up, then --runs timed requests (median), raw prompt to /completion, --tokens greedy
tokens with EOS ignored, no prompt cache. "prompt" is the server's prompt_ms (prefill), decode is its
predicted_per_second. Speculative configs report tokens per step and are checked against --ref.
"""
import argparse
import hashlib
import json
import os
import shlex
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

# llama-server is local: never route through http_proxy
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))

# genai_bench.py's PROMPTS after the model's chat template (OpenVINO GenAI's apply_chat_template)
CHAT = {
    "photo": "<|im_start|>user\nTell me about photosynthesis in 200 words<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "math": "<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips to her friends in May. How many clips did Natalia sell altogether in April and May? Think step by step.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "code": "<|im_start|>user\nRewrite the following Python function so that it uses type hints and a docstring, keep everything else the same.\n\n```python\ndef merge_sorted(a, b):\n    i = 0\n    j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i])\n            i += 1\n        else:\n            out.append(b[j])\n            j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out\n```<|im_end|>\n<|im_start|>assistant\n<think>\n"
}


def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--args", required=True, help="llama-server model/backend flags")
    ap.add_argument("--prompts", default="photo,math,code")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--pace", type=float, default=0.0, help="idle seconds before each timed request")
    ap.add_argument("--ref", default="", help="json of plain-run texts to compare against")
    ap.add_argument("--save", default="", help="write this run's texts here")
    ap.add_argument("--log", required=True)
    opt = ap.parse_args()

    port = free_port()
    cmd = [opt.server, "--host", "127.0.0.1", "--port", str(port), "-np", "1", "-c", "4096", "--no-warmup"] + shlex.split(opt.args)
    log = open(opt.log, "w")
    log.write(" ".join(cmd) + "\n")
    log.flush()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(600):
            if srv.poll() is not None:
                print(f"{opt.name}: server exited rc={srv.returncode}, see {opt.log}")
                return 1
            try:
                if json.loads(urllib.request.urlopen(url + "/health", timeout=2).read()).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        ref = json.load(open(opt.ref)) if opt.ref and os.path.exists(opt.ref) else {}
        texts = {}
        for name in opt.prompts.split(","):
            # same chat-templated prompt as genai_sd_bench.py (the GGUF's embedded template)
            chat = CHAT[name]
            log.write(f"PROMPT[{name}] {chat!r}\n"); log.flush()
            payload = {"prompt": chat, "n_predict": opt.tokens, "temperature": 0.0, "top_k": 1,
                       "ignore_eos": True, "cache_prompt": False, "stream": False}
            post(url + "/completion", payload)
            rates, steps, prompt_ms, text = [], [], [], None
            for _ in range(opt.runs):
                if opt.pace:
                    time.sleep(opt.pace)
                out = post(url + "/completion", payload)
                t = out["timings"]
                assert out["tokens_predicted"] == opt.tokens, out["tokens_predicted"]
                rates.append(t["predicted_per_second"])
                prompt_ms.append(t["prompt_ms"])
                if t.get("draft_n"):
                    n_steps = t["predicted_n"] - t["draft_n_accepted"]
                    steps.append(t["predicted_n"] / max(n_steps, 1))
                text = out["content"]
            texts[name] = text
            md5 = hashlib.md5(text.encode()).hexdigest()[:12]
            same = "ref" if not ref else ("same" if ref.get(name) == text else "DIFF")
            tps = f"{statistics.median(steps):4.2f}" if steps else "1.00"
            print(f"{opt.name:28s} {name:5s} prompt {statistics.median(prompt_ms):7.1f} ms  decode "
                  f"{statistics.median(rates):7.1f} tok/s  tok/step {tps}  md5 {md5} {same}", flush=True)
        if opt.save:
            json.dump(texts, open(opt.save, "w"))
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
