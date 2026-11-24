#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Bonsai 2 TTFT / decode bench on OpenVINO GenAI (ContinuousBatchingPipeline), plain and MTP.

    python genai_bench.py <genai model dir> --modes plain,mtp --k 3
    python genai_bench.py <genai model dir> --modes plain --long 512,2048 --n 2

TTFT = time of a 1-token generation. Decode tok/s = (n - 1) / (T(n) - TTFT), ignore_eos, median of --runs.
The chat prompts are the ones used for the llama.cpp comparison (README.md); MTP outputs are checked
token for token against plain greedy decoding.
"""
import argparse
import hashlib
import os
import statistics
import time

import openvino_genai as og

PROMPTS = {
    "photo": "Tell me about photosynthesis in 200 words",
    "math": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
            "to her friends in May. How many clips did Natalia sell altogether in April and May? "
            "Think step by step.",
    "code": "Rewrite the following Python function so that it uses type hints and a docstring, "
            "keep everything else the same.\n\n```python\n"
            "def merge_sorted(a, b):\n    i = 0\n    j = 0\n    out = []\n"
            "    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n"
            "            out.append(a[i])\n            i += 1\n        else:\n"
            "            out.append(b[j])\n            j += 1\n    out.extend(a[i:])\n"
            "    out.extend(b[j:])\n    return out\n```",
}


def make_pipe(model, device, mode, chunk):
    sched = og.SchedulerConfig()
    sched.cache_size = 4
    sched.max_num_seqs = 1
    sched.enable_prefix_caching = False
    sched.max_num_batched_tokens = chunk  # prefill chunk; 512 = llama-bench's default ubatch
    props = {"CACHE_DIR": os.environ.get("OV_CACHE_DIR", "ov_cache")}
    if mode == "mtp":
        ds = og.SchedulerConfig()
        ds.enable_prefix_caching = False
        ds.num_kv_blocks = 512
        props["draft_model"] = og.draft_model(model, device, scheduler_config=ds)
    return og.ContinuousBatchingPipeline(model, sched, device, props)


def run(pipe, prompt, n, k):
    cfg = og.GenerationConfig(max_new_tokens=n, do_sample=False, ignore_eos=True, apply_chat_template=False)
    if k:
        cfg.num_assistant_tokens = k
    t = time.perf_counter()
    h = pipe.add_request(0, prompt, cfg)
    steps = 0
    while pipe.has_non_finished_requests():
        pipe.step()
        steps += 1
    return time.perf_counter() - t, list(h.read_all()[0].generated_ids), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--modes", default="plain,mtp")
    ap.add_argument("--k", default="3", help="MTP draft tokens, comma separated")
    ap.add_argument("--n", type=int, default=256, help="generated tokens per run")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--prompts", default="photo,math,code")
    ap.add_argument("--long", default="", help="extra prompts of exactly N tokens (code text repeated), e.g. 512,2048")
    ap.add_argument("--chunk", type=int, default=512, help="max_num_batched_tokens (prefill chunk)")
    ap.add_argument("--pace", type=float, default=10.0, help="seconds idle before each timed run (LNL: avoids throttling)")
    a = ap.parse_args()

    tok = og.Tokenizer(a.model)
    prompts = {}
    for p in filter(None, a.prompts.split(",")):
        hist = og.ChatHistory()
        hist.append({"role": "user", "content": PROMPTS[p]})
        prompts[p] = tok.apply_chat_template(hist, add_generation_prompt=True)
    filler = tok.encode(PROMPTS["code"] * 64, add_special_tokens=False).input_ids.data[0]
    for n in filter(None, a.long.split(",")):
        assert len(filler) >= int(n), "--long too large"
        text = tok.decode(filler[:int(n)].tolist())
        prompts[f"tok{n}"] = text
        print(f"tok{n}: {tok.encode(text, add_special_tokens=False).input_ids.get_shape()[1]} prompt tokens", flush=True)

    ref = {}
    for mode in a.modes.split(","):
        ks = [0] if mode == "plain" else [int(x) for x in a.k.split(",")]
        t0 = time.perf_counter()
        pipe = make_pipe(a.model, a.device, mode, a.chunk)
        print(f"[{mode}] pipeline ready in {time.perf_counter() - t0:.1f}s", flush=True)
        for k in ks:
            for p, prompt in prompts.items():
                run(pipe, prompt, 8, k)  # warm-up: builds the kernels for this prompt length
                ttfts = []
                for _ in range(a.runs):
                    time.sleep(a.pace)
                    ttfts.append(run(pipe, prompt, 1, k)[0])
                t1 = statistics.median(ttfts)
                line = f"{mode:5s} k={k} {p:7s} ttft {t1 * 1e3:7.1f} ms"
                if a.n > 1:
                    times = []
                    for _ in range(a.runs):
                        time.sleep(a.pace)
                        dt, ids, steps = run(pipe, prompt, a.n, k)
                        times.append(dt)
                    md5 = hashlib.md5(",".join(map(str, ids)).encode()).hexdigest()[:12]
                    if mode == "plain":
                        ref[p], same = ids, "ref"
                    elif p in ref:
                        same = "same" if ids == ref[p] else \
                            f"DIFF@{next((i for i, (x, y) in enumerate(zip(ids, ref[p])) if x != y), len(ids))}"
                    else:
                        same = ""
                    line += (f"  decode {(a.n - 1) / (statistics.median(times) - t1):6.1f} tok/s"
                             f"  tok/step {a.n / max(steps - 1, 1):.2f}  md5 {md5} {same}")
                print(line, flush=True)
        del pipe


if __name__ == "__main__":
    main()
