#!/usr/bin/env python3
"""Build a 1-bit Bonsai IR from its Q1_0 GGUF, using the ternary Bonsai IR of the
same architecture as the template.

prism-ml/Bonsai-8B-gguf (qwen3) and prism-ml/Bonsai-27B-gguf (qwen35) store every
linear layer, the token embedding and lm_head as Q1_0: per 128 weights one fp16
scale d and 128 sign bits (LSB first, 1 -> +d, 0 -> -d). The ternary IRs
(Ternary-Bonsai-8B / 27B, u2) hold the same graphs, so this tool keeps the graph
and swaps every weight:

  * each u2 decompression subgraph becomes u1 bits -> Convert -> Subtract(1/2)
    -> Multiply(2d) -> Reshape, which the GPU plugin runs on the TernOCL int1
    kernels; output widths that are not a multiple of 16 (the 8B vocabulary,
    151669) get zero rows and the MatMul output is sliced back;
  * dense tensors (norms, GDN A_log/dt_bias/conv1d) are read from the GGUF, with
    llama.cpp's V-head reorder undone for qwen35;
  * the token embedding stays 1-bit: a u8 bit-plane table plus fp16 scales,
    gathered and decoded on the device (inside the 8B graph; as the separate
    text-embeddings model for 27B).

    python bonsai1_gguf_to_ir.py --gguf Bonsai-8B-Q1_0.gguf \
        --template-dir <bonsai8b-u2> --out-dir <bonsai8b-u1>
    python bonsai1_gguf_to_ir.py --gguf Bonsai-27B-Q1_0.gguf \
        --template-dir <bonsai27b-u2> --out-dir <bonsai27b-u1>

Reading Q1_0 needs a gguf-py that knows it (type id 41), e.g. the PrismML
llama.cpp fork's.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import openvino as ov
from openvino import opset13 as ops

from bonsai2_gguf_to_ir import (DENSE_MAP, GROUP, LM, PREFIX, QUANT_MAP, const_of, find_gguf_py,
                                gguf_f32, replace_const, u2_chain, vperm)

Q1_BYTES = 2 + GROUP // 8
# OpenVINO packs u1 MSB first, Q1_0 LSB first.
BITREV = np.array([int(f"{b:08b}"[::-1], 2) for b in range(256)], np.uint8)

QWEN3_DENSE = {
    "input_layernorm": "attn_norm.weight",
    "post_attention_layernorm": "ffn_norm.weight",
    "self_attn.q_norm": "attn_q_norm.weight",
    "self_attn.k_norm": "attn_k_norm.weight",
}


def q1_blocks(t) -> np.ndarray:
    """Q1_0 tensor -> uint8 [N, K/128, 18] (fp16 scale, then 16 sign bytes)."""
    n, k = (int(x) for x in reversed(t.shape))
    if t.tensor_type.name != "Q1_0":
        sys.exit(f"{t.name}: expected Q1_0, got {t.tensor_type.name}")
    raw = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
    if raw.size != n * (k // GROUP) * Q1_BYTES:
        sys.exit(f"{t.name}: Q1_0 byte length mismatch")
    return raw.reshape(n, k // GROUP, Q1_BYTES)


def q1_scales(blocks: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(blocks[:, :, :2]).view("<f2").reshape(blocks.shape[:2])


def u1_weight(blocks: np.ndarray, out_type: ov.Type, name: str):
    """[N, G, 18] Q1_0 blocks -> decompression subgraph of a [N', K] weight, N' = N rounded up to 16."""
    n, g, _ = blocks.shape
    n_pad = -(-n // 16) * 16
    bits = np.zeros((n_pad, g, GROUP // 8), np.uint8)
    bits[:n] = BITREV[blocks[:, :, 2:]]
    scale = np.zeros((n_pad, g, 1), np.float16)
    scale[:n, :, 0] = 2 * q1_scales(blocks).astype(np.float32)
    if not np.isfinite(scale).all():
        sys.exit(f"{name}: 2 * scale overflows fp16")
    tq = ov.Tensor(ov.Type.u1, [n_pad, g, GROUP])
    np.frombuffer(tq.data, dtype=np.uint8)[:] = bits.reshape(-1)
    q = ops.constant(tq)
    q.set_friendly_name(name + "_compressed")
    w = ops.subtract(ops.convert(q, ov.Type.f16), ops.constant(np.full((1, 1, 1), 0.5, np.float16)))
    w = ops.multiply(w, ops.constant(scale))
    w = ops.reshape(w, ops.constant(np.array([n_pad, g * GROUP], np.int64)), False)
    return ops.convert(w, out_type), n_pad


def bit_planes(blocks: np.ndarray):
    """[V, G, 18] -> u8 [V, K/8] table whose byte j holds columns j + p*K/8 at bit p, and fp16 [V, G] scales."""
    v, g, _ = blocks.shape
    k8 = g * GROUP // 8
    table = np.zeros((v, k8), np.uint8)
    for r0 in range(0, v, 16384):
        b = np.unpackbits(blocks[r0:r0 + 16384, :, 2:], axis=-1, bitorder="little").reshape(-1, g * GROUP)
        for p in range(8):
            table[r0:r0 + 16384] |= b[:, p * k8:(p + 1) * k8] << p
    return table, q1_scales(blocks).copy()


def binary_embedding(blocks: np.ndarray, idx) -> ov.Output:
    """Gather the 1-bit rows of `idx` (i32) and dequantise them to f32 on the device."""
    v, g, _ = blocks.shape
    table, scales = bit_planes(blocks)
    axis = ops.constant(np.int32(0))
    qf = ops.convert(ops.gather(ops.constant(table), idx, axis), ov.Type.f32)
    planes = []
    for p in range(8):
        x = ops.floor(ops.divide(qf, ops.constant(np.float32(2 ** p)))) if p else qf
        planes.append(ops.floor_mod(x, ops.constant(np.float32(2))) if p < 7 else x)
    w = ops.subtract(ops.multiply(ops.concat(planes, -1), ops.constant(np.float32(2))),
                     ops.constant(np.float32(1)))                                  # bits -> -1 / +1
    s = ops.convert(ops.gather(ops.constant(scales), idx, axis), ov.Type.f32)
    w = ops.multiply(ops.reshape(w, ops.constant(np.array([0, 0, g, GROUP], np.int64)), True),
                     ops.unsqueeze(s, ops.constant(np.int64(-1))))
    return ops.reshape(w, ops.constant(np.array([0, 0, g * GROUP], np.int64)), True).output(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--template-dir", required=True,
                    help="dir with the ternary (u2) openvino_model.xml of the same architecture")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gguf-py")
    a = ap.parse_args()

    sys.path.insert(0, find_gguf_py(a.gguf_py))
    from gguf import GGUFReader  # noqa: E402

    t0 = time.perf_counter()
    r = GGUFReader(a.gguf)
    fields = {k: v.contents() for k, v in r.fields.items()}
    arch = fields.get("general.architecture")
    if arch not in ("qwen3", "qwen35"):
        sys.exit(f"unsupported architecture {arch!r}")
    g = lambda key: fields[f"{arch}.{key}"]  # noqa: E731
    tensors = {t.name: t for t in r.tensors}
    print(f"[ir] {a.gguf}: {arch}, {len(tensors)} tensors")

    rows_of = cols_of = {}
    if arch == "qwen35":
        nv, nk = int(g("ssm.time_step_rank")), int(g("ssm.group_count"))
        hd = int(g("ssm.inner_size")) // nv
        if hd != GROUP:
            sys.exit("ssm_out column permutation needs head_v_dim == 128")
        qk_rows = 2 * nk * int(g("ssm.state_size"))
        perm_hd, perm_1 = vperm(nv, nk, hd), vperm(nv, nk, 1)
        # undo llama.cpp's tiled V-head order
        rows_of = {"attn_qkv": np.concatenate([np.arange(qk_rows), qk_rows + perm_hd]),
                   "attn_gate": perm_hd, "ssm_alpha": perm_1, "ssm_beta": perm_1}
        cols_of = {"ssm_out": perm_1}      # whole 128-groups move

    core = ov.Core()
    model = core.read_model(os.path.join(a.template_dir, "openvino_model.xml"))

    # ---- 1-bit MatMul weights ---------------------------------------------------
    layer_re = re.compile(r".*layers\.(\d+)\.(.+)/(?:aten|ov_ext)::linear/MatMul$")
    head_re = re.compile(r".*lm_head/(?:aten|ov_ext)::linear/MatMul$")
    n_u1 = n_pad = 0
    for mm in [n for n in model.get_ordered_ops() if n.get_type_name() == "MatMul"]:
        name = mm.get_friendly_name()
        if head_re.match(name):
            gname, stem = "output.weight", None
        else:
            m = layer_re.match(name)
            if not m:
                continue
            if m.group(2) not in QUANT_MAP:
                sys.exit(f"unmapped MatMul {name}")
            stem = QUANT_MAP[m.group(2)]
            gname = f"blk.{m.group(1)}.{stem}.weight"
        u2c, _ = u2_chain(mm)
        if u2c is None:
            sys.exit(f"{name}: template weight is not u2")
        t = tensors.get(gname)
        if t is None:
            sys.exit(f"{name}: {gname} missing from GGUF")
        blocks = q1_blocks(t)
        if stem in rows_of:
            blocks = blocks[rows_of[stem]]
        if stem in cols_of:
            blocks = blocks[:, cols_of[stem]]
        if list(u2c.get_output_shape(0)) != [blocks.shape[0], blocks.shape[1], GROUP]:
            sys.exit(f"{name}: template {list(u2c.get_output_shape(0))} vs GGUF {list(blocks.shape[:2])}")
        w, padded = u1_weight(blocks, mm.input_value(1).get_element_type(), u2c.get_friendly_name().replace("_compressed", ""))
        mm.input(1).replace_source_output(w.output(0))
        if padded != blocks.shape[0]:
            out = mm.output(0)
            rank = out.get_partial_shape().rank.get_length()
            sl = ops.slice(out, np.array([0], np.int64), np.array([blocks.shape[0]], np.int64),
                           np.array([1], np.int64), np.array([rank - 1], np.int64))
            for tgt in list(out.get_target_inputs()):
                if tgt.get_node() != sl:
                    tgt.replace_source_output(sl.output(0))
            names = out.get_names()
            out.get_tensor().set_names(set())
            sl.output(0).get_tensor().set_names(names)
            n_pad += 1
        n_u1 += 1
    print(f"[ir] weights: {n_u1} binary, {n_pad} padded to 16 ({time.perf_counter() - t0:.0f}s)")

    # ---- dense per-layer tensors --------------------------------------------------
    n_dense = 0
    for node in model.get_ordered_ops():
        name = node.get_friendly_name()
        if not name.startswith(PREFIX):
            continue
        rest = re.sub(r"^.*?(?=layers\.|norm/)", "", name[len(PREFIX):], count=1)
        if rest == "norm/aten::mul/Multiply_1":
            gname, kind = "output_norm.weight", "norm"
        else:
            m = re.match(r"layers\.(\d+)\.(.+)$", rest)
            if not m:
                continue
            layer, suffix = int(m.group(1)), m.group(2)
            if arch == "qwen35":
                if suffix not in DENSE_MAP:
                    continue
                stem, kind = DENSE_MAP[suffix]
            else:
                mod = suffix.split("/")[0]
                if suffix != f"{mod}/aten::mul/Multiply_1" or mod not in QWEN3_DENSE:
                    continue
                stem, kind = QWEN3_DENSE[mod], "norm"
            gname = f"blk.{layer}.{stem}"
        t = tensors.get(gname)
        if t is None:
            sys.exit(f"{name}: {gname} missing from GGUF")
        x = gguf_f32(t)
        c = next((c for c in (const_of(node, p) for p in ((0,) if kind == "a_log" else (1, 0)))
                  if c is not None and int(np.prod(c.get_output_shape(0))) == x.size), None)
        if c is None:
            sys.exit(f"{name}: no constant of {x.size} elements")
        if kind == "a_log":
            if not (x < 0).all():
                sys.exit(f"{gname}: expected A = -exp(A_log) < 0")
            x = (-x)[perm_1]                    # template holds exp(A_log)
        elif kind == "dt_bias":
            x = x[perm_1]
        elif kind == "conv1d":
            x = x[rows_of["attn_qkv"]]
        replace_const(c, x.reshape(list(c.get_output_shape(0))))
        n_dense += 1
    print(f"[ir] dense tensors: {n_dense} ({time.perf_counter() - t0:.0f}s)")

    # ---- embedding ------------------------------------------------------------------
    emb_blocks = q1_blocks(tensors["token_embd.weight"])
    os.makedirs(a.out_dir, exist_ok=True)
    if arch == "qwen3":
        gather = next(n for n in model.get_ordered_ops()
                      if n.get_type_name() == "Gather" and "embed_tokens" in n.get_friendly_name())
        emb = binary_embedding(emb_blocks, gather.input_value(1))
        for tgt in list(gather.output(0).get_target_inputs()):
            tgt.replace_source_output(emb)
    else:
        ids = ops.parameter([-1, -1], ov.Type.i64, name="input")
        res = ops.result(binary_embedding(emb_blocks, ops.convert(ids, ov.Type.i32)))
        res.output(0).get_tensor().set_names({"inputs_embeds"})
        ov.save_model(ov.Model([res], [ids], "text_embeddings"),
                      os.path.join(a.out_dir, "openvino_text_embeddings_model.xml"), compress_to_fp16=False)
    print(f"[ir] embedding: binary {emb_blocks.shape[0]}x{emb_blocks.shape[1] * GROUP} "
          f"({time.perf_counter() - t0:.0f}s)")
    del emb_blocks

    model.validate_nodes_and_infer_types()
    ov.save_model(model, os.path.join(a.out_dir, "openvino_model.xml"), compress_to_fp16=False)
    print(f"[ir] wrote {a.out_dir} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
