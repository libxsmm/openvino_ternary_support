#!/usr/bin/env python3
"""Build the MTP (multi-token prediction) draft IR for Bonsai 2 27B.

The draft is the ProCreations MTP head (model_mtp.safetensors: one Qwen3.5
full-attention decoder layer + fc + norms) sharing the target's lm_head. The
graph is cut out of the Bonsai 2 u2 IR written by bonsai2_gguf_to_ir.py, so the
attention block (gated q_proj, q/k norm, M-RoPE, SDPA with a stateful KV cache)
and the u2 lm_head come out exactly as the target has them, and the in-process
SDPAToPagedAttention conversion treats both models the same way:

    inputs_embeds [B, S, 10240] = [ embed(token t+1) | h_t ]
        -> pre_fc_norm_embedding / pre_fc_norm_hidden -> concat -> fc
        -> layers.3 of the IR with the MTP weights (no rotation)
        -> mtp.norm -> Hadamard -> u2 lm_head (the target's)
    outputs: logits [B, S, V], last_hidden_state [B, S, 5120]

The runtime inserts a row Gather in front of the final norm after
SDPAToPagedAttention, so only the rows it needs (the last one of each
sequence) go through the head.

h_t is the target's post-final-norm hidden state (vLLM's convention for
Qwen3.5 MTP), taken where the target IR feeds its lm_head, i.e. before the
Hadamard reshape. Bonsai 2 folds the lm_head input sign flip s into that norm
weight, so the hidden state the runtime sees is s * h. RMSNorm commutes with a
sign flip, so s is folded into pre_fc_norm_hidden; mtp.norm gets s folded the
same way so the draft's own last_hidden_state (fed back for the next draft
step) and its lm_head input stay in the target's convention.

    python bonsai2_mtp_to_ir.py \
        --ir     <bonsai2-27b-u2>/openvino_model.xml \
        --mtp    <Ternary-Bonsai-2-27B-MTP>/model_mtp.safetensors \
        --gguf   <Ternary-Bonsai-2-27B-PQ2_0.gguf> --gguf-py <prism llama.cpp>/gguf-py \
        --out    <bonsai2-27b-u2>/openvino_mtp_model.xml [--weights i8]

--weights i8 stores the head matrices as int8 with one fp16 scale per 128
inputs (the GPU plugin runs them as compressed FCs; ~30% faster draft steps,
same acceptance on the bench prompt).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import openvino as ov
from openvino import opset13 as ops

P = "__module.model.model.language_model."
TEMPLATE_LAYER = 3          # first full-attention layer of the 64
HIDDEN = 5120
EPS = 1e-6

# MTP layer MatMuls (IR leaf -> safetensors key)
MATMULS = {
    "self_attn.q_proj": "mtp.layers.0.self_attn.q_proj.weight",
    "self_attn.k_proj": "mtp.layers.0.self_attn.k_proj.weight",
    "self_attn.v_proj": "mtp.layers.0.self_attn.v_proj.weight",
    "self_attn.o_proj": "mtp.layers.0.self_attn.o_proj.weight",
    "mlp.gate_proj": "mtp.layers.0.mlp.gate_proj.weight",
    "mlp.up_proj": "mtp.layers.0.mlp.up_proj.weight",
    "mlp.down_proj": "mtp.layers.0.mlp.down_proj.weight",
}
NORMS = {
    "input_layernorm": "mtp.layers.0.input_layernorm.weight",
    "post_attention_layernorm": "mtp.layers.0.post_attention_layernorm.weight",
    "self_attn.q_norm": "mtp.layers.0.self_attn.q_norm.weight",
    "self_attn.k_norm": "mtp.layers.0.self_attn.k_norm.weight",
}


def load_signs(gguf: str, gguf_py: str) -> np.ndarray:
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader  # noqa: E402

    fields = {k: v.contents() for k, v in GGUFReader(gguf).fields.items()
              if k.startswith("prism.hadamard.")}
    if int(fields.get("prism.hadamard.version", 0)) != 1:
        sys.exit("GGUF has no prism.hadamard contract")
    vals = np.asarray(fields["prism.hadamard.sign_values"], dtype=np.float32)
    off = 0
    for w in fields["prism.hadamard.sign_widths"]:
        w = int(w)
        if w == HIDDEN:
            return vals[off:off + w]
        off += w
    sys.exit(f"no {HIDDEN}-wide sign vector in the GGUF")


def const_behind(node, port: int):
    src = node.input_value(port).get_node()
    while src.get_type_name() in ("Convert", "Reshape"):
        src = src.input_value(0).get_node()
    return src if src.get_type_name() == "Constant" else None


def replace_const(old, data: np.ndarray):
    new = ops.constant(data.astype(old.get_output_element_type(0).to_dtype()))
    new.set_friendly_name(old.get_friendly_name())
    for tgt in list(old.output(0).get_target_inputs()):
        tgt.replace_source_output(new.output(0))


def unrotate(mm):
    """Rewire MatMul input 0 past the Reshape -> MatMul(H) -> Reshape [-> sign
    Multiply] chain bonsai2_gguf_to_ir.py inserted (the MTP weights are in the
    plain basis)."""
    src = mm.input_value(0).get_node()
    if not src.get_friendly_name().endswith("/hadamard"):
        return False
    h = src.input_value(0).get_node()            # MatMul with H
    r = h.input_value(0).get_node()              # Reshape to [.., K/1024, 1024]
    x = r.input_value(0)
    if x.get_node().get_type_name() == "Multiply" and const_behind(x.get_node(), 1) is not None \
            and "/aten::" not in x.get_node().get_friendly_name():
        x = x.get_node().input_value(0)          # explicit sign multiply
    if h.get_type_name() != "MatMul" or r.get_type_name() != "Reshape":
        sys.exit(f"{mm.get_friendly_name()}: unexpected rotation chain")
    mm.input(0).replace_source_output(x)
    return True


def rms_norm(x, weight: np.ndarray):
    sq = ops.multiply(x, x)
    var = ops.reduce_mean(sq, ops.constant(np.array([-1], np.int64)), True)
    inv = ops.divide(ops.constant(np.array(1.0, np.float32)),
                     ops.sqrt(ops.add(var, ops.constant(np.array(EPS, np.float32)))))
    return ops.multiply(ops.multiply(x, inv), ops.constant(weight.reshape(1, 1, -1).astype(np.float32)))


def weight(w: np.ndarray, et: ov.Type, fmt: str):
    """[N, K] weight subgraph: fp16 constant, or int8 with one fp16 scale per
    128 inputs (symmetric; the GPU plugin compresses it into the FC)."""
    if fmt == "f16":
        return ops.convert(ops.constant(w.astype(np.float16)), et)
    n, k = w.shape
    g = w.reshape(n, k // 128, 128)
    scale = np.maximum(np.abs(g).max(axis=2, keepdims=True), 1e-12) / 127.0
    q = np.clip(np.rint(g / scale), -127, 127).astype(np.int8)
    x = ops.multiply(ops.convert(ops.constant(q), ov.Type.f16), ops.constant(scale.astype(np.float16)))
    x = ops.reshape(x, ops.constant(np.array([n, k], np.int64)), False)
    return ops.convert(x, et)


def dense(x, w: np.ndarray, fmt: str):
    return ops.matmul(x, weight(w, ov.Type.f32, fmt), False, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True, help="Bonsai 2 u2 openvino_model.xml")
    ap.add_argument("--mtp", required=True, help="model_mtp.safetensors")
    ap.add_argument("--gguf", required=True, help="Ternary-Bonsai-2-27B-PQ2_0.gguf (sign vectors)")
    ap.add_argument("--gguf-py", required=True, help="PrismML llama.cpp fork gguf-py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", choices=("f16", "i8"), default="f16",
                    help="head matrices: fp16, or int8 with a scale per 128 inputs")
    a = ap.parse_args()

    t0 = time.perf_counter()
    from safetensors import safe_open
    with safe_open(a.mtp, "pt") as f:
        W = {k: f.get_tensor(k).float().numpy() for k in f.keys()}
    s = load_signs(a.gguf, a.gguf_py)
    print(f"[mtp] {len(W)} head tensors, sign vector {s.shape} ({time.perf_counter() - t0:.0f}s)")

    model = ov.Core().read_model(a.ir)
    ops_by_name = {n.get_friendly_name(): n for n in model.get_ordered_ops()}
    L = f"{P}layers.{TEMPLATE_LAYER}."

    def node(name):
        n = ops_by_name.get(name)
        if n is None:
            sys.exit(f"node {name} not found")
        return n

    # ---- front: [emb | hidden] -> norms -> concat -> fc ---------------------
    old_in = next(p for p in model.get_parameters() if p.get_friendly_name() == "inputs_embeds")
    x_in = ops.parameter(ov.PartialShape([-1, -1, 2 * HIDDEN]), ov.Type.f32)
    x_in.set_friendly_name("inputs_embeds")
    x_in.output(0).get_tensor().set_names({"inputs_embeds"})
    for tgt in list(old_in.output(0).get_target_inputs()):
        tgt.replace_source_output(x_in.output(0))
    emb, hid = ops.variadic_split(x_in, ops.constant(np.array(-1, np.int64)),
                                  ops.constant(np.array([HIDDEN, HIDDEN], np.int64))).outputs()
    e_n = rms_norm(emb, 1.0 + W["mtp.pre_fc_norm_embedding.weight"])
    h_n = rms_norm(hid, (1.0 + W["mtp.pre_fc_norm_hidden.weight"]) * s)
    fc = dense(ops.concat([e_n, h_n], -1), W["mtp.fc.weight"], a.weights)
    fc.set_friendly_name("mtp.fc")

    # ---- the decoder layer ---------------------------------------------------
    prev = node(f"{P}layers.{TEMPLATE_LAYER - 1}/aten::add/Add_1")
    in_layer = (L, L[:-1] + "/")        # "layers.3.<module>/..." and "layers.3/aten::add/..."
    n_fed = 0
    for tgt in list(prev.output(0).get_target_inputs()):
        if not tgt.get_node().get_friendly_name().startswith(in_layer):
            continue
        tgt.replace_source_output(fc.output(0))
        n_fed += 1
    if n_fed < 2:
        sys.exit(f"layer {TEMPLATE_LAYER} input: rewired only {n_fed} consumers")
    n_rot = 0
    for leaf, key in MATMULS.items():
        mm = node(f"{L}{leaf}/ov_ext::linear/MatMul")
        n_rot += unrotate(mm)
        w = W[key]
        shp = list(mm.input_value(1).get_partial_shape().to_shape())
        if shp != list(w.shape):
            sys.exit(f"{leaf}: IR weight {shp} vs MTP {list(w.shape)}")
        if not mm.get_transpose_b():
            sys.exit(f"{leaf}: expected transpose_b")
        wc = weight(w, mm.input_value(1).get_element_type(), a.weights)
        mm.input(1).replace_source_output(wc.output(0))
    for leaf, key in NORMS.items():
        mul = node(f"{L}{leaf}/aten::mul/Multiply_1")
        c = const_behind(mul, 1)
        if c is None or int(np.prod(c.get_output_shape(0))) != W[key].size:
            sys.exit(f"{leaf}: norm constant not found")
        replace_const(c, (1.0 + W[key]).reshape(c.get_output_shape(0)))
    print(f"[mtp] layer {TEMPLATE_LAYER}: {len(MATMULS)} {a.weights} projections, {n_rot} rotations removed")

    # ---- tail: mtp.norm -> (Hadamard) lm_head ---------------------------------
    # Every row goes through the head here; the runtime gathers the rows it
    # wants in front of the final norm after SDPAToPagedAttention (the token
    # axis only becomes known there), as it does for the target.
    out = node(f"{L[:-1]}/aten::add/Add_1")
    tail_in = node(f"{P}layers.63/aten::add/Add_1")
    for tgt in list(tail_in.output(0).get_target_inputs()):
        if not tgt.get_node().get_friendly_name().startswith(P + "norm/"):
            sys.exit(f"unexpected final-norm consumer {tgt.get_node().get_friendly_name()}")
        tgt.replace_source_output(out.output(0))
    fnorm = node(f"{P}norm/aten::mul/Multiply_1")
    replace_const(const_behind(fnorm, 1), ((1.0 + W["mtp.norm.weight"]) * s).reshape(-1))

    logits = next(r for r in model.get_results() if "logits" in r.output(0).get_names()
                  or r.input_value(0).get_node().get_friendly_name().endswith("lm_head/ov_ext::linear/MatMul"))
    hidden = ops.result(fnorm.output(0))
    hidden.output(0).get_tensor().set_names({"last_hidden_state"})
    fnorm.output(0).get_tensor().set_names({"last_hidden_state"})

    # ---- state of the one attention layer -------------------------------------
    sdpa = node(f"{L}self_attn/aten::scaled_dot_product_attention/ScaledDotProductAttention")
    var_ids = set()
    stack = [sdpa.input_value(1).get_node(), sdpa.input_value(2).get_node()]
    seen = set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if n.get_type_name() == "ReadValue":
            var_ids.add(n.get_variable_id())
            continue
        stack.extend(v.get_node() for v in n.input_values())
    sinks = [snk for snk in model.get_sinks() if snk.get_variable_id() in var_ids]
    if len(sinks) != 2:
        sys.exit(f"expected 2 KV Assigns for layer {TEMPLATE_LAYER}, found {len(sinks)}")

    params = [x_in] + [p for p in model.get_parameters()
                       if p.get_friendly_name() in ("attention_mask", "position_ids", "beam_idx")]
    draft = ov.Model([logits, hidden], sinks, params, "bonsai2_mtp_draft")

    # The mask/position subgraph takes the batch and sequence sizes from a
    # ShapeOf on some tensor of the cut-away layers (shared shape
    # subexpressions); re-source those from the draft's own input.
    def dead(n):
        name = n.get_friendly_name()
        return "language_model.layers." in name and not name.startswith(in_layer)
    n_shape = 0
    for n in draft.get_ordered_ops():
        if n.get_type_name() != "ShapeOf" or not dead(n.input_value(0).get_node()):
            continue
        ps = n.input_value(0).get_partial_shape()
        xs = ops.shape_of(x_in, n.get_output_element_type(0))
        parts, dyn = [], 0
        for d in ps:
            if d.is_dynamic:
                parts.append(ops.gather(xs, ops.constant(np.array([dyn], np.int64)),
                                        ops.constant(np.array(0, np.int64))))
                dyn += 1
            else:
                parts.append(ops.constant(np.array([d.get_length()],
                                                   n.get_output_element_type(0).to_dtype())))
        if dyn > 2:
            sys.exit(f"{n.get_friendly_name()}: cannot map shape {ps} onto [batch, seq]")
        new = ops.concat(parts, 0)
        for tgt in list(n.output(0).get_target_inputs()):
            tgt.replace_source_output(new.output(0))
        n_shape += 1
    draft = ov.Model([logits, hidden], sinks, params, "bonsai2_mtp_draft")
    stray = [n.get_friendly_name() for n in draft.get_ordered_ops() if dead(n)]
    if stray:
        sys.exit(f"draft still reaches cut-away layers: {stray[:5]}")
    names = sorted(p.get_friendly_name() for p in draft.get_parameters())
    n_ops = len(draft.get_ordered_ops())
    print(f"[mtp] draft: {n_ops} ops ({n_shape} ShapeOf re-sourced), inputs {names}, outputs "
          f"{[sorted(o.get_names()) for o in draft.outputs]}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ov.save_model(draft, a.out, compress_to_fp16=False)
    print(f"[mtp] wrote {a.out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
