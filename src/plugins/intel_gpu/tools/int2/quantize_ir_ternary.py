# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rewrite the fp16 MatMul weights of an exported LLM into ternary u2 weights.

OpenVINO has no signed 2-bit type, so ternary is expressed the way the GPU
plugin's weight-compression pattern expects: u2 codes {0,1,2} with a scalar zero
point of 1, times a per-group fp16 scale. Weights that are already exactly
ternary are reproduced rather than approximated.
"""

import argparse
import gc
import numpy as np
import openvino as ov
from openvino import opset13 as ops

GROUP = 128


def pack_u2(codes: np.ndarray) -> np.ndarray:
    """[..] uint8 in {0,1,2} -> packed u2, value j at bits [2j, 2j+1]."""
    flat = codes.reshape(-1).astype(np.uint8) & 0x3
    if flat.size % 4:
        flat = np.pad(flat, (0, 4 - flat.size % 4))
    q = flat.reshape(-1, 4)
    return (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).astype(np.uint8)


def to_f32(const) -> np.ndarray:
    """Constant data as fp32, decoding bf16 by hand.

    numpy has no bf16, so Constant.get_data() hands back a float16-typed *view*
    of the raw bf16 bits rather than a conversion. Reading that as fp16 keeps the
    ternary codes (the sign is bit 15 either way) but corrupts every group scale,
    which is silent: the model loads and generates fluent nonsense.
    """
    raw = const.get_data()
    if const.get_output_element_type(0) == ov.Type.bf16:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32)


def pack_i4(q: np.ndarray) -> np.ndarray:
    """[..] int8 in [-8, 7] -> packed i4, value j at bits [4j, 4j+3]."""
    flat = q.reshape(-1).astype(np.uint8) & 0xF
    if flat.size % 2:
        flat = np.pad(flat, (0, 1))
    q2 = flat.reshape(-1, 2)
    return (q2[:, 0] | (q2[:, 1] << 4)).astype(np.uint8)


def ternary_quantize(w: np.ndarray):
    """w is [..., N, K] fp32. Returns codes {-1,0,1} int8 [..., N, K] and fp16 scales [..., N, K/GROUP]."""
    *lead, k = w.shape
    g = w.reshape(*lead, k // GROUP, GROUP)
    scale = np.abs(g).max(axis=-1)
    safe = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.rint(g / safe[..., None]), -1, 1).astype(np.int8).reshape(*lead, k)
    return q, scale.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--min-k", type=int, default=1024, help="skip tiny MatMuls")
    ap.add_argument("--skip-n", type=int, default=0, help="keep MatMuls with this N dense (non-ternary lm_head)")
    ap.add_argument("--dtype", choices=["u2", "i4"], default="u2",
                    help="i4: the same ternary codes in signed 4-bit without a zero point (a lossless baseline)")
    ap.add_argument("--only-ternary", action="store_true",
                    help="leave weights dense unless they are exactly ternary (e.g. MoE router gates)")
    args = ap.parse_args()

    core = ov.Core()
    model = core.read_model(args.src)

    targets = []
    for node in model.get_ordered_ops():
        if node.get_type_name() != "MatMul":
            continue
        w = node.input_value(1).get_node()
        # optimum emits fp16 weights behind a Convert to the inference precision
        conv = None
        if w.get_type_name() == "Convert":
            conv, w = w, w.input_value(0).get_node()
        if w.get_type_name() != "Constant":
            continue
        shape = list(w.get_output_shape(0))
        # rank 3 is [experts, N, K] (MoE)
        if len(shape) not in (2, 3) or shape[-1] % GROUP or shape[-1] < args.min_k or shape[-2] == args.skip_n:
            continue
        targets.append((node, conv, w, shape))

    print(f"rewriting {len(targets)} MatMul weights")
    total_in = total_out = 0
    skipped = []
    for i, (mm, conv, const, shape) in enumerate(targets):
        *lead, k = shape
        grouped = [*lead, k // GROUP, GROUP]
        # whatever fed the MatMul before, the replacement has to keep that type
        orig_et = mm.input_value(1).get_element_type()
        w = to_f32(const)
        q, scale = ternary_quantize(w)
        if args.only_ternary:
            s = np.repeat(scale.astype(np.float32), GROUP, axis=-1)
            dev = np.abs(w - q * s).max() / max(float(np.abs(s).max()), 1e-30)
            del s
            if dev > 1e-2:
                skipped.append((mm.get_friendly_name(), shape, round(float(dev), 3)))
                continue
        total_in += w.size * 2
        del w

        if args.dtype == "u2":
            packed, et, bits = pack_u2((q + 1).astype(np.uint8)), ov.Type.u2, 2
        else:
            packed, et, bits = pack_i4(q), ov.Type.i4, 4
        t = ov.Tensor(et, grouped)
        np.frombuffer(t.data, dtype=np.uint8)[: packed.size] = packed
        wq = ops.constant(t)
        del q, packed

        sc = ops.constant(scale.reshape(*grouped[:-1], 1))
        total_out += scale.size * 2 + int(np.prod(shape)) * bits // 8
        del scale

        deq = ops.convert(wq, "f16")
        if args.dtype == "u2":
            # MoE ops require num_experts as the first dimension of every weight input, zp included
            zp_shape = [lead[0]] + [1] * (len(grouped) - 1) if len(lead) > 1 else [1] * len(grouped)
            deq = ops.subtract(deq, ops.constant(np.ones(zp_shape, np.float16)))
        deq = ops.multiply(deq, sc)
        flat = ops.reshape(deq, ops.constant(np.array(shape, np.int64)), False)
        out = flat if orig_et == ov.Type.f16 else ops.convert(flat, orig_et)
        mm.input(1).replace_source_output(out.output(0))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(targets)}")
            gc.collect()

    print(f"weights {total_in / 2**30:.2f} GiB -> {total_out / 2**30:.2f} GiB")
    if skipped:
        print(f"left {len(skipped)} non-ternary weights dense, e.g. {skipped[:3]}")
    ov.save_model(model, args.dst, compress_to_fp16=False)
    print("saved", args.dst)


if __name__ == "__main__":
    main()
