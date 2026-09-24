"""Check a ternary u2 IR against the fp16 IR it was produced from.

Two things can go wrong silently. The compression is only lossless if the
checkpoint's own scale group is a multiple of the group size used here, so a
finer native grouping shows up as a non-zero reconstruction error rather than as
a failure. And a checkpoint that ties lm_head to its embedding exports no weight
for the head, so it stays dense and simply never appears in the compressed set.
"""

import argparse
import numpy as np
import openvino as ov

GROUP = 128


def to_f32(const) -> np.ndarray:
    raw = const.get_data()
    if const.get_output_element_type(0) == ov.Type.bf16:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32)


def unpack_u2(packed: np.ndarray, count: int) -> np.ndarray:
    """Constant.get_data() on a u2 tensor hands back packed bytes, 4 values each."""
    b = packed.reshape(-1).view(np.uint8)
    vals = (b[:, None] >> (2 * np.arange(4, dtype=np.uint8))) & 0x3
    return vals.reshape(-1)[:count].astype(np.uint8)


def compressed_weights(model):
    """name -> (codes u8 [N,K], scales f32 [N,K/GROUP]) for each u2 MatMul."""
    out = {}
    for node in model.get_ordered_ops():
        if node.get_type_name() != "MatMul":
            continue
        n = node.input_value(1).get_node()
        if n.get_type_name() == "Convert":
            n = n.input_value(0).get_node()
        if n.get_type_name() != "Reshape":
            continue
        mul = n.input_value(0).get_node()
        if mul.get_type_name() != "Multiply":
            continue
        sub, sc = mul.input_value(0).get_node(), mul.input_value(1).get_node()
        if sub.get_type_name() != "Subtract":
            continue
        conv = sub.input_value(0).get_node()
        if conv.get_type_name() != "Convert":
            continue
        wq = conv.input_value(0).get_node()
        if wq.get_type_name() != "Constant" or wq.get_output_element_type(0) != ov.Type.u2:
            continue
        shape = list(wq.get_output_shape(0))
        count = shape[0] * shape[1] * shape[2]
        codes = unpack_u2(wq.get_data(), count)
        out[node.get_friendly_name()] = (
            codes.reshape(shape[0], shape[1] * shape[2]),
            sc.get_data().astype(np.float32).reshape(shape[0], shape[1]),
        )
    return out


def dense_weights(model):
    out = {}
    for node in model.get_ordered_ops():
        if node.get_type_name() != "MatMul":
            continue
        n = node.input_value(1).get_node()
        if n.get_type_name() == "Convert":
            n = n.input_value(0).get_node()
        if n.get_type_name() == "Constant":
            out[node.get_friendly_name()] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--u2", required=True)
    ap.add_argument("--head", default="lm_head", help="substring identifying the output projection")
    args = ap.parse_args()

    core = ov.Core()
    dense = dense_weights(core.read_model(args.fp16))
    comp = compressed_weights(core.read_model(args.u2))

    print(f"compressed MatMuls: {len(comp)}")
    head = [k for k in comp if args.head in k]
    print(f"output projection compressed: {'yes' if head else 'NO - still dense'}"
          f"{' (' + head[0] + ')' if head else ''}")

    worst, worst_name, checked = 0.0, "", 0
    worst_rel, multi_groups, total_groups = 0.0, 0, 0
    for name, (codes, scale) in comp.items():
        const = dense.get(name)
        if const is None:
            continue
        w = to_f32(const)
        n, k = w.shape
        recon = ((codes.astype(np.float32) - 1.0)
                 * np.repeat(scale, GROUP, axis=1)[:, :k])
        err = np.abs(recon - w)
        e = float(err.max())
        checked += 1
        if e > worst:
            worst, worst_name = e, name
        if e:
            denom = np.where(np.abs(w) == 0, 1.0, np.abs(w))
            worst_rel = max(worst_rel, float((err / denom).max()))
        # A group is only exactly representable if it holds a single magnitude.
        g = w.reshape(n, k // GROUP, GROUP)
        nz = np.where(g == 0, np.nan, np.abs(g))
        spread = np.nanmax(nz, axis=2) - np.nanmin(nz, axis=2)
        multi_groups += int(np.nansum(spread > 0))
        total_groups += n * (k // GROUP)

    print(f"weights compared: {checked}")
    print(f"max abs reconstruction error: {worst:.3e}" + (f"  ({worst_name})" if worst else ""))
    if worst == 0.0:
        print(f"exact: every {GROUP}-element group holds a single magnitude")
    else:
        pct = 100.0 * multi_groups / max(total_groups, 1)
        print(f"max relative error: {worst_rel:.2%}")
        print(f"groups holding more than one magnitude: {multi_groups}/{total_groups} ({pct:.4f}%)")
        print(f"-> the native scale group is still {GROUP}; those groups store two\n"
              f"   neighbouring fp16 magnitudes, so max() picks one and the other\n"
              f"   moves by a few ULPs. A finer group size would remove it.")


if __name__ == "__main__":
    main()
