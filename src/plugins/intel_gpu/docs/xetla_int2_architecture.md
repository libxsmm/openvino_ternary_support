# XeTLA int2 FullyConnected — Architecture

How 2-bit (ternary) `FullyConnected` layers are executed by XeTLA kernels inside
the OpenVINO GPU plugin.

---

## 1. Overview

Ternary models store each weight as one of `{-1, 0, +1}` with a per-group fp16
scale. OpenVINO has no signed 2-bit type, so they are carried in the IR as
unsigned `u2` codes `{0,1,2}` with a scalar zero point of 1, dequantized as
`(code - 1) * scale`.

The implementation does three things:

1. **At compile time**, converts those constants once into the exact layout the
   kernels read, so inference performs no unpacking, reordering or allocation.
2. **At execution time**, launches a single XeTLA GEMM per layer, applying the
   SiLU and residual post-ops and producing the output type the consumer wants.
3. **Per shape**, selects a tile and k-slicing configuration from a table of
   tuned entries.

Two kernel variants implement step 2. Both run on DPAS and differ in the operand
types fed to it:

| Variant | Weights | Activations | MMA |
|---|---|---|---|
| **up-convert** (default) | dequantized to fp16 in registers | native fp16 | fp16 DPAS |
| **int2 x int8** | consumed as 2-bit directly | quantized to int8 | integer DPAS |

Up-convert is accurate and is the fast choice for batch-1 decode. int2 x int8 is
faster on the large GEMMs of a long prefill but costs roughly an order of
magnitude more relative error per GEMM, because it quantizes the activations.
Section 7 covers how the variant is chosen at runtime; the rest of this document
notes where the two differ.

| File | Role |
|---|---|
| `impls/sycl/fully_connected_xetla_int2.hpp` | Registration and acceptance rules |
| `impls/sycl/fully_connected_xetla_int2.cpp` | Weight packing, buffers, execution, variant dispatch |
| `impls/sycl/xetla/xetla_int2_gemv.cpp` | Up-convert kernel, epilogue, tuning table |
| `impls/sycl/xetla/xetla_int2_dpas.cpp` | int2 x int8 kernel and its activation-scale reduction |
| `registry/fully_connected_impls.cpp` | Priority relative to oneDNN/OpenCL |

XeTLA is used as an unmodified header-only template library.

---

## 2. Registration and acceptance

`XetlaInt2FCImplementationManager` is registered for `FullyConnected` ahead of
the oneDNN and OpenCL managers, for both dynamic and static shapes:

```cpp
OV_GPU_CREATE_INSTANCE_SYCL_OCL(cldnn::sycl::XetlaInt2FCImplementationManager, shape_types::dynamic_shape)
OV_GPU_CREATE_INSTANCE_SYCL_OCL(cldnn::sycl::XetlaInt2FCImplementationManager, shape_types::static_shape)
OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::FullyConnectedImplementationManager, shape_types::static_shape)
OV_GPU_GET_INSTANCE_OCL(fully_connected, ...)
```

`validate_impl` accepts a node only when every assumption the kernels make
holds; anything else falls through to the existing paths unchanged:

- weights are constant, `u2`, shaped `[N, K]`, with `K % 128 == 0`
- activations are `f16`, `bfyx`, unpadded
- output is `f16` or `f32`, `bfyx`, unpadded
- no bias
- fused primitives limited to a trailing `swish`, or an `eltwise` `sum`/`prod`
  with an f16 outer dependency

Acceptance is conservative and per-node, so the path is incremental: a layer
that does not qualify simply runs as before. Acceptance does not depend on the
variant; both consume the same packed buffers.

---

## 3. Compile-time preparation

Performed once in `create()`, and shared by both variants.

### 3.1 Reading the constants

Weights and scales are read from their `data` nodes with explicit blocking
copies (`copy_to(stream, ..., true)`) into host buffers.

### 3.2 Re-encoding and VNNI16 packing

The kernels consume ternary values in 2-bit two's complement (`-1` as `0b11`),
16 values per `int32`, laid out K-major so a subgroup reads contiguous words:

```cpp
const int32_t code = read_u2(src, n * K + k) - zp;         // {0,1,2} - 1 -> {-1,0,+1}
dst[(k / kPackFactor) * n_stride + n] |= (code & 0x3) << (2 * (k % kPackFactor));
```

with `kPackFactor = 16` and `kGroupSize = 128`. The result is an
`[K/16, n_pad]` `int32` buffer in USM device memory.

### 3.3 N padding

`N` is padded up to a multiple of 64 — the workgroup tile width — so the kernel
stays on its aligned epilogue:

```cpp
const size_t n_pad = (N + 63) & ~static_cast<size_t>(63);
```

Padded columns are zero-filled by the packer and contribute nothing. The
packing stride is stored on the implementation alongside the buffers, so the
weights are always read back with the stride they were written with.

### 3.4 Scales

OpenVINO stores scales per output channel as `[N, K/group]`; the kernels want
`[groups, n_pad]`, so they are transposed once into a padded device buffer:

```cpp
scale_host[g * n_pad + n] = scales_are_n_major ? src[n * groups + g] : src[g * N + n];
```

These are the weight scales. The int2 x int8 variant additionally needs
activation scales, which cannot be precomputed; see section 5.2.

### 3.5 Staging buffers

When `n_pad != N` or the consumer wants f32, a staging buffer of `[M, n_pad]`
is allocated at compile time. The GEMM writes there and a short kernel removes
the row padding, so inference never allocates.

A configuration that uses different variants for prefill and decode also
allocates a single-row f32 staging buffer for decode. The output type is derived
from the buffer actually in use.

### 3.6 Packed-buffer cache

OpenVINO caches `primitive_impl` objects by `kernel_impl_params`, so a single
implementation object can serve several `FullyConnected` nodes of identical
shape. Packed weights therefore cannot live solely on the implementation: they
are held in a process-wide map keyed by node id, and each execution resolves
its own node's buffers. Entries are created once per node and reused
thereafter, so the buffers a running implementation uses remain stable for the
lifetime of the model.

---

## 4. Execution path

```mermaid
flowchart TD
    A["execute_impl"] --> B["resolve packed weights,<br/>scales, staging by node id"]
    B --> C["select variant<br/>(XETLA_INT2_KERNEL, M)"]
    C --> D{"variant"}
    D -->|"up-convert"| E["inspect fused post-ops<br/>POSTOP = 0 / 1 / 2"]
    D -->|"int2 x int8"| F["quantize activations to int8<br/>(scale reduction kernel)"]
    E --> G["gemv_f16(M, n_pad, K, ..., postop, other, out_f32)"]
    F --> H["gemv_f16_dpas(M, n_pad, K, ..., postop, other)"]
    G --> I{"post-ops folded?"}
    H --> I
    I -->|no| J["separate post-op pass"]
    I -->|yes| K{"staging used?"}
    J --> K
    K -->|yes| L["un-pad row stride<br/>(and convert if needed)"]
    K -->|no| M["done"]
    L --> M
```

`M` is derived from the output shape, so one implementation serves both prefill
(`M > 1`) and decode (`M = 1`), and the variant may differ between them.

### Post-op fusion

Post-ops are read from `params->fused_desc` and mapped onto compiled epilogue
variants, with the second operand taken from the fused primitive's outer
dependency:

| Graph pattern | Typical layer | `POSTOP` |
|---|---|---|
| `swish` then `eltwise prod` | gate_proj x up_proj | 1 |
| `eltwise sum` | o_proj, down_proj (residual add) | 2 |

Folding removes a full-tensor read-modify-write pass per layer: the activation
and the residual add are applied in registers while the accumulator tile is
still live. The epilogue is compiled for both variants. Up-convert folds by
default; the int2 x int8 variant runs the separate pass unless
`OV_XETLA_INT2_DPAS_FOLD` is set.

### Output type

Where the consumer wants f32 logits, the up-convert kernel is instantiated with
an f32 `C` type and stores directly; its accumulator is f32 already. Under the
int2 x int8 variant the un-padding pass performs the conversion.

---

## 5. Kernel structure

### 5.1 Up-convert

`int2_upcvt_gemm_impl` is templated on tile shape, k-slicing, output type and
post-op:

```cpp
template <typename XT, int WGM, int WGN, int SGM, int SGN, int SGK,
          int KS, int LS, bool kUnaligned, typename CT = XT, int POSTOP = 0>
```

- `XT` — activation/scale type (`fp16`)
- `CT` — output type (`fp16`, or `float`); `using data_type_c = CT`
- `WGM/WGN`, `SGM/SGN/SGK` — workgroup and subgroup tile shape
- `KS`, `LS` — global and local k-slicing factors
- `POSTOP` — selects the epilogue

The epilogue is composed at compile time from XeTLA's existing tile-op building
blocks:

```cpp
using tile_op_t = std::conditional_t<POSTOP == 1,
    chained_tile_op_t<silu_op_t, elemwise_reduce_op_t<reduce_op::prod, XT, Xe>>,
    chained_tile_op_t<elemwise_reduce_op_t<reduce_op::sum, XT, Xe>>>;

using epilogue_policy = std::conditional_t<POSTOP != 0,
    epilogue_policy_tile_op<tile_op_t, arch_tag>,
    /* default store-only policy */>;
```

Weights are upconverted from 2-bit to fp16 in registers and multiplied by their
group scale before the DPAS, so the mainloop runs as a standard fp16 GEMM.

### 5.2 int2 x int8

`DpasFp16Cfg<POSTOP>` carries the tile shape and feeds the integer DPAS
pipeline. Like the up-convert kernel it is tuned per shape; this integration
compiles a single tuned configuration (`wg 64x256`, `sg 8x128`, `sg_k 64`)
rather than dispatching from a table. It runs in two steps:

1. A reduction kernel computes per-(row, K-group) activation scales as a
   forward quant factor, `127 / absmax`, into a process-wide scratch buffer.
2. The GEMM consumes the 2-bit weights, their fp16 group scales, and those
   activation scales, accumulating in `int32`.

It uses the same `epilogue_t` template as the up-convert path, instantiated with
either the default store-only policy or the same tile-op policy, selected by
`POSTOP`.

### Kernel naming

Both variants are launched as named SYCL kernels
(`parallel_for<int2_upcvt_kernel_tag<...>>`,
`parallel_for<int2_fp16_dpas_kernel_tag<POSTOP>>`), keeping every instantiation
distinct within the module.

---

## 6. Configuration selection

Both variants are tuned per shape. The shape-keyed dispatch below is wired for
the up-convert variant; the int2 x int8 variant currently compiles one tuned
configuration (section 5.2), and a table for it would follow the same pattern.

`gemv_f16` selects a configuration from a table keyed on shape. Decode
(`M = 1`) shapes are matched exactly; anything else falls back to generic tiers
by output width.

The entries below are an **example table, tuned on a B70** for a 4096-wide
hidden size. They are illustrative, not universal: the winning tile and
k-slicing factor depend on the memory bandwidth and EU count of the part, so a
different GPU should be re-swept rather than assumed to match.

| Layer | N | K | WGN | KS | LS |
|---|---|---|---|---|---|
| qkv (fused) | 6144 | 4096 | 32 | 1 | 4 |
| gate/up | 12288 | 4096 | 32 | 1 | 2 |
| o_proj | 4096 | 4096 | 32 | 1 | 8 |
| down_proj | 4096 | 12288 | 32 | 1 | 8 |
| lm_head | 151680 | 4096 | 32 | 1 | 4 |

Narrow workgroups (`WGN = 32`) suit GEMV, where parallelism across `N` matters
more than tile reuse; the local k-slicing factor `LS` balances the K reduction
against subgroup occupancy and is tuned per shape.

`XETLA_INT2_CFG_DEBUG=1` prints the selection;
`XETLA_INT2_DECODE_CFG` / `XETLA_INT2_OPROJ_CFG` override it for sweeps.

K-slicing scratch is a single process-wide buffer that only grows, sized for
the worst case across the compiled instantiations.

Prefill (`M > 1`) on the up-convert path uses a generic configuration and is not
tuned.

---

## 7. Choosing a variant at runtime

`XETLA_INT2_KERNEL` selects the variant, per process:

| Value | Prefill (M>1) | Decode (M=1) |
|---|---|---|
| `upcvt` (default) | up-convert, fp16 DPAS | up-convert, fp16 DPAS |
| `dpas_prefill` | int2 x int8 DPAS | up-convert, fp16 DPAS |
| `dpas` | int2 x int8 DPAS | int2 x int8 DPAS |

The best variant depends on prompt length and on the target device, so it should
be measured rather than assumed.
