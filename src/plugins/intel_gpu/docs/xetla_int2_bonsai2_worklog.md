# Bonsai 2 27B on the XeTLA int2 GPU path: work record

Date: 2026-09-18. Branch `feature_integrate_2bit_xetla_kernels`, commits
`0908e5c7b2` (Bonsai 2 support), `0febf672eb` (M-tiled prefill, LNL results)
and this note. Companion to [xetla_int2_build_and_run.md](xetla_int2_build_and_run.md)
section 7, which has the user-facing instructions; this file records what was
done, in order, and what was measured.

## 1. Baseline

* Fresh clone of this fork at `eaf52c90a9`, built per the BKM (OCL runtime,
  oneAPI 2026.0, XeTLA `ov-int2-integration`), 5 min on the B70 node.
* Bonsai 27B (1st gen) u2 IR, paged, 256 tokens on the Arc Pro B70:
  **44.9 tok/s** (doc: 45.0). Same tokens as documented.

## 2. Model conversion (`tools/xetla_int2/bonsai2_gguf_to_ir.py`)

Bonsai 2 27B ships only as a PQ2_0 GGUF and has exactly the Qwen3.5-27B
architecture of Bonsai 27B (configs differ only in `mtp_num_hidden_layers` and
`eos_token_id`). Instead of reconstructing a 54 GB dense checkpoint and
re-exporting with optimum-intel, the Bonsai 1 u2 IR is used as the graph
template and every weight is swapped in from the GGUF (~40 s):

* PQ2_0 blocks (2-byte fp16 scale + 32 code bytes per 128) are byte-for-byte
  the `u2 {0,1,2}` / zero-point-1 / per-128-scale layout the plugin's
  compression pattern expects, so the 401 ternary matrices are copied, not
  re-quantized. Optimum's node names carry the HF module path, which gives the
  mapping.
* llama.cpp's tiled GDN value-head order is undone on `in_proj_qkv` (V rows),
  `in_proj_z`, `in_proj_a/b`, `A_log`, `dt_bias`, `conv1d` (V channels); with
  `prism.hadamard.gdn_v_grouped` the exporter left `out_proj`'s columns
  grouped, so they are not touched (Bonsai 1's GGUF needs the column
  permutation, handled too).
* `in_proj_a/b` are dense bf16 in Bonsai 2 (u2 in Bonsai 1): the u2
  decompression subgraph is replaced by a plain fp16 constant.
* Norms are stored as `1+w` in the GGUF, which is what the exported graph's
  constant holds; `A_log` arrives as `-exp(A_log)`; the embedding is
  dequantised and inverse-rotated offline, written bf16 like the template.
* `--verify` runs the extraction on the Bonsai 1 GGUF against the Bonsai 1 IR:
  497/497 ternary matrices and the embedding bit-identical, dense tensors at
  fp16 rounding (4.9e-4 abs). This is the regression test for the mapping.

Hadamard rotation (whitepaper A.2, `prism.hadamard.*` metadata: H_1024,
normalised, sign vectors per input width 5120/6144/17408):

* Inserted as `Reshape(...,K/1024,1024) -> MatMul(H) -> Reshape(...,K)` in
  front of the 257 folded projections; one shared fp32 H constant.
* Sign flips folded into weights wherever the producer is a per-channel
  weight: the 129 RMSNorm weights (input/post-attention/final; the dense
  `in_proj_a/b` consumers are compensated by flipping their columns) and the
  64 `up_proj` row sets for `down_proj`'s input (`c -> 2-c` on the u2 codes).
  The 64 attention/GDN output projections keep an explicit `Multiply`: for
  `o_proj` the signs are not constant across the query heads that share a KV
  head, and for the GDN `out_proj` the gated norm weight is per head-dim
  (128), so neither can absorb a 6144-vector. `--explicit-signs`,
  `--no-fold-norm`, `--no-fold-up` exist for bisecting.
* Bug found on the way: the norm fold read the *template's* constant after it
  had been replaced (detached node), so it wrote Bonsai 1 norms x signs.
  Symptom was fluent-looking garbage ids; found by comparing against the
  all-explicit variant. Fixed by keeping the replacement constant.

First correct run (rotation as graph ops): **39.7 tok/s**, TTFT 382 ms.

## 3. Fused input transform in the plugin

* `FuseHadamardIntoFC` (plugin/transformations): ModelPass over
  `FullyConnectedCompressed` nodes; walks `Reshape <- FullyConnected|MatMul(H)
  <- Reshape <- [Multiply(+-1)]`, verifies the constant is H_1024 (sampled
  `(-1)^popcount(i&j)/32`), rewires the FC to the source and stores
  `xetla_hadamard_block` / `xetla_hadamard_signs` in rt_info. Registered
  after `FullyConnectedHorizontalFusion` so a merged gate/up FC absorbs the
  shared rotation once. Two things the first attempt missed:
  `ConvertMatMulToFullyConnected` leaves `Convert(Transpose(H))` (looked
  through; H is symmetric), and the GDN `out_proj` source has a dynamic last
  dim (accepted; the Reshape pins it to K).
* `plugin/ops/fully_connected.cpp` moves the rt_info onto the cldnn primitive
  (`hadamard_block`, `hadamard_signs`; in hash/==/save/load).
* XeTLA int2 FC impl: `xetla/hadamard_fwht.cpp` (sign + 1024-point FWHT, 128
  items per block, fp32 butterflies through SLM, ported from the vLLM plugin)
  runs into a per-node f16 scratch and the GEMV reads that. Signs live on the
  device as i8. Buffers are kept both in the node-id cache and on the impl
  because the stateful bench renames the lm_head FC (`result:...`), which
  misses the cache. `XETLA_REJECT` now throws if an FC carrying the transform
  is rejected, since every other impl would silently ignore it.
* 208/257 fused: 43.3 tok/s; 257/257: **44.6 tok/s**.

## 4. M-tiled prefill

The up-convert kernel ran the per-row GEMV tier for `1 < M < 128`. Ported the
vLLM plugin's M tile (WGM 8/16/32 by M, WGN 64, SGM 8, SGN 16, SGK 128, KS=LS=1):
TTFT on the 21-token prompt B70 382 -> **107 ms**, LNL 1833 -> ~440 ms,
tokens bit-identical to the GEMV tier on the same binary.
`XETLA_INT2_PREFILL_CFG=-1` restores the old tier.

## 5. Final numbers (256 tokens, `BENCH_NO_EOS=1`, photosynthesis prompt)

| | B70 paged | B70 stateful | LNL paged |
|---|---|---|---|
| Bonsai 27B | 44.9 tok/s, 107 ms | 41.3 | 8.0 tok/s, 453 ms |
| Bonsai 2 27B | **44.7 tok/s, 107 ms** | **40.9** | **7.0-7.8 tok/s, ~440 ms** |

Output (decoded with the Bonsai 2 tokenizer): *"The user wants a concise
explanation of photosynthesis in exactly or approximately 200 words. Let me
craft a clear, informative paragraph..."* followed by a correct essay. Runs
are deterministic on the B70. Fused vs graph-level rotation, paged vs
stateful, and B70 vs LNL all agree through token 135 and fork at token 136
(fp32 FWHT vs fp16 MatMul accumulation; a near-tie argmax).

## 6. Not done / open

* No BITCOS variant here (int2 only, as requested).
* The FWHT is a separate kernel in front of the GEMV, not a prologue inside
  the XeTLA mainloop; the remaining ~0.5% vs Bonsai 1 is its launch cost.
* LNL decode tiles were not re-tuned for the rotated model; the M-tile sweep
  was done on the B70 only.
* Native ZE direct-list path skips FCs that carry the transform (falls through
  to the SYCL submission).

Artifacts on this cluster: `ov_models/bonsai2-27b-u2/` (IR + embeddings),
`ov_models/bonsai27b-u2/` + `bonsai27b-fp16/` (template), GGUF under
`models/Ternary-Bonsai-2-27B-gguf/`, gguf-py in `third_party/llama.cpp-prism-v7`.
