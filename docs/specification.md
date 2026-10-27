# Specification and theorem scope

## Prefix kernel

The target is `chunk_cumsum_kernel` in [NVIDIA/TileGym at the pinned revision](https://github.com/NVIDIA/TileGym/blob/3a871364f1fbd59ef30b1afcc5a79fc3634888ce/src/tilegym/ops/cutile/mamba2.py). It computes chunk-local inclusive decay prefixes and writes three buffers: `out_bth`, `out_bht`, and `dt_bht`.

For token count `T`, head count `H`, token tile width `C`, and head tile width `K`, a program has identifiers `(chunk, headTile, batch)`. Lane `(i,j)` represents

$$
t=chunk\,C+i,\qquad h=headTile\,K+j.
$$

For arbitrary exact real scale `L`, the independent prefix specification is

$$
g_{i,j}=L\sum_{k=0}^{i} a_h\,dt_{batch,\,chunk\,C+k,\,h}.
$$

The normalized model squeezes the source's leading singleton batch axis, so source scan axis 1 becomes axis 0 of a `[C,K]` tile. The transposed stores are represented as scatters to BHT scalar offsets. See [translation.md](translation.md) for the operation-by-operation correspondence.

`Mamba2.ChunkCumsum.kernel_correctness` quantifies over arbitrary initial `BlockState`, natural dimensions `T,H,C,K`, and real `L`. Under `C ≤ T` and `K ≤ H`, it proves existence of a final state such that:

1. The normalized kernel execution succeeds.
2. `out_bth` at the BTH offset equals `g` for every tile lane.
3. `out_bht` at the corresponding BHT offset equals the same `g`.
4. `dt_bht` at the BHT offset equals the original timestep at the BTH offset.
5. Every memory cell outside `WriteFootprint` is unchanged.

The width hypotheses ensure per-tile offset injectivity. Source applicability additionally requires valid launch identifiers and complete in-bounds token/head tiles. Those bounds are separate indexing theorems; the region model itself has unbounded named buffers.

`execution_memory` gives an unconditional characterization of final memory as three output scatters. `spec` is a finite-sum definition independent of kernel execution.

## Recurrence and chunk decomposition

Fix one batch and head. Let `a_t = A_head * dt_t`. State `H_n` is the state before token `n`; the output at token `t` reads `H_(t+1)`:

$$
H_{t+1}=e^{a_t}H_t+dt_t B_tX_t^\top,
\qquad Y_t=C_t^\top H_{t+1}+DX_t.
$$

Define the exclusive log prefix

$$
G_n=\sum_{i<n}a_i.
$$

The source's inclusive prefix at token `t` corresponds to `G_(t+1)`. The closed form is

$$
H_n=e^{G_n}H_0+\sum_{i<n}e^{G_n-G_{i+1}}dt_iB_iX_i^\top.
$$

The exponent uses `i+1`: the injection at token `i` receives no additional decay at that token. `scalarState_closed_form` proves this identity, and `scalarState_restart` / `scalarState_chunk_jump` establish it at arbitrary chunk offsets while retaining the actual incoming state.

`recurrentOutput_eq_masked_chunk` proves, for every row `t` in an arbitrary-length chunk,

$$
Y_t=e^{G_{t+1}}C_t^\top H_0+
\sum_{i\le t}e^{G_{t+1}-G_{i+1}}(C_t^\top B_i)dt_iX_i+DX_t.
$$

The theorem expresses the sum as a full finite matrix row with future columns masked to zero. State dimension is an arbitrary finite type; input values and initial state are arbitrary. These are exact-real identities for a fixed batch/head, not execution theorems for the source state/output kernels or their head-to-group mapping.

If every `a_t ≤ 0`, `causalWeight_bounds` bounds the retained masked decay coefficients in `[0,1]`. This condition is unnecessary for the algebraic identities. The bound does not certify finite-precision exponentiation or the source's intermediate pairwise exponentials evaluated before masking.

`ideal_exp2_scale` proves `2^(x / log 2) = exp x`. Equality of the source's rounded Python `LOG2E` constant with ideal `1 / log 2` is not asserted.

## Indexing and layout

`Mamba2/Indexing.lean` proves grid coverage and unique logical tile ownership, validity of the full-tile grid under positive/divisibility hypotheses, validity of the prefix of an active lane, and BTH/BHT transpose correspondence for arbitrary values. `Mamba2/Layout.lean` proves bounds and injectivity for contiguous natural-number scalar offsets.

The public source wrapper requires positive chunk size dividing the sequence length; the selected cumsum head block divides the head count. Logical masked-tail results do not establish tail support for the full public pipeline.

The physical lifting theorem assumes injective coordinate-to-address maps on valid addresses, including disjoint allocations. These proofs do not establish finite-width pointer safety, byte layout, runtime allocation separation, arbitrary strides, or CUDA race freedom.
