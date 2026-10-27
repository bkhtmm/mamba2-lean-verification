# Source-to-model correspondence and unproved boundaries

Target: [NVIDIA/TileGym mamba2.py at 3a871364f1fbd59ef30b1afcc5a79fc3634888ce](https://github.com/NVIDIA/TileGym/blob/3a871364f1fbd59ef30b1afcc5a79fc3634888ce/src/tilegym/ops/cutile/mamba2.py). An unmodified copy is `vendor/mamba2.py`, SHA256 `0742e95b8c84fc1d88833feacd1c86aca75e6f892192e5832bf19088a59cc811`.

## Prefix-kernel normalization

`Mamba2/ChunkCumsum.lean` manually represents a complete `chunk_cumsum_kernel` tile using exact-real operational semantics. The leading singleton batch dimension is removed, so the source token axis 1 becomes model axis 0 of a `[C,K]` tile. The model is intentionally for full token and head tiles, with contiguous scalar-element BTH/BHT linearization. The original array API supports strides; this project does not prove a general strided-memory refinement.

| Pinned source operation | Lean representation | Status |
|---|---|---|
| `ct.bid(0), ct.bid(1), ct.bid(2)` | `s.pids 0, 1, 2` for chunk, head tile, batch | Manually mapped |
| `dt` tile `(1,C,K)` at `(batch,chunk,headTile)` | `[C,K]` lanes at `((batch*T+chunk*C+i)*H+headTile*K+j)` | Bounds and logical indexing proved |
| `a` tile at `(headTile,)` | `a[headTile*K+j]` | Same head mapping |
| ZERO padding | Unmasked reads on complete in-bounds tiles | Padding inactive under stated full-tile conditions |
| `astype(float32)` and output casts | Real-valued identities | Explicit idealization |
| broadcast `a_tile * dt_tile` | `a[None,:] * dt` | Manually normalized |
| `ct.cumsum(..., axis=1)` | `tl.cumsum(..., axis=0)` after singleton removal | Inclusive real prefix-sum theorem |
| `* LOG2E` | `* L` for arbitrary real L | Proved for any L; source constant is not equated to exact `1/log 2` |
| store `out_bth` | BTH scatter | Model execution/readback/frame proof |
| `permute(d_tile,(0,2,1))`; store `out_bht` | Same scalar tile scattered to `((batch*H+h)*T+t)` | Logical transpose correspondence proved; source normalization reviewed manually |
| `permute(dt_raw,(0,2,1))`; store `dt_bht` | Raw timestep tile scattered to BHT | Same boundary |

The [official load documentation](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.load.html) defines `order` as a permutation of array axes before tiling; the [store documentation](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.store.html) defines bounded tile stores. The CPU simulator implements these conventions manually. It has not been certified against cuTile.

The region model gives named, separate memory buffers and natural-number offsets. Full-cell frame claims concern that model. `Mamba2/Layout.lean` proves contiguous scalar address bounds, and `Mamba2/Indexing.lean` proves coverage and unique ownership. These theorems do not themselves establish CUDA race freedom, byte-level memory safety, finite-width address arithmetic, runtime alias checks, or a formal concurrent whole-grid execution. A flat-memory `TraceSafe`/`FlattenOk` bridge has not been instantiated for this kernel.

## State and output stages

For a fixed batch and head, source `a_head*dt[t]` becomes arbitrary real log decay `a(t)` in `Mamba2/Algebra.lean`. State injection is `B(t,s)*dt(t)*X(t,p)`. The actual source stores state as `(head_size,state_size)`; the Lean component is indexed `(s,p)`. These transpose conventions are harmless in the mathematical formulas but remain part of manual source review.

The state jump theorem covers arbitrary start, chunk length and incoming state. The output theorem rearranges finite sums into the carried state term, masked `C*B` interaction with `dt*X`, and `D*X` skip term. These are mathematical formulas; no execution theorem for the `fwd_pass_state_kernel`, `_co_tile`, `_co_swap_tile`, persistent scheduler, host wrappers or autotuner is asserted.

`ideal_exp2_scale` proves `2^(x/log 2)=exp x`. Python computes `math.log2(math.e)` with finite precision; neither that constant nor GPU exponential approximations are certified. Likewise the nonpositive-decay theorem bounds *retained causal weights over reals*. Source code evaluates pairwise exponentials before masking. The theorem does not certify safety of those intermediate evaluations, overflow/underflow behavior, NaNs or a finite error budget.

## Evidence edges

| Edge | Evidence obtained |
|---|---|
| Pinned upstream source → normalized Lean prefix model | Manual mapping above, exact source hash, sampled CPU source simulation; no universal translation proof |
| Lean prefix model → independent prefix-sum specification | Lean execution/readback/frame theorems |
| Recurrence → chunk state and causal output formulas | Lean exact-real identities |
| Logical tile grid → coordinate ownership and full-tile bounds | Lean natural-number theorems |
| Original selected Python bodies → sequential reference | CPU differential tests, casts idealized, finite samples |
| cuTile compiler → compiled program → NVIDIA hardware | Not validated in this environment |
