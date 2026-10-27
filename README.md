# Mamba-2 verification in Lean

Lean proofs for Mamba-2's chunk decomposition and a manually normalized model of TileGym's cuTile `chunk_cumsum_kernel`.

The prefix-kernel result covers exact-real execution, all three output buffers, and preservation of memory outside the output footprints. The algebra results relate the sequential recurrence to the chunk state jump and masked causal output formula. The cuTile-to-Lean correspondence is manual; floating-point arithmetic, compiled GPU execution and hardware concurrency are outside the proofs.

## Main results

| Result | Lean entry point |
|---|---|
| Prefix model terminates, writes both decay layouts and transposed timesteps, and preserves other memory | [`Mamba2.ChunkCumsum.kernel_correctness`](Mamba2/ChunkCumsum.lean) |
| Complete final-memory characterization of the prefix model | [`Mamba2.ChunkCumsum.execution_memory`](Mamba2/ChunkCumsum.lean) |
| Sequential recurrence equals the chunk jump for arbitrary incoming state | [`Mamba2.Exact.scalarState_chunk_jump`](Mamba2/Algebra.lean) |
| Recurrent readout equals the masked causal output formula | [`Mamba2.Exact.recurrentOutput_eq_masked_chunk`](Mamba2/Algebra.lean) |
| Logical tile coverage, unique ownership and transpose correspondence | [`Mamba2/Indexing.lean`](Mamba2/Indexing.lean) |
| Bounds and injectivity of contiguous natural-number offsets | [`Mamba2/Layout.lean`](Mamba2/Layout.lean) |

See the [specification](docs/specification.md), [source-to-model correspondence](docs/translation.md), and [validation record](docs/validation.md).

## Source and dependencies

- Target: [NVIDIA/TileGym `mamba2.py`](https://github.com/NVIDIA/TileGym/blob/3a871364f1fbd59ef30b1afcc5a79fc3634888ce/src/tilegym/ops/cutile/mamba2.py), commit `3a871364f1fbd59ef30b1afcc5a79fc3634888ce`. The unmodified module is included in [`vendor/mamba2.py`](vendor/mamba2.py).
- Lean `v4.29.0` and Mathlib `v4.29.0`. All dependency revisions are pinned in [`lake-manifest.json`](lake-manifest.json).

## Reproduce

Install Git, Python and [elan](https://github.com/leanprover/elan), then run from the repository root:

```sh
python3 -m venv work/venv
work/venv/bin/python -m pip install -r requirements.txt
lake exe cache get Mathlib.Analysis.SpecialFunctions.Exp Mathlib.Analysis.SpecialFunctions.Pow.Real Mathlib.Tactic
work/venv/bin/python scripts/check.py
```

The recorded run used Python 3.9.6 and NumPy 2.0.2. Lake obtains the pinned Lean version and dependencies. The initial build needs network access and several GB of storage.

The check script builds the Lean project, audits theorem axiom footprints and specification independence, runs idealized CPU source comparisons, and checks two deliberately incorrect Lean kernel variants. Results and checked-file hashes are in [`evidence/verification-status.json`](evidence/verification-status.json).

## Verification boundary

The prefix theorem concerns a manually normalized full tile under exact-real, named-region semantics. It does not prove the normalization itself, instantiate a flat-memory or concurrent CUDA execution model, or cover arbitrary tensor strides. The state and output results are mathematical identities; operational correctness of their complete source kernels is not proved.

CPU checks execute selected original Python function bodies using a manually implemented tile API, identity casts and float64 arithmetic. They are sampled differential tests, not GPU executions or floating-point error bounds. Host wrappers, autotuner selection, persistent scheduling and backward kernels are not verified. No supported-input upstream bug was identified.
