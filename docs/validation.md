# Validation record

The recorded verification run completed on 27 October 2026 with Lean v4.29.0, Python 3.9.6 and NumPy 2.0.2. All checked proof and script hashes are recorded in [`evidence/verification-status.json`](../evidence/verification-status.json).

## Lean checks

| Check | Result | Evidence |
|---|---|---|
| `lake build` | Passed | [build log](../evidence/lean-build.log) |
| `lake env lean Audit.lean` | Passed | [audit log](../evidence/axiom-audit.log) |
| Independent prefix and algebra specifications do not refer to the corresponding implementations | Passed | [Audit.lean](../Audit.lean) |
| Wrong scan axis in the normalized kernel body | Existing proof rejected the mutation | [diagnostic](../evidence/lean-mutation-wrong_scan_axis.log) |
| Omitted prefix scale in the normalized kernel body | Existing proof rejected the mutation | [diagnostic](../evidence/lean-mutation-omitted_scale.log) |

The axiom audit checks all project theorem declarations, including generated declarations, against Lean's standard base: `propext`, `Classical.choice`, and `Quot.sound`. This is an axiom-footprint audit, not an independent implementation of the Lean kernel. Specification independence is a structural check, not a proof that the chosen specification captures every intended requirement.

The project's Lean files compile without warnings. Some imported dependency modules emit linter warnings. Errors in mutation logs are expected negative-control results; the unmodified baseline compiles first.

A second complete check ran from a fresh project source directory without copied project oleans, reusing the pinned dependency cache. Checked-file hashes matched: [relocation record](../evidence/relocation.json). This was not a fresh network download of dependencies.

## CPU source differential checks

[`scripts/source_differential.py`](../scripts/source_differential.py) selects the original SHA256-pinned function bodies for `chunk_cumsum_kernel`, `fwd_pass_state_kernel`, `_co_tile`, and `_co_swap_tile`, removing decorators and annotations. It executes them with a manually implemented NumPy tile API and compares them against an independently sequential recurrence.

- Four sampled full-pipeline shapes, each exercising sixteen combinations of source control flags: 64 branch combinations.
- Multiple chunks, grouped heads, nonzero initial state, state/output operand swaps, alternate BTH/BHT inputs, and dual state stores.
- Two additional direct-prefix padding cases, outside the full public wrapper's accepted domain.
- Three detected source mutations: wrong scan axis, excluded causal diagonal, and wrong decay subtraction sign.

Results: [source differential record](../evidence/source-differential.json). Casts are identities and arithmetic uses float64, with absolute/relative comparison tolerances of `2e-12`. These tolerances check simulator consistency; they are not GPU error bounds. Some small source-body configurations are outside production autotuner choices.

No cuTile compiler, NVIDIA GPU or sanitizer campaign was executed. Host wrappers, autotuner selection, persistent scheduling, and backward kernels are not executed by the harness. The source-to-Lean translation and tile API are manual and are not certified by these samples. No supported-input upstream bug was identified.

## Reproduction

Use the setup commands in the [README](../README.md), then run `python scripts/check.py` with the configured Python environment. The script performs the build, audits, source comparisons and Lean mutations and writes the resulting logs and hashes under `evidence/`.
