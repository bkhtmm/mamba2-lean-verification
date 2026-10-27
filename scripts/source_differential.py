#!/usr/bin/env python3
"""CPU source simulation and differential checks, NOT GPU/IEEE verification.

Execute selected function bodies parsed from the pinned upstream mamba2.py.
Imports, decorators, annotations and wrappers are not executed. NumPy supplies
an idealized tile API: casts are identity, MMA is float64 matrix multiply, and
loads/stores follow a manually implemented tile-index/permutation convention.
An independently sequential recurrence supplies the reference. Numerical
closeness here checks this simulator; it supplies no GPU error bound.

Requires Python 3 and NumPy (tested with NumPy 2.0.2).
    python3 scripts/source_differential.py --json work/source-differential.json
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "0742e95b8c84fc1d88833feacd1c86aca75e6f892192e5832bf19088a59cc811"
SOURCE_REVISION = "3a871364f1fbd59ef30b1afcc5a79fc3634888ce"
FUNCTIONS = (
    "chunk_cumsum_kernel",
    "fwd_pass_state_kernel",
    "_co_tile",
    "_co_swap_tile",
)


class Tile(np.ndarray):
    """NumPy tile whose source dtype casts deliberately preserve every value."""

    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def astype(self, dtype, *args, **kwargs):
        # Deliberate abstraction: neither FP32/BF16 nor TF32 rounding occurs.
        return self.view(Tile)


class TileAPI:
    PaddingMode = SimpleNamespace(ZERO="zero")
    float32 = "idealized_float32"
    tfloat32 = "idealized_tfloat32"
    int32 = np.int32

    def __init__(self):
        self.program = (0, 0, 0)
        self.loads = 0
        self.stores = 0

    def bid(self, axis):
        return self.program[axis]

    @staticmethod
    def cdiv(x, y):
        return (x + y - 1) // y

    @staticmethod
    def _view(array, order):
        return np.transpose(array, order) if order is not None else array

    def load(self, array, *, index, shape, order=None, padding_mode=None, **hints):
        self.loads += 1
        view = self._view(array, order)
        if len(shape) != view.ndim or len(index) != view.ndim:
            raise AssertionError("simulator rank mismatch")
        result = np.zeros(shape, dtype=np.float64)
        for lane in np.ndindex(shape):
            address = tuple(index[d] * shape[d] + lane[d] for d in range(view.ndim))
            if all(0 <= address[d] < view.shape[d] for d in range(view.ndim)):
                result[lane] = view[address]
            elif padding_mode != self.PaddingMode.ZERO:
                raise AssertionError(f"unmodeled out-of-bounds load {address}, {view.shape}")
        return Tile(result)

    def store(self, array, *, index, tile, order=None, **hints):
        self.stores += 1
        view = self._view(array, order)
        tile = np.asarray(tile)
        if tile.ndim != view.ndim or len(index) != view.ndim:
            raise AssertionError("simulator store rank mismatch")
        for lane in np.ndindex(tile.shape):
            address = tuple(index[d] * tile.shape[d] + lane[d] for d in range(view.ndim))
            # cuTile tile stores discard out-of-bounds lanes in this model.
            if all(0 <= address[d] < view.shape[d] for d in range(view.ndim)):
                view[address] = tile[lane]

    @staticmethod
    def full(shape, value, dtype=None):
        return Tile(np.full(shape, value, dtype=np.float64))

    @staticmethod
    def cumsum(tile, axis):
        return Tile(np.cumsum(np.asarray(tile), axis=axis))

    @staticmethod
    def permute(tile, axes):
        return Tile(np.transpose(tile, axes))

    @staticmethod
    def exp2(tile):
        return Tile(np.exp2(tile))

    @staticmethod
    def mma(a, b, accumulator):
        a, b = np.asarray(a), np.asarray(b)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise AssertionError("simulator MMA shape mismatch")
        # Spell out the finite contraction so the simulator does not depend on
        # an installed BLAS backend or its accumulation/exception conventions.
        return Tile(np.sum(a[:, :, None] * b[None, :, :], axis=1) + np.asarray(accumulator))

    @staticmethod
    def extract(tile, *, index, shape):
        slices = tuple(slice(start, start + width) for start, width in zip(index, shape))
        return Tile(tile[slices])

    @staticmethod
    def arange(size, dtype=None):
        return Tile(np.arange(size, dtype=dtype))

    @staticmethod
    def where(condition, yes, no):
        return Tile(np.where(condition, yes, no))


def selected_module(source):
    parsed = ast.parse(source)
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in functions} != set(FUNCTIONS):
        raise AssertionError("pinned source function inventory changed")
    for node in functions:
        node.decorator_list = []
        node.returns = None
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            arg.annotation = None
    return ast.Module(body=functions, type_ignores=[])


class Mutation(ast.NodeTransformer):
    """Known wrong variants are negative controls, not upstream bug reports."""

    def __init__(self, kind):
        self.kind = kind
        self.changes = 0

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.kind == "strict_causal_mask" and len(node.ops) == 1 and isinstance(node.ops[0], ast.GtE):
            node.ops[0] = ast.Gt()
            self.changes += 1
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if self.kind == "wrong_scan_axis" and isinstance(node.func, ast.Attribute) and node.func.attr == "cumsum":
            for keyword in node.keywords:
                if keyword.arg == "axis":
                    keyword.value = ast.Constant(value=2)
                    self.changes += 1
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if (self.kind == "wrong_decay_sign" and isinstance(node.op, ast.Sub)
                and isinstance(node.left, ast.Name) and node.left.id == "d_last"
                and isinstance(node.right, ast.Name) and node.right.id == "d_tile"):
            node.op = ast.Add()
            self.changes += 1
        return node


def instantiate(module, mutation=None):
    module = copy.deepcopy(module)
    changes = 0
    if mutation:
        transform = Mutation(mutation)
        module = transform.visit(module)
        changes = transform.changes
        expected = {"wrong_scan_axis": 1, "strict_causal_mask": 2, "wrong_decay_sign": 2}[mutation]
        if changes != expected:
            raise AssertionError(f"mutation target drift: {mutation}: {changes} != {expected}")
    api = TileAPI()
    scope = {"ct": api, "LOG2E": math.log2(math.e)}
    exec(compile(ast.fix_missing_locations(module), "<pinned-upstream-function-bodies>", "exec"), scope)
    return api, scope, changes


def check_close(actual, expected, label):
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise AssertionError(f"{label}: nonfinite or unwritten values")
    error = float(np.max(np.abs(actual - expected))) if np.size(actual) else 0.0
    if not np.allclose(actual, expected, atol=2e-12, rtol=2e-12):
        raise AssertionError(f"{label}: CPU float64 differential check failed, max_abs={error:.6g}")
    return error


def source_prefix(api, scope, a, dt, chunk_size, block_h):
    batch_size, num_tokens, num_heads = dt.shape
    bth = np.full(dt.shape, np.nan)
    bht = np.full((batch_size, num_heads, num_tokens), np.nan)
    dt_bht = np.full(bht.shape, np.nan)
    for program in itertools.product(range(api.cdiv(num_tokens, chunk_size)),
                                     range(api.cdiv(num_heads, block_h)), range(batch_size)):
        api.program = program
        scope["chunk_cumsum_kernel"](a, dt, bth, bht, dt_bht, chunk_size, block_h)
    return bth, bht, dt_bht


def prefix_reference(a, dt, chunk_size):
    answer = np.empty_like(dt)
    for batch, token, head in np.ndindex(dt.shape):
        start = token // chunk_size * chunk_size
        answer[batch, token, head] = sum(float(a[head]) * float(dt[batch, s, head])
                                                 for s in range(start, token + 1)) * math.log2(math.e)
    return answer


def reference_recurrence(a, b, c, d, dt, x, initial, chunk_size):
    batch_size, num_tokens, num_heads, head_size = x.shape
    groups = b.shape[2]
    state = initial.copy()
    out = np.empty_like(x)
    entries = []
    for token in range(num_tokens):
        if token % chunk_size == 0:
            entries.append(state.copy())
        for batch in range(batch_size):
            for head in range(num_heads):
                group = head // (num_heads // groups)
                step = float(dt[batch, token, head])
                state[batch, head] = (math.exp(float(a[head]) * step) * state[batch, head]
                                      + step * np.outer(x[batch, token, head], b[batch, token, group]))
                out[batch, token, head] = state[batch, head] @ c[batch, token, group] + d[head] * x[batch, token, head]
    return out, state, np.stack(entries, axis=2)


def pipeline_case(module, seed, shape, mutation=None):
    batch_size, chunks, chunk_size, num_heads, groups, head_size, state_size, block_head, block_state = shape
    num_tokens = chunks * chunk_size
    rng = np.random.default_rng(seed)
    a = -rng.uniform(0.05, 0.8, num_heads)
    dt = rng.uniform(0.01, 0.2, (batch_size, num_tokens, num_heads))
    x = rng.normal(0, 0.3, (batch_size, num_tokens, num_heads, head_size))
    b = rng.normal(0, 0.3, (batch_size, num_tokens, groups, state_size))
    c = rng.normal(0, 0.3, b.shape)
    d = rng.normal(0, 0.2, num_heads)
    initial = rng.normal(0, 0.3, (batch_size, num_heads, head_size, state_size))
    expected_out, expected_final, expected_entries = reference_recurrence(a, b, c, d, dt, x, initial, chunk_size)
    api, scope, changes = instantiate(module, mutation)
    decay_bth, decay_bht, dt_bht = source_prefix(api, scope, a, dt, chunk_size, 4)
    prefix = prefix_reference(a, dt, chunk_size)
    errors = [check_close(decay_bth, prefix, "prefix original layout"),
              check_close(decay_bht, prefix.transpose(0, 2, 1), "prefix transposed layout"),
              check_close(dt_bht, dt.transpose(0, 2, 1), "timestep transposed layout")]
    variants = 0
    # Exercise all source control-flow combinations, including the BTH branch.
    for swap, bht, dual, output_swap in itertools.product((False, True), repeat=4):
        hidden_shape = (batch_size, num_heads, chunks, head_size, state_size)
        hiddens = np.full(hidden_shape, np.nan)
        hid_narrow = np.full(hidden_shape, np.nan)
        final = np.full(initial.shape, np.nan)
        for program in itertools.product(range(head_size // block_head), range(state_size // block_state),
                                         range(batch_size * num_heads)):
            api.program = program
            scope["fwd_pass_state_kernel"](
                x, b, dt_bht if bht else dt, decay_bht if bht else decay_bth,
                initial, hiddens, hid_narrow, final, chunk_size, block_state, block_head,
                num_heads // groups, 1, 1, 1, swap, bht, dual)
        errors.extend((check_close(final, expected_final, "final state"),
                       check_close(hiddens, expected_entries, "chunk-entry states")))
        if dual:
            errors.append(check_close(hid_narrow, expected_entries, "dual state output"))
        elif not np.all(np.isnan(hid_narrow)):
            raise AssertionError("DUAL=False unexpectedly wrote auxiliary buffer")
        out = np.full(x.shape, np.nan)
        helper = scope["_co_swap_tile" if output_swap else "_co_tile"]
        for batch, head, chunk, head_tile in itertools.product(range(batch_size), range(num_heads),
                                                              range(chunks), range(head_size // block_head)):
            args = (batch, head, head // (num_heads // groups), chunk, head_tile,
                    x, b, c, d, dt_bht, decay_bht, hiddens, out, chunk_size, block_state, block_head, state_size)
            helper(*args, 1, 1) if output_swap else helper(*args, 1)
        errors.append(check_close(out, expected_out, "causal token outputs"))
        variants += 1
    return {"seed": seed, "shape": list(shape), "source_branch_combinations": variants,
            "max_abs_float64_difference": max(errors), "loads": api.loads, "stores": api.stores,
            "mutation_sites": changes}


def run(source_path):
    source_bytes = source_path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != SOURCE_SHA256:
        raise AssertionError(f"upstream source hash mismatch: {digest}")
    module = selected_module(source_bytes.decode())
    # Shape: batch, chunks, chunk_size, heads, groups, head_size, state_size,
    # head_tile_width, state_tile_width. Small sizes keep CPU tests practical.
    # Test source bodies directly; these are not all selected by the autotuner.
    shapes = [
        (1, 3, 4, 4, 1, 4, 16, 4, 16),
        (2, 2, 8, 8, 2, 8, 32, 4, 16),
        (1, 2, 4, 4, 4, 8, 32, 4, 16),
        (1, 1, 128, 4, 2, 4, 16, 4, 16),
    ]
    cases = [pipeline_case(module, seed, shape) for seed, shape in enumerate(shapes, start=101)]
    # Extra direct-prefix checks exercise zero padding and partial head tiles.
    padding_checks = []
    for seed, (batch, tokens, heads, chunk_size, block_h) in enumerate(
            ((1, 10, 5, 4, 4), (2, 3, 3, 8, 4)), start=201):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=heads)
        dt = rng.normal(size=(batch, tokens, heads))
        api, scope, _ = instantiate(module)
        outputs = source_prefix(api, scope, a, dt, chunk_size, block_h)
        expected = prefix_reference(a, dt, chunk_size)
        errors = [check_close(outputs[0], expected, "padded prefix"),
                  check_close(outputs[1], expected.transpose(0, 2, 1), "padded transposed prefix"),
                  check_close(outputs[2], dt.transpose(0, 2, 1), "padded timestep transpose")]
        padding_checks.append({"seed": seed, "shape": [batch, tokens, heads, chunk_size, block_h],
                               "scope": "direct prefix kernel only; not a full public-wrapper configuration",
                               "max_abs_float64_difference": max(errors)})
    negative_controls = []
    for mutation in ("wrong_scan_axis", "strict_causal_mask", "wrong_decay_sign"):
        try:
            pipeline_case(module, 101, shapes[0], mutation)
        except AssertionError as error:
            # Distinguish an expected numerical mismatch from emulator failure.
            if "CPU float64 differential check failed" not in str(error):
                raise
            negative_controls.append({"mutation": mutation, "detected": True, "reason": str(error)})
        else:
            raise AssertionError(f"negative control not detected: {mutation}")
    return {
        "status": "passed",
        "evidence_kind": "CPU simulation and differential testing of selected pinned Python source bodies",
        "source_revision": SOURCE_REVISION,
        "source_sha256": digest,
        "executed_functions": list(FUNCTIONS),
        "numpy_version": np.__version__,
        "semantics": "identity casts, float64 arithmetic, manually modeled tile load/store/order, no concurrency",
        "comparison": {"atol": 2e-12, "rtol": 2e-12,
                       "meaning": "simulator consistency check only, not a GPU numerical error guarantee"},
        "pipeline_cases": cases,
        "direct_prefix_padding_cases": padding_checks,
        "negative_controls": negative_controls,
        "total_source_branch_combinations": sum(c["source_branch_combinations"] for c in cases),
        "limitations": [
            "No CUDA/cuTile compiler or GPU execution; no NVBit or RESOLVE execution.",
            "No IEEE/BF16/TF32 arithmetic semantics or error-bound proof; casts are identity.",
            "Tile API and tile permutation conventions were implemented manually.",
            "Sampling cannot prove source-to-Lean translation equivalence.",
            "Host wrappers, autotuner selection, persistent output scheduling and backward kernels are not executed.",
            "Some small source-body configurations extend beyond autotuner-selected configurations.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "vendor/mamba2.py")
    parser.add_argument("--json", type=Path, help="save the complete machine-readable result")
    args = parser.parse_args()
    result = run(args.source)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: {len(result['pipeline_cases'])} source pipeline cases, "
          f"{result['total_source_branch_combinations']} branch combinations, "
          f"{len(result['direct_prefix_padding_cases'])} direct prefix padding cases; "
          f"{len(result['negative_controls'])} negative controls detected.")
    print("CPU simulation only. Identity casts; no GPU execution, IEEE proof or GPU error bound.")


if __name__ == "__main__":
    main()
