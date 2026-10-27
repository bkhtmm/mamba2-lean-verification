#!/usr/bin/env python3
"""Check that two wrong, well-typed kernel bodies break the existing Lean proof.

This is a diagnostic guard against a disconnected specification, not a general
proof of translation fidelity. Originals and theorem statements are unchanged.
"""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run():
    source = (ROOT / "Mamba2/ChunkCumsum.lean").read_text()
    start = source.index("def kernel ")
    body_start = source.index("triton {", start)
    body_end = source.index("\n}", body_start) + 2
    before, body, after = source[:body_start], source[body_start:body_end], source[body_end:]
    mutations = {
        "wrong_scan_axis": ("tl.cumsum(z, axis=0)", "tl.cumsum(z, axis=1)"),
        "omitted_scale": ("tl.cumsum(z, axis=0) * $(L)", "tl.cumsum(z, axis=0)"),
    }
    work = ROOT / "work/lean-mutations"
    work.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / "evidence"
    records = []
    # Baseline first so an environment failure cannot masquerade as detection.
    baseline = subprocess.run(["lake", "env", "lean", "Mamba2/ChunkCumsum.lean"],
                              cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if baseline.returncode:
        raise RuntimeError("Baseline does not compile:\n" + baseline.stdout)
    for name, (old, new) in mutations.items():
        if body.count(old) != 1:
            raise AssertionError(f"Mutation target drift for {name}")
        path = work / f"{name}.lean"
        path.write_text(before + body.replace(old, new, 1) + after)
        result = subprocess.run(["lake", "env", "lean", str(path.relative_to(ROOT))],
                                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            raise AssertionError(f"Wrong kernel still satisfies proof: {name}")
        if re.search(r"unexpected token|unknown identifier|unknown constant|object file.*does not exist", result.stdout, re.I):
            raise AssertionError(f"Invalid control/environment failure for {name}:\n{result.stdout}")
        if not re.search(r"unsolved goals|type mismatch|Tactic.*failed|tactic.*failed|did not find instance", result.stdout, re.I):
            raise AssertionError(f"Inspect unclassified Lean failure for {name}:\n{result.stdout}")
        (evidence / f"lean-mutation-{name}.log").write_text(result.stdout)
        records.append({"mutation": name, "detected": True, "lean_exit_code": result.returncode,
                        "change": {"from": old, "to": new}, "log": f"evidence/lean-mutation-{name}.log"})
    report = {"status": "passed", "baseline_exit_code": baseline.returncode, "negative_controls": records,
              "meaning": "Existing proof scripts fail after changing only the kernel body; no hardware or translation proof."}
    (evidence / "lean-mutations.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: baseline compiles; wrong scan axis and omitted scale rejected by Lean proof checks.")

if __name__ == "__main__":
    run()
