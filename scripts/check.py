#!/usr/bin/env python3
"""Reproduce formal-model and CPU-source checks. No GPU validation is implied."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
EVIDENCE.mkdir(exist_ok=True)

def run(label, command):
    print(f"Checking {label}...", flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (EVIDENCE / f"{label}.log").write_text(result.stdout)
    if result.returncode:
        print(result.stdout[-12000:], file=sys.stderr)
        raise SystemExit(f"FAILED: {label} (see evidence/{label}.log)")
    return {"command": command, "exit_code": result.returncode, "log": f"evidence/{label}.log"}

def main():
    # Remove stale success before starting a new campaign.
    status_path = EVIDENCE / "verification-status.json"
    if status_path.exists():
        status_path.unlink()
    source = ROOT / "vendor/mamba2.py"
    expected = "0742e95b8c84fc1d88833feacd1c86aca75e6f892192e5832bf19088a59cc811"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise SystemExit("Pinned source SHA256 mismatch")
    checks = [
        run("lean-build", ["lake", "build"]),
        run("axiom-audit", ["lake", "env", "lean", "Audit.lean"]),
        run("source-differential", [sys.executable, "-W", "error", "scripts/source_differential.py",
                                   "--json", "evidence/source-differential.json"]),
    ]
    # Formal mutation checks are separate from sampling source bodies.
    checks.append(run("lean-mutations", [sys.executable, "scripts/lean_mutations.py"]))
    lean = subprocess.check_output(["lake", "env", "lean", "--version"], cwd=ROOT, text=True).strip()
    checked_files = [ROOT / "lean-toolchain", ROOT / "lakefile.toml", ROOT / "lake-manifest.json",
                     ROOT / "Mamba2.lean", ROOT / "Audit.lean", ROOT / "vendor/mamba2.py"]
    checked_files += sorted((ROOT / "Mamba2").glob("*.lean")) + sorted((ROOT / "scripts").glob("*.py"))
    digests = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in checked_files}
    result = {"status": "passed", "checked_files_sha256": digests, "evidence_kind": "Lean model proofs plus idealized CPU source tests",
              "date": "2026-10-27", "lean": lean, "python": platform.python_version(),
              "platform": platform.platform(), "source_sha256": expected, "checks": checks,
              "gpu_execution": False, "independent_lean_kernel_recheck": False,
              "warning": "No proof of cuTile translation, machine floating-point accuracy, compiler correctness or hardware race freedom."}
    status_path.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: Lean build, axiom/noncircularity audit, CPU source checks and Lean mutations.")
    print("See evidence/verification-status.json. GPU execution: not run.")

if __name__ == "__main__":
    main()
