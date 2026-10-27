import Mamba2
import VeriTile.Meta.StatementAudit

/-! Audit every project theorem's transitive axiom footprint, not just selected
headline names.  Lean's standard base is propext, Classical.choice, Quot.sound.
This is an axiom audit, not an independent Lean-kernel recheck. -/
open Lean Elab Command

elab "#auditMamba2" : command => do
  let env ← getEnv
  let mut count := 0
  for (name, info) in env.constants.toList do
    let userName := ((privateToUserName? name).getD name).eraseMacroScopes
    if ! (`Mamba2).isPrefixOf userName then continue
    match info with
    | .axiomInfo _ => throwError m!"Project axiom is forbidden: {name}"
    | .thmInfo _ =>
        VeriTile.Meta.auditAxioms name
        count := count + 1
    | _ => pure ()
  logInfo m!"PASS: {count} Mamba2 theorems have only standard-base axioms."

#auditMamba2
#specNonCircular Mamba2.ChunkCumsum.spec avoiding [Mamba2.ChunkCumsum.kernel]
#specNonCircular Mamba2.Exact.chunkOutput avoiding [Mamba2.Exact.recurrentOutput]
#specNonCircular Mamba2.Exact.closedState avoiding [Mamba2.Exact.scalarState]
