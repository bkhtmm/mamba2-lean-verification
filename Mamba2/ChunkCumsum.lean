import VeriTile.Triton.DSL
import VeriTile.Triton.Semantics
import VeriTile.Triton.KernelLemmas.OffsetInjective
import VeriTile.Triton.KernelLemmas.ScatterStore

/-!
# Manually normalized cuTile `chunk_cumsum_kernel`

Source: NVIDIA/TileGym 3a871364f1fbd59ef30b1afcc5a79fc3634888ce,
`src/tilegym/ops/cutile/mamba2.py`, lines 833–854.

The leading singleton batch axis is squeezed. The token/head tile is [C,K].
The source's `permute` + BHT store is represented by a scatter to BHT offsets.
This is an explicit manual source model, NOT a verified cuTile translation.
The theorem is about VeriTile's exact-real operational semantics. Casts are
identity and log2(e) is a parameter L, so it also covers the source literal
regarded as an exact real. Full token/head tiles only: source ZERO padding
is inactive under the separately stated bounds. No machine FP/concurrency claim.
-/

namespace Mamba2.ChunkCumsum
open VeriTile.Triton
open scoped BigOperators

/-- Token coordinate and head coordinate use the actual cuTile grid axes. -/
def token (s : BlockState) (C : Nat) (i : Nat) := s.pids 0 * C + i
def head (s : BlockState) (K : Nat) (j : Nat) := s.pids 1 * K + j

def bth (s : BlockState) (T H C K : Nat) (i : TileIndex [C,K]) : Nat :=
  (s.pids 2 * T + token s C i.1.val) * H + head s K i.2.1.val

def bht (s : BlockState) (T H C K : Nat) (i : TileIndex [C,K]) : Nat :=
  (s.pids 2 * H + head s K i.2.1.val) * T + token s C i.1.val

/-- The complete three-output kernel on full tiles, in normalized coordinates. -/
def kernel (T H C K : Nat) (L : ℝ) : ComputeKernel := triton {
  chunk = tl.program_id(0)
  ht = tl.program_id(1)
  batch = tl.program_id(2)
  ts = chunk * $(C) + tl.arange(0, $(C))
  hs = ht * $(K) + tl.arange(0, $(K))
  ix = (batch * $(T) + ts[:, None]) * $(H) + hs[None, :]
  it = (batch * $(H) + hs[None, :]) * $(T) + ts[:, None]
  dt = tl.load($((⟨"dt"⟩ : RegionName)) + ix)
  a = tl.load($((⟨"a"⟩ : RegionName)) + hs)
  z = a[None, :] * dt
  g = tl.cumsum(z, axis=0) * $(L)
  tl.store($((⟨"out_bth"⟩ : RegionName)) + ix, g)
  tl.store($((⟨"out_bht"⟩ : RegionName)) + it, g)
  tl.store($((⟨"dt_bht"⟩ : RegionName)) + it, dt)
}

/-- Independent inclusive-prefix specification, with no cross-chunk carry. -/
noncomputable def spec (s : BlockState) (T H C K : Nat) (L : ℝ)
    (i : TileIndex [C,K]) : ℝ :=
  (∑ k ∈ Finset.univ.filter (fun k : Fin C => k.val ≤ i.1.val),
    s.readMem ⟨"a"⟩ (head s K i.2.1.val) *
    s.readMem ⟨"dt"⟩ (bth s T H C K (k, i.2))) * L

/-- A real-add scan has the ordinary finite prefix-sum semantics. -/
private theorem scan_prefix {α : Type*} (g : α → ℝ) (l : List α) :
    (l.map (fun k => (some (g k) : WithBot ℝ))).foldl WithBot.realAdd
      ((0 : ℝ) : WithBot ℝ) = some ((l.map g).sum) := by
  have helper (rs : List ℝ) (r : ℝ) :
      (rs.map (fun x => ((x : ℝ) : WithBot ℝ))).foldl WithBot.realAdd
        ((r : ℝ) : WithBot ℝ) = (((rs.foldl (· + ·) r : ℝ)) : WithBot ℝ) := by
    induction rs generalizing r with
    | nil => rfl
    | cons a t ih =>
      simp only [List.map_cons, List.foldl_cons, WithBot.realAdd_coe_coe]
      exact ih (r+a)
  rw [show l.map (fun k => (some (g k) : WithBot ℝ)) =
    (l.map g).map (fun x => ((x : ℝ) : WithBot ℝ)) by simp only [List.map_map, Function.comp_def]; rfl]
  rw [helper]
  congr 1
  exact List.sum_eq_foldl.symm

/-- Universal 2D token-axis scan theorem reused by the source model. -/
theorem scan_token_axis (C K : Nat) (g : TileIndex [C,K] → ℝ)
    (i : TileIndex [C,K]) :
    (Tile.scan .sum 0 .forward
      (⟨fun q => some (g q)⟩ : Tile .real [C,K])).data i =
      some (∑ k ∈ Finset.univ.filter (fun k : Fin C => k.val ≤ i.1.val),
        g (k, i.2)) := by
  change (Tile.scan .sum ⟨0, by simp⟩ .forward
      (⟨fun q => some (g q)⟩ : Tile .real [C,K])).data i = _
  rw [Tile.scan_data]
  simp only [ScanOp.eval_sum, TileShape.axisCoord, TileShape.replaceAxisCoord,
    TileShape.axisDim]
  refine Eq.trans (scan_prefix (fun k : Fin C => g (k, i.2)) _) ?_
  rw [Finset.sum_map_toList]
  rfl

/-- The BTH store offset is injective whenever the head tile fits the head stride. -/
theorem bth_injective (s : BlockState) (T H C K : Nat) (hK : K ≤ H) :
    Function.Injective (bth s T H C K) := by
  have he : bth s T H C K = fun i : TileIndex [C,K] =>
      ((s.pids 2 * T + s.pids 0 * C) * H + s.pids 1 * K) +
        i.1.val * H + i.2.1.val := by
    funext i
    dsimp [bth, token, head]
    ring
  rw [he]
  exact rowMajor2D_inj _ H hK

/-- The transposed store is injective whenever the token tile fits the token stride. -/
theorem bht_injective (s : BlockState) (T H C K : Nat) (hC : C ≤ T) :
    Function.Injective (bht s T H C K) := by
  let flip : TileIndex [C,K] → TileIndex [K,C] := fun i => (i.2.1, i.1, PUnit.unit)
  have hflip : Function.Injective flip := by
    rintro ⟨a, b, u⟩ ⟨a', b', u'⟩ he
    have ha : a = a' := congrArg (fun i => i.2.1) he
    have hb : b = b' := congrArg (fun i => i.1) he
    cases u
    cases u'
    simp [ha, hb]
  intro i j he
  apply hflip
  apply rowMajor2D_inj ((s.pids 2 * H + s.pids 1 * K) * T + s.pids 0 * C) T hC
  change _ + i.2.1.val * T + i.1.val = _ + j.2.1.val * T + j.1.val
  dsimp [bht, token, head] at he
  nlinarith only [he]

/-- Finite unmasked scatter used only to describe the resulting memory. -/
noncomputable def scatter (C K : Nat) (r : RegionName)
    (offset : TileIndex [C,K] → Nat) (value : TileIndex [C,K] → ℝ)
    (s : BlockState) : BlockState :=
  (TileShape.allIndices [C,K]).foldl
    (fun acc i => acc.writeMem r (offset i) (value i)) s

/-- Three source output stores, evaluated from the original input memory. -/
noncomputable def stores (s : BlockState) (T H C K : Nat) (L : ℝ) : BlockState :=
  scatter C K ⟨"dt_bht"⟩ (bht s T H C K)
    (fun i => s.readMem ⟨"dt"⟩ (bth s T H C K i))
    (scatter C K ⟨"out_bht"⟩ (bht s T H C K) (spec s T H C K L)
      (scatter C K ⟨"out_bth"⟩ (bth s T H C K) (spec s T H C K L) s))

/-- A scatter leaves every memory cell outside its listed region/offset pairs unchanged. -/
private theorem scatter_frame (C K : Nat) (region : RegionName)
    (offset : TileIndex [C,K] → Nat) (value : TileIndex [C,K] → ℝ)
    (r : RegionName) (o : Nat) (s : BlockState)
    (hmiss : ∀ i, ¬(region = r ∧ offset i = o)) :
    (scatter C K region offset value s).mem r o = s.mem r o := by
  unfold scatter
  induction TileShape.allIndices [C,K] generalizing s with
  | nil => rfl
  | cons i rest ih =>
      rw [List.foldl_cons, ih, BlockState.writeMem_mem]
      exact if_neg (fun h => hmiss i ⟨h.1.symm, h.2.symm⟩)

/-- The complete normalized program terminates, and its memory is exactly the
three source output scatters. Initial registers and all non-output memory are
arbitrary. This is an execution theorem in VeriTile's named-region semantics. -/
theorem execution_memory (s : BlockState) (T H C K : Nat) (L : ℝ) :
    ∃ s1, exec (kernel T H C K L) s = some s1 ∧
      s1.mem = (stores s T H C K L).mem := by
  have hg (i : TileIndex [C,K]) := scan_token_axis C K
    (fun q => s.readMem ⟨"a"⟩ (s.pids 1 * K + q.2.1.val) *
      s.readMem ⟨"dt"⟩ ((s.pids 2 * T + (s.pids 0 * C + q.1.val)) * H +
        (s.pids 1 * K + q.2.1.val))) i
  simp [exec, kernel, stepStmts, stepStmt, evalOp, Option.bind,
    Tile.bop, NumericDType.add, NumericDType.mul,
    -Tile.scan_data,
    stores, scatter, spec, bth, bht, token, head]
  repeat unfold evalOp
  simp [Tile.expandDim, Option.bind, -Tile.scan_data]
  funext r o
  apply BlockState.foldl_writeMem_mem_congr _ _ _ _ (fun _ _ => rfl) r o
  apply BlockState.foldl_writeMem_mem_congr _ _ _ _ (by
    intro i _
    exact congrArg (fun v : WithBot ℝ => WithBot.unbotD 0 (v.map (fun a => a * L))) (hg i)) r o
  apply BlockState.foldl_writeMem_mem_congr _ _ _ _ (by
    intro i _
    exact congrArg (fun v : WithBot ℝ => WithBot.unbotD 0 (v.map (fun a => a * L))) (hg i)) r o
  rfl

/-- Exactly the three output windows of this program, in the named-region model. -/
def WriteFootprint (s : BlockState) (T H C K : Nat) (r : RegionName) (o : Nat) : Prop :=
  (r = ⟨"out_bth"⟩ ∧ ∃ i, bth s T H C K i = o) ∨
  (r = ⟨"out_bht"⟩ ∧ ∃ i, bht s T H C K i = o) ∨
  (r = ⟨"dt_bht"⟩ ∧ ∃ i, bht s T H C K i = o)

private theorem stores_frame (s : BlockState) (T H C K : Nat) (L : ℝ)
    (r : RegionName) (o : Nat) (hmiss : ¬WriteFootprint s T H C K r o) :
    (stores s T H C K L).mem r o = s.mem r o := by
  unfold stores
  rw [scatter_frame C K ⟨"dt_bht"⟩ _ _ r o _ (by
    intro i hi
    exact hmiss (Or.inr (Or.inr ⟨hi.1.symm, i, hi.2⟩)))]
  rw [scatter_frame C K ⟨"out_bht"⟩ _ _ r o _ (by
    intro i hi
    exact hmiss (Or.inr (Or.inl ⟨hi.1.symm, i, hi.2⟩)))]
  exact scatter_frame C K ⟨"out_bth"⟩ _ _ r o s (by
    intro i hi
    exact hmiss (Or.inl ⟨hi.1.symm, i, hi.2⟩))

private theorem stores_bth (s : BlockState) (T H C K : Nat) (L : ℝ)
    (hK : K ≤ H) (i : TileIndex [C,K]) :
    (stores s T H C K L).readMem ⟨"out_bth"⟩ (bth s T H C K i) =
      spec s T H C K L i := by
  unfold stores scatter
  rw [BlockState.scatter_preserves_other_region ⟨"dt_bht"⟩ _ _ ⟨"out_bth"⟩ (by decide)]
  rw [BlockState.scatter_preserves_other_region ⟨"out_bht"⟩ _ _ ⟨"out_bth"⟩ (by decide)]
  exact BlockState.scatter_readback_nd s _ _ (bth_injective s T H C K hK) i

private theorem stores_bht (s : BlockState) (T H C K : Nat) (L : ℝ)
    (hC : C ≤ T) (i : TileIndex [C,K]) :
    (stores s T H C K L).readMem ⟨"out_bht"⟩ (bht s T H C K i) =
      spec s T H C K L i := by
  unfold stores scatter
  rw [BlockState.scatter_preserves_other_region ⟨"dt_bht"⟩ _ _ ⟨"out_bht"⟩ (by decide)]
  exact BlockState.scatter_readback_nd _ _ _ (bht_injective s T H C K hC) i

private theorem stores_dt (s : BlockState) (T H C K : Nat) (L : ℝ)
    (hC : C ≤ T) (i : TileIndex [C,K]) :
    (stores s T H C K L).readMem ⟨"dt_bht"⟩ (bht s T H C K i) =
      s.readMem ⟨"dt"⟩ (bth s T H C K i) := by
  unfold stores scatter
  exact BlockState.scatter_readback_nd _ _ _ (bht_injective s T H C K hC) i

/-- Complete region-model correctness of the manually normalized three-output
kernel: termination, both inclusive-prefix output layouts, exact transposed
step sizes, and preservation of every cell outside this program's output
windows. The width hypotheses ensure injective row-major offsets. Input
values and the initial register file are arbitrary. Buffer names are distinct
regions, and all arithmetic here is exact real/natural arithmetic. -/
theorem kernel_correctness (s : BlockState) (T H C K : Nat) (L : ℝ)
    (hC : C ≤ T) (hK : K ≤ H) :
    ∃ s1, exec (kernel T H C K L) s = some s1 ∧
      (∀ i, s1.readMem ⟨"out_bth"⟩ (bth s T H C K i) = spec s T H C K L i) ∧
      (∀ i, s1.readMem ⟨"out_bht"⟩ (bht s T H C K i) = spec s T H C K L i) ∧
      (∀ i, s1.readMem ⟨"dt_bht"⟩ (bht s T H C K i) =
        s.readMem ⟨"dt"⟩ (bth s T H C K i)) ∧
      (∀ r o, ¬WriteFootprint s T H C K r o → s1.mem r o = s.mem r o) := by
  obtain ⟨s1, hexec, hmem⟩ := execution_memory s T H C K L
  have hread (r : RegionName) (o : Nat) :
      s1.readMem r o = (stores s T H C K L).readMem r o := by
    simp only [BlockState.readMem, hmem]
  refine ⟨s1, hexec, ?_, ?_, ?_, ?_⟩
  · intro i
    rw [hread]
    exact stores_bth s T H C K L hK i
  · intro i
    rw [hread]
    exact stores_bht s T H C K L hC i
  · intro i
    rw [hread]
    exact stores_dt s T H C K L hC i
  · intro r o hmiss
    rw [hmem]
    exact stores_frame s T H C K L r o hmiss

end Mamba2.ChunkCumsum
