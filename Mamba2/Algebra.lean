import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Ring

/-!
# Exact-real algebra for the Mamba-2 forward pass

This file proves identities for a mathematical recurrence.  It does not give
semantics to CUDA, cuTile, machine floating point, `ct.mma`, or a compiler.

For one batch and head, `a t` is the discretized log decay A * dt[t].
`logPrefix a (t+1)` is the *inclusive* cumsum at token t.  The stored source
cumsum is scaled by LOG2E; this file states the ideal natural-exponential
formula.  All dimensions, sequence lengths, input values, and initial states
are arbitrary.  Nonpositivity is needed only for the final decay bounds.
-/

open scoped BigOperators

noncomputable section

namespace Mamba2.Exact

/-- Ideal exact-real conversion underlying `exp2(LOG2E * x)`.
The Python `math.log2(math.e)` constant is rounded; no equality between that
machine constant and `1 / log 2` is asserted here. -/
theorem ideal_exp2_scale (x : ℝ) : Real.rpow 2 (x / Real.log 2) = Real.exp x := by
  change (2 : ℝ) ^ (x / Real.log 2) = Real.exp x
  rw [Real.rpow_def_of_pos (zero_lt_two : (0 : ℝ) < 2)]
  congr 1
  exact mul_div_cancel₀ x (ne_of_gt (Real.log_pos (one_lt_two : (1 : ℝ) < 2)))

/-- Sum of log decays before index n.  Token t uses `logPrefix a (t+1)`. -/
def logPrefix (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i ∈ Finset.range n, a i

@[simp] theorem prefix_zero (a : ℕ → ℝ) : logPrefix a 0 = 0 := by
  simp [logPrefix]

@[simp] theorem prefix_succ (a : ℕ → ℝ) (n : ℕ) :
    logPrefix a (n + 1) = logPrefix a n + a n := by
  simp [logPrefix, Finset.sum_range_succ]

/-- Scalar component of h[n+1] = exp(a[n]) h[n] + u[n].
`scalarState ... n` is the state after n updates, before token n. -/
def scalarState (a u : ℕ → ℝ) (initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | n + 1 => Real.exp (a n) * scalarState a u initial n + u n

@[simp] theorem scalarState_zero (a u : ℕ → ℝ) (initial : ℝ) :
    scalarState a u initial 0 = initial := rfl

@[simp] theorem scalarState_succ (a u : ℕ → ℝ) (initial : ℝ) (n : ℕ) :
    scalarState a u initial (n + 1) =
      Real.exp (a n) * scalarState a u initial n + u n := rfl

/-- Decay from the injection at token i through the state after n updates.
The injection has no decay at i itself, hence the index `i+1`. -/
def weight (a : ℕ → ℝ) (n i : ℕ) : ℝ :=
  Real.exp (logPrefix a n - logPrefix a (i + 1))

@[simp] theorem weight_diagonal (a : ℕ → ℝ) (n : ℕ) :
    weight a (n + 1) n = 1 := by
  simp [weight]

theorem weight_step (a : ℕ → ℝ) (n i : ℕ) :
    weight a (n + 1) i = Real.exp (a n) * weight a n i := by
  rw [weight, weight, ← Real.exp_add, prefix_succ]
  congr 1
  ring

theorem prefix_exp_step (a : ℕ → ℝ) (n : ℕ) :
    Real.exp (logPrefix a (n + 1)) = Real.exp (a n) * Real.exp (logPrefix a n) := by
  rw [prefix_succ, ← Real.exp_add]
  congr 1
  ring

/-- Closed form of the scalar state, retaining an arbitrary initial state. -/
def closedState (a u : ℕ → ℝ) (initial : ℝ) (n : ℕ) : ℝ :=
  Real.exp (logPrefix a n) * initial +
    ∑ i ∈ Finset.range n, weight a n i * u i

/-- The weighted chunk formula is exactly the sequential recurrence over ℝ. -/
theorem scalarState_closed_form (a u : ℕ → ℝ) (initial : ℝ) (n : ℕ) :
    scalarState a u initial n = closedState a u initial n := by
  induction n with
  | zero => simp [closedState]
  | succ n ih =>
    rw [scalarState_succ, ih]
    unfold closedState
    rw [Finset.sum_range_succ, weight_diagonal, one_mul, prefix_exp_step]
    rw [mul_add, Finset.mul_sum]
    simp_rw [weight_step, mul_assoc]
    ring

/-- A recurrence supplied independently of `scalarState` satisfies the same
closed form, provided its initial value and every update obey the specification. -/
theorem recurrence_closed_form (a u h : ℕ → ℝ) (initial : ℝ)
    (hzero : h 0 = initial)
    (hstep : ∀ n, h (n + 1) = Real.exp (a n) * h n + u n) (n : ℕ) :
    h n = closedState a u initial n := by
  have heq : ∀ k, h k = scalarState a u initial k := by
    intro k
    induction k with
    | zero => exact hzero
    | succ k ih => rw [hstep, scalarState_succ, ih]
  exact (heq n).trans (scalarState_closed_form a u initial n)

/-- Restarting a recurrence at any chunk boundary preserves its state. -/
theorem scalarState_restart (a u : ℕ → ℝ) (initial : ℝ) (start n : ℕ) :
    scalarState a u initial (start + n) =
      scalarState (fun i => a (start + i)) (fun i => u (start + i))
        (scalarState a u initial start) n := by
  induction n with
  | zero => simp
  | succ n ih =>
    simp only [Nat.add_succ, scalarState_succ, ih]

/-- Exact chunk jump for every start, chunk length, and incoming state.
This is the algebraic target of `_fwd_pass_state` in the cuTile source. -/
theorem scalarState_chunk_jump (a u : ℕ → ℝ) (initial : ℝ) (start n : ℕ) :
    scalarState a u initial (start + n) =
      Real.exp (logPrefix (fun i => a (start + i)) n) *
          scalarState a u initial start +
        ∑ i ∈ Finset.range n,
          weight (fun j => a (start + j)) n i * u (start + i) := by
  rw [scalarState_restart, scalarState_closed_form]
  rfl

/-- State coordinate with arbitrary state and head coordinate types.
B is indexed by token and state coordinate; X by token and head coordinate. -/
def stateCoord {S P : Type*} (a dt : ℕ → ℝ) (b : ℕ → S → ℝ)
    (x : ℕ → P → ℝ) (initial : S → P → ℝ) (s : S) (p : P) (n : ℕ) : ℝ :=
  scalarState a (fun i => b i s * dt i * x i p) (initial s p) n

/-- The coordinate model follows the specified Mamba-2 state update. -/
@[simp] theorem stateCoord_succ {S P : Type*} (a dt : ℕ → ℝ) (b : ℕ → S → ℝ)
    (x : ℕ → P → ℝ) (initial : S → P → ℝ) (s : S) (p : P) (n : ℕ) :
    stateCoord a dt b x initial s p (n + 1) =
      Real.exp (a n) * stateCoord a dt b x initial s p n + b n s * dt n * x n p :=
  rfl

/-- State-space readout at token t, including the direct D * X term. -/
def recurrentOutput {S P : Type*} [Fintype S]
    (a dt : ℕ → ℝ) (b c : ℕ → S → ℝ) (x : ℕ → P → ℝ)
    (d : ℝ) (initial : S → P → ℝ) (t : ℕ) (p : P) : ℝ :=
  (∑ s, c t s * stateCoord a dt b x initial s p (t + 1)) + d * x t p

/-- The intra-chunk C B product and causal triangular sum, plus incoming state.
This is the exact-real algebraic target of `_co_tile` and `_co_swap_tile`. -/
def chunkOutput {S P : Type*} [Fintype S]
    (a dt : ℕ → ℝ) (b c : ℕ → S → ℝ) (x : ℕ → P → ℝ)
    (d : ℝ) (initial : S → P → ℝ) (t : ℕ) (p : P) : ℝ :=
  Real.exp (logPrefix a (t + 1)) * (∑ s, c t s * initial s p) +
    (∑ i ∈ Finset.range (t + 1),
      weight a (t + 1) i * (∑ s, c t s * b i s) * dt i * x i p) +
    d * x t p

/-- The sequential state-space output equals the causal C B chunk formula.
There is no zero-initial-state assumption and no fixed numerical dimension. -/
theorem recurrentOutput_eq_chunkOutput {S P : Type*} [Fintype S]
    (a dt : ℕ → ℝ) (b c : ℕ → S → ℝ) (x : ℕ → P → ℝ)
    (d : ℝ) (initial : S → P → ℝ) (t : ℕ) (p : P) :
    recurrentOutput a dt b c x d initial t p =
      chunkOutput a dt b c x d initial t p := by
  unfold recurrentOutput
  simp_rw [stateCoord, scalarState_closed_form, closedState, mul_add]
  rw [Finset.sum_add_distrib]
  unfold chunkOutput
  apply congrArg (fun z : ℝ => z + d * x t p)
  apply congrArg₂ (· + ·)
  · rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro s hs
    ring
  · calc
      (∑ s, c t s *
          ∑ i ∈ Finset.range (t + 1),
            weight a (t + 1) i * (b i s * dt i * x i p)) =
          ∑ s, ∑ i ∈ Finset.range (t + 1),
            c t s * (weight a (t + 1) i * (b i s * dt i * x i p)) := by
        apply Finset.sum_congr rfl
        intro s hs
        rw [Finset.mul_sum]
      _ = ∑ i ∈ Finset.range (t + 1), ∑ s,
            c t s * (weight a (t + 1) i * (b i s * dt i * x i p)) :=
        Finset.sum_comm
      _ = ∑ i ∈ Finset.range (t + 1),
            weight a (t + 1) i * (∑ s, c t s * b i s) * dt i * x i p := by
        apply Finset.sum_congr rfl
        intro i hi
        rw [Finset.mul_sum, Finset.sum_mul, Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro s hs
        ring

/-- Inclusive source cumsums are nonincreasing when each log decay is ≤ 0. -/
theorem prefix_antitone (a : ℕ → ℝ) (ha : ∀ i, a i ≤ 0) :
    Antitone (logPrefix a) := by
  intro i j hij
  induction j, hij using Nat.le_induction with
  | base => exact le_rfl
  | succ j hij ih =>
    rw [prefix_succ]
    exact (add_le_of_nonpos_right (ha j)).trans ih

/-- Every causal exact-real decay lies in (0,1], with no positivity assumption
on dt itself: it is the product A*dt in a that must be nonpositive. -/
theorem causal_weight_bounds (a : ℕ → ℝ) (ha : ∀ i, a i ≤ 0)
    (n i : ℕ) (hi : i < n) : 0 < weight a n i ∧ weight a n i ≤ 1 := by
  constructor
  · exact Real.exp_pos _
  · apply Real.exp_le_one_iff.mpr
    exact sub_nonpos.mpr (prefix_antitone a ha (Nat.succ_le_of_lt hi))

/-- Entry of the explicitly masked causal decay matrix. -/
def causalWeight (a : ℕ → ℝ) (row col : ℕ) : ℝ :=
  if col ≤ row then weight a (row + 1) col else 0

@[simp] theorem causalWeight_future (a : ℕ → ℝ) (row col : ℕ)
    (h : row < col) : causalWeight a row col = 0 := by
  simp [causalWeight, Nat.not_le_of_lt h]

@[simp] theorem causalWeight_diagonal (a : ℕ → ℝ) (row : ℕ) :
    causalWeight a row row = 1 := by
  simp [causalWeight]

/-- A full finite matrix row with a causal mask equals its prefix sum.
The row may lie anywhere in the finite chunk; future columns contribute zero. -/
theorem causalWeight_sum (a v : ℕ → ℝ) (length row : ℕ) (hrow : row < length) :
    (∑ col ∈ Finset.range length, causalWeight a row col * v col) =
      ∑ col ∈ Finset.range (row + 1), weight a (row + 1) col * v col := by
  calc
    (∑ col ∈ Finset.range length, causalWeight a row col * v col) =
        ∑ col ∈ Finset.range (row + 1), causalWeight a row col * v col := by
      symm
      apply Finset.sum_subset (Finset.range_mono (Nat.succ_le_of_lt hrow))
      intro col hcol hnot
      have hnle : ¬ col ≤ row := by
        intro hle
        exact hnot (Finset.mem_range.mpr (Nat.lt_succ_of_le hle))
      simp [causalWeight, hnle]
    _ = ∑ col ∈ Finset.range (row + 1), weight a (row + 1) col * v col := by
      apply Finset.sum_congr rfl
      intro col hcol
      have hle : col ≤ row := Nat.le_of_lt_succ (Finset.mem_range.mp hcol)
      simp [causalWeight, hle]

/-- Exact-real output equivalence in the full masked-matrix form used by the
source: for any row in a chunk of arbitrary length, future columns vanish. -/
theorem recurrentOutput_eq_masked_chunk {S P : Type*} [Fintype S]
    (a dt : ℕ → ℝ) (b c : ℕ → S → ℝ) (x : ℕ → P → ℝ)
    (d : ℝ) (initial : S → P → ℝ) (length t : ℕ) (p : P) (ht : t < length) :
    recurrentOutput a dt b c x d initial t p =
      Real.exp (logPrefix a (t + 1)) * (∑ s, c t s * initial s p) +
        (∑ i ∈ Finset.range length,
          causalWeight a t i * (∑ s, c t s * b i s) * dt i * x i p) + d * x t p := by
  have hsum :
      (∑ i ∈ Finset.range length,
        causalWeight a t i * (∑ s, c t s * b i s) * dt i * x i p) =
      ∑ i ∈ Finset.range (t + 1),
        weight a (t + 1) i * (∑ s, c t s * b i s) * dt i * x i p := by
    simpa only [mul_assoc] using
      causalWeight_sum a (fun i => (∑ s, c t s * b i s) * dt i * x i p) length t ht
  rw [recurrentOutput_eq_chunkOutput]
  unfold chunkOutput
  rw [hsum]

/-- A masked causal matrix never needs a decay greater than one when a ≤ 0. -/
theorem causalWeight_bounds (a : ℕ → ℝ) (ha : ∀ i, a i ≤ 0) (row col : ℕ) :
    0 ≤ causalWeight a row col ∧ causalWeight a row col ≤ 1 := by
  unfold causalWeight
  split_ifs with h
  · obtain ⟨hp, hle⟩ := causal_weight_bounds a ha (row + 1) col (lt_of_le_of_lt h (Nat.lt_succ_self row))
    exact ⟨le_of_lt hp, hle⟩
  · exact ⟨le_rfl, zero_le_one⟩

end Mamba2.Exact
