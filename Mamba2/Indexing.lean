import Std

/-!
# Logical indexing of TileGym's Mamba-2 `chunk_cumsum_kernel`

Source correspondence (TileGym commit `3a871364f1fbd59ef30b1afcc5a79fc3634888ce`):
`tilegym/ops/cutile/mamba2.py`, `chunk_cumsum_kernel` and `_chunk_cumsum`.
The launch identifiers are `(chunk_idx, head_tile_idx, batch)` and the input/
BTH tile shape is `(1, chunk_size, BLOCK_H)`. The BHT stores exchange the
last two axes. This module verifies the induced *mathematical* indices.

Theorems quantify over natural-number indices. They do not model GPU execution,
finite-width pointer arithmetic, physical strides, aliasing, or CUDA stores.
`Buffer` names designate separate abstract allocations. To lift the results to
physical memory, each allocation's coordinate-to-address map must be injective
on its valid domain and the allocations must not overlap.

The public forward/backward wrapper requires complete token chunks, and the
cumsum autotuner chooses a head block dividing the head count. The masked-grid
results below additionally characterize valid logical positions for arbitrary
tails; they do not establish that the public Mamba pipeline accepts tails.
-/

namespace Mamba2.Indexing

/-- Natural-number version of CUDA's positive-divisor ceil division. -/
def ceilDiv (n size : Nat) : Nat := (n + size - 1) / size

/-- A tile index and a local lane determine one global axis coordinate. -/
def tileIndex (size tile lane : Nat) : Nat := size * tile + lane

theorem tileIndex_source_order (size tile lane : Nat) :
    tileIndex size tile lane = tile * size + lane := by
  simp [tileIndex, Nat.mul_comm]

/-- Every global index has a canonical quotient/remainder owner. -/
theorem tileIndex_reconstruct (size i : Nat) :
    tileIndex size (i / size) (i % size) = i := by
  exact Nat.div_add_mod i size

theorem tileIndex_owner (size tile lane : Nat) (hs : 0 < size)
    (hl : lane < size) :
    tileIndex size tile lane / size = tile ∧
    tileIndex size tile lane % size = lane := by
  simp [tileIndex, Nat.mul_add_div hs, Nat.div_eq_of_lt hl,
    Nat.mod_eq_of_lt hl]

/-- Different tile/lane pairs cannot target the same coordinate. -/
theorem tileIndex_injective (size tile₁ tile₂ lane₁ lane₂ : Nat)
    (hs : 0 < size) (hl₁ : lane₁ < size) (hl₂ : lane₂ < size)
    (he : tileIndex size tile₁ lane₁ = tileIndex size tile₂ lane₂) :
    tile₁ = tile₂ ∧ lane₁ = lane₂ := by
  have h₁ := tileIndex_owner size tile₁ lane₁ hs hl₁
  have h₂ := tileIndex_owner size tile₂ lane₂ hs hl₂
  constructor
  · exact h₁.1.symm.trans ((congrArg (· / size) he).trans h₂.1)
  · exact h₁.2.symm.trans ((congrArg (· % size) he).trans h₂.2)

theorem tileIndex_lt (size count tile lane : Nat)
    (ht : tile < count) (hl : lane < size) :
    tileIndex size tile lane < size * count := by
  calc
    size * tile + lane < size * tile + size := Nat.add_lt_add_left hl _
    _ = size * (tile + 1) := by simp [Nat.mul_add]
    _ ≤ size * count := Nat.mul_le_mul_left size (by omega)

/-- A ceil-div launch has enough tiles for every valid logical coordinate. -/
theorem le_ceilDiv_mul (n size : Nat) (hs : 0 < size) :
    n ≤ size * ceilDiv n size := by
  have h := Nat.div_add_mod (n + size - 1) size
  have hr := Nat.mod_lt (n + size - 1) hs
  unfold ceilDiv
  omega

theorem quotient_lt_ceilDiv (n size i : Nat) (hs : 0 < size) (hi : i < n) :
    i / size < ceilDiv n size := by
  exact Nat.div_lt_of_lt_mul (Nat.lt_of_lt_of_le hi (le_ceilDiv_mul n size hs))

/-- On complete tile dimensions the actual ceil-div launch is exact. -/
theorem ceilDiv_full (size count : Nat) (hs : 0 < size) :
    ceilDiv (size * count) size = count := by
  have hsub : size - 1 < size := by omega
  have he : size * count + size - 1 = size * count + (size - 1) := by omega
  simp [ceilDiv, he, Nat.mul_add_div hs, Nat.div_eq_of_lt hsub]

/-- A hardware program and a local cumsum lane (no GPU thread interpretation). -/
structure TileLane where
  batch : Nat
  chunk : Nat
  headTile : Nat
  tokenLane : Nat
  headLane : Nat
  deriving DecidableEq

/-- Coordinates in physical tensor-axis order, without a physical address map. -/
structure Coord where
  first : Nat
  second : Nat
  third : Nat
  deriving DecidableEq

/-- Launch and local-lane bounds of `_chunk_cumsum`'s BTH input tile. -/
def GridLane (batches tokens heads chunkSize blockH : Nat) (p : TileLane) : Prop :=
  p.batch < batches ∧
  p.chunk < ceilDiv tokens chunkSize ∧
  p.headTile < ceilDiv heads blockH ∧
  p.tokenLane < chunkSize ∧ p.headLane < blockH

/-- Global coordinate written by `ct.store(out_bth, (batch, chunk, headTile))`. -/
def bthCoord (chunkSize blockH : Nat) (p : TileLane) : Coord :=
  ⟨p.batch, tileIndex chunkSize p.chunk p.tokenLane,
    tileIndex blockH p.headTile p.headLane⟩

/-- The same scalar after `ct.permute(tile, (0, 2, 1))` and the BHT store. -/
def bhtCoord (chunkSize blockH : Nat) (p : TileLane) : Coord :=
  ⟨p.batch, tileIndex blockH p.headTile p.headLane,
    tileIndex chunkSize p.chunk p.tokenLane⟩

/-- Valid coordinate (one per logical tensor element) within a possibly padded grid. -/
def ActiveLane (batches tokens heads chunkSize blockH : Nat) (p : TileLane) : Prop :=
  GridLane batches tokens heads chunkSize blockH p ∧
  tileIndex chunkSize p.chunk p.tokenLane < tokens ∧
  tileIndex blockH p.headTile p.headLane < heads

/-- Every logical BTH scalar has a grid owner, for arbitrary tail sizes. -/
theorem grid_coverage (batches tokens heads chunkSize blockH b t h : Nat)
    (hc : 0 < chunkSize) (hh : 0 < blockH)
    (hb : b < batches) (ht : t < tokens) (hhead : h < heads) :
    ∃ p, ActiveLane batches tokens heads chunkSize blockH p ∧
      bthCoord chunkSize blockH p = ⟨b, t, h⟩ := by
  let p : TileLane := ⟨b, t / chunkSize, h / blockH, t % chunkSize, h % blockH⟩
  have hct := quotient_lt_ceilDiv tokens chunkSize t hc ht
  have hht := quotient_lt_ceilDiv heads blockH h hh hhead
  have hcl := Nat.mod_lt t hc
  have hhl := Nat.mod_lt h hh
  have htc := tileIndex_reconstruct chunkSize t
  have hhc := tileIndex_reconstruct blockH h
  refine ⟨p, ?_, ?_⟩
  · exact ⟨⟨hb, hct, hht, hcl, hhl⟩, htc ▸ ht, hhc ▸ hhead⟩
  · change Coord.mk b (tileIndex chunkSize (t / chunkSize) (t % chunkSize))
      (tileIndex blockH (h / blockH) (h % blockH)) = ⟨b, t, h⟩
    rw [htc, hhc]

/-- No two grid programs/local lanes write the same logical BTH scalar. -/
theorem bthCoord_injective (batches tokens heads chunkSize blockH : Nat)
    (hc : 0 < chunkSize) (hh : 0 < blockH) (p q : TileLane)
    (hp : GridLane batches tokens heads chunkSize blockH p)
    (hq : GridLane batches tokens heads chunkSize blockH q)
    (he : bthCoord chunkSize blockH p = bthCoord chunkSize blockH q) : p = q := by
  have hb : p.batch = q.batch := congrArg Coord.first he
  have ht := tileIndex_injective chunkSize p.chunk q.chunk p.tokenLane q.tokenLane
    hc hp.2.2.2.1 hq.2.2.2.1 (congrArg Coord.second he)
  have hh' := tileIndex_injective blockH p.headTile q.headTile p.headLane q.headLane
    hh hp.2.2.2.2 hq.2.2.2.2 (congrArg Coord.third he)
  cases p
  cases q
  simp_all

/-- The bounds/coverage/ownership theorem for a masked logical cumsum store. -/
theorem grid_coverage_unique (batches tokens heads chunkSize blockH b t h : Nat)
    (hc : 0 < chunkSize) (hh : 0 < blockH)
    (hb : b < batches) (ht : t < tokens) (hhead : h < heads) :
    ∃ p, (ActiveLane batches tokens heads chunkSize blockH p ∧
      bthCoord chunkSize blockH p = ⟨b, t, h⟩) ∧
      ∀ q, (ActiveLane batches tokens heads chunkSize blockH q ∧
        bthCoord chunkSize blockH q = ⟨b, t, h⟩) → q = p := by
  obtain ⟨p, hp, he⟩ := grid_coverage batches tokens heads chunkSize blockH b t h
    hc hh hb ht hhead
  refine ⟨p, ⟨hp, he⟩, ?_⟩
  intro q hq
  exact bthCoord_injective batches tokens heads chunkSize blockH hc hh q p
    hq.1.1 hp.1 (hq.2.trans he.symm)

/-- Complete token/head tiles have no out-of-bounds logical lanes. This matches
public wrapper validation and the head-block autotuning filter. -/
theorem full_grid_active (batches chunks headTiles chunkSize blockH : Nat)
    (hc : 0 < chunkSize) (hh : 0 < blockH) (p : TileLane)
    (hp : GridLane batches (chunkSize * chunks) (blockH * headTiles) chunkSize blockH p) :
    ActiveLane batches (chunkSize * chunks) (blockH * headTiles) chunkSize blockH p := by
  refine ⟨hp, ?_, ?_⟩
  · exact tileIndex_lt chunkSize chunks p.chunk p.tokenLane
      ((ceilDiv_full chunkSize chunks hc) ▸ hp.2.1) hp.2.2.2.1
  · exact tileIndex_lt blockH headTiles p.headTile p.headLane
      ((ceilDiv_full blockH headTiles hh) ▸ hp.2.2.1) hp.2.2.2.2

/-- The exact shape assumptions used by the public cumsum call: complete
positive token chunks and complete positive head blocks. -/
theorem full_grid_active_of_mod (batches tokens heads chunkSize blockH : Nat)
    (hc : 0 < chunkSize) (hh : 0 < blockH)
    (ht : tokens % chunkSize = 0) (hhead : heads % blockH = 0) (p : TileLane)
    (hp : GridLane batches tokens heads chunkSize blockH p) :
    ActiveLane batches tokens heads chunkSize blockH p := by
  have htc : chunkSize * (tokens / chunkSize) = tokens := by
    have h := Nat.div_add_mod tokens chunkSize
    omega
  have hhc : blockH * (heads / blockH) = heads := by
    have h := Nat.div_add_mod heads blockH
    omega
  have hg : GridLane batches (chunkSize * (tokens / chunkSize))
      (blockH * (heads / blockH)) chunkSize blockH p := by simpa [htc, hhc] using hp
  have ha := full_grid_active batches (tokens / chunkSize) (heads / blockH)
    chunkSize blockH hc hh p hg
  simpa [htc, hhc] using ha

/-- Earlier positions in an inclusive chunk scan have earlier global token
indices. Thus a valid output's prefix cannot depend on padded tail positions. -/
theorem scan_prefix_active (batches tokens heads chunkSize blockH : Nat)
    (p : TileLane) (hp : ActiveLane batches tokens heads chunkSize blockH p)
    (j : Nat) (hj : j ≤ p.tokenLane) :
    ActiveLane batches tokens heads chunkSize blockH {p with tokenLane := j} := by
  have hjc : j < chunkSize := Nat.lt_of_le_of_lt hj hp.1.2.2.2.1
  have horder : tileIndex chunkSize p.chunk j ≤
      tileIndex chunkSize p.chunk p.tokenLane := Nat.add_le_add_left hj _
  exact ⟨⟨hp.1.1, hp.1.2.1, hp.1.2.2.1, hjc, hp.1.2.2.2.2⟩,
    Nat.lt_of_le_of_lt horder hp.2.1, hp.2.2⟩

/-- Permutation `(0,2,1)` on a three-axis coordinate. -/
def swapTH (x : Coord) : Coord := ⟨x.first, x.third, x.second⟩

theorem swapTH_involution (x : Coord) : swapTH (swapTH x) = x := by
  cases x
  rfl

/-- Local tile transpose and exchange of global tile axes commute exactly. -/
theorem transpose_store_correspondence (chunkSize blockH : Nat) (p : TileLane) :
    swapTH (bthCoord chunkSize blockH p) = bhtCoord chunkSize blockH p := by
  rfl

theorem bhtCoord_injective (batches tokens heads chunkSize blockH : Nat)
    (hc : 0 < chunkSize) (hh : 0 < blockH) (p q : TileLane)
    (hp : GridLane batches tokens heads chunkSize blockH p)
    (hq : GridLane batches tokens heads chunkSize blockH q)
    (he : bhtCoord chunkSize blockH p = bhtCoord chunkSize blockH q) : p = q := by
  apply bthCoord_injective batches tokens heads chunkSize blockH hc hh p q hp hq
  exact congrArg swapTH he

/-- A BHT store of arbitrary scalar values is the pointwise transpose of BTH.
This includes `dt_raw` itself and the already-computed cumsum tile; it makes no
assumption about floating-point arithmetic used to produce those scalars. -/
theorem transpose_values {α : Type} (values : Coord → α)
    (chunkSize blockH : Nat) (p : TileLane) :
    (fun x => values (swapTH x)) (bhtCoord chunkSize blockH p) =
      values (bthCoord chunkSize blockH p) := by
  rfl

/-- Three output allocations created separately by `_chunk_cumsum`.
These constructors stand for *assumed disjoint* abstract buffers, not a proof
of a real allocator or of byte-address disjointness. -/
inductive Buffer where
  | decayBTH
  | decayBHT
  | dtBHT
  deriving DecidableEq

/-- Abstract allocation identity plus logical coordinate. -/
structure Address where
  buffer : Buffer
  coord : Coord
  deriving DecidableEq

/-- Under the disjoint-buffer abstraction, transpose outputs cannot collide
with one another even when their numerical coordinates happen to agree. -/
theorem output_buffers_separate (x y z : Coord) :
    (Address.mk .decayBTH x ≠ Address.mk .decayBHT y) ∧
    (Address.mk .decayBTH x ≠ Address.mk .dtBHT z) ∧
    (Address.mk .decayBHT y ≠ Address.mk .dtBHT z) := by
  simp

/-- Output coordinate appropriate to each distinct allocation. -/
def outputCoord (buffer : Buffer) (chunkSize blockH : Nat) (p : TileLane) : Coord :=
  match buffer with
  | .decayBTH => bthCoord chunkSize blockH p
  | .decayBHT | .dtBHT => bhtCoord chunkSize blockH p

/-- Allocation-dependent valid axis bounds. -/
def ValidAddress (batches tokens heads : Nat) (a : Address) : Prop :=
  a.coord.first < batches ∧
  match a.buffer with
  | .decayBTH => a.coord.second < tokens ∧ a.coord.third < heads
  | .decayBHT | .dtBHT => a.coord.second < heads ∧ a.coord.third < tokens

theorem outputCoord_valid (batches tokens heads chunkSize blockH : Nat)
    (buffer : Buffer) (p : TileLane)
    (hp : ActiveLane batches tokens heads chunkSize blockH p) :
    ValidAddress batches tokens heads ⟨buffer, outputCoord buffer chunkSize blockH p⟩ := by
  cases buffer <;> exact ⟨hp.1.1, by first | exact ⟨hp.2.1, hp.2.2⟩ | exact ⟨hp.2.2, hp.2.1⟩⟩

/-- Precise boundary of the logical memory theorem: with an injective layout
map on *valid* addresses (including disjoint output allocation ranges), unique
logical ownership implies unique physical writes. The layout hypothesis is
required, rather than inferred for CUDA, and no assertion is made about
padded lanes' physical memory accesses. -/
theorem lift_logical_ownership_to_physical {A : Type} (layout : Address → A)
    (batches tokens heads chunkSize blockH : Nat)
    (hinj : ∀ a a', ValidAddress batches tokens heads a →
      ValidAddress batches tokens heads a' → layout a = layout a' → a = a')
    (hc : 0 < chunkSize) (hh : 0 < blockH) (buffer : Buffer) (p q : TileLane)
    (hp : ActiveLane batches tokens heads chunkSize blockH p)
    (hq : ActiveLane batches tokens heads chunkSize blockH q)
    (he : layout ⟨buffer, outputCoord buffer chunkSize blockH p⟩ =
      layout ⟨buffer, outputCoord buffer chunkSize blockH q⟩) : p = q := by
  have ha := hinj _ _ (outputCoord_valid batches tokens heads chunkSize blockH buffer p hp)
    (outputCoord_valid batches tokens heads chunkSize blockH buffer q hq) he
  have he' := congrArg Address.coord ha
  cases buffer
  · exact bthCoord_injective batches tokens heads chunkSize blockH hc hh p q hp.1 hq.1 he'
  · exact bhtCoord_injective batches tokens heads chunkSize blockH hc hh p q hp.1 hq.1 he'
  · exact bhtCoord_injective batches tokens heads chunkSize blockH hc hh p q hp.1 hq.1 he'

end Mamba2.Indexing
