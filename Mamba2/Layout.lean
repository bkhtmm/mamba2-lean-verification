import Mamba2.Indexing

/-!
Contiguous scalar-element address bounds for the full-tile source model.
These are natural-number offsets, not machine pointer or byte-address proofs.
-/
namespace Mamba2.Layout
open Mamba2.Indexing

/-- Two valid row/column coordinates encode below the complete allocation. -/
theorem encode_lt (rows cols row col : Nat) (hr : row < rows) (hc : col < cols) :
    row * cols + col < rows * cols := by
  calc
    row * cols + col < row * cols + cols := Nat.add_lt_add_left hc _
    _ = (row + 1) * cols := by simp [Nat.add_mul]
    _ ≤ rows * cols := Nat.mul_le_mul_right cols (by omega)

/-- Every BTH address of a valid logical scalar fits its contiguous allocation. -/
theorem bth_offset_lt (B T H b t h : Nat) (hb : b < B) (ht : t < T) (hh : h < H) :
    (b * T + t) * H + h < B * T * H :=
  encode_lt (B*T) H (b*T+t) h (encode_lt B T b t hb ht) hh

/-- Every BHT address of the permuted scalar fits its contiguous allocation. -/
theorem bht_offset_lt (B T H b t h : Nat) (hb : b < B) (ht : t < T) (hh : h < H) :
    (b * H + h) * T + t < B * H * T :=
  encode_lt (B*H) T (b*H+h) t (encode_lt B H b h hb hh) ht

/-- Actual scalar offsets of the source tile are all in bounds on full tiles. -/
theorem full_tile_access_bounds (B T H C K batch chunk headTile i j : Nat)
    (hb : batch < B) (htile : chunk * C + C ≤ T)
    (hhead : headTile * K + K ≤ H) (hi : i < C) (hj : j < K) :
    headTile * K + j < H ∧
    (batch * T + (chunk * C + i)) * H + (headTile * K + j) < B * T * H ∧
    (batch * H + (headTile * K + j)) * T + (chunk * C + i) < B * H * T := by
  have ht : chunk * C + i < T := by omega
  have hh : headTile * K + j < H := by omega
  exact ⟨hh, bth_offset_lt B T H _ _ _ hb ht hh,
    bht_offset_lt B T H _ _ _ hb ht hh⟩

/-- Valid indices give unique scalar offsets; no assumption about padding. -/
theorem encode_injective (cols r₁ r₂ c₁ c₂ : Nat)
    (hc₁ : c₁ < cols) (hc₂ : c₂ < cols)
    (he : r₁ * cols + c₁ = r₂ * cols + c₂) : r₁ = r₂ ∧ c₁ = c₂ := by
  have hp : 0 < cols := by omega
  have h₁ : (r₁ * cols + c₁) / cols = r₁ := by
    simpa [tileIndex, Nat.mul_comm] using (tileIndex_owner cols r₁ c₁ hp hc₁).1
  have h₂ : (r₂ * cols + c₂) / cols = r₂ := by
    simpa [tileIndex, Nat.mul_comm] using (tileIndex_owner cols r₂ c₂ hp hc₂).1
  have hr : r₁ = r₂ := by rw [← h₁, ← h₂, he]
  exact ⟨hr, by simpa [hr] using he⟩

/-- Contiguous BTH linearization is injective on valid tensor coordinates. -/
theorem bth_offset_injective (T H b₁ b₂ t₁ t₂ h₁ h₂ : Nat)
    (ht₁ : t₁ < T) (ht₂ : t₂ < T) (hh₁ : h₁ < H) (hh₂ : h₂ < H)
    (he : (b₁*T+t₁)*H+h₁ = (b₂*T+t₂)*H+h₂) :
    b₁ = b₂ ∧ t₁ = t₂ ∧ h₁ = h₂ := by
  obtain ⟨hr, hh⟩ := encode_injective H _ _ _ _ hh₁ hh₂ he
  obtain ⟨hb, ht⟩ := encode_injective T _ _ _ _ ht₁ ht₂ hr
  exact ⟨hb, ht, hh⟩

end Mamba2.Layout
