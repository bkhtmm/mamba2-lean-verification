# Third-party sources and acknowledgments

- NVIDIA and Jane Street: Mamba-2 cuTile implementation and [technical article](https://developer.nvidia.com/blog/accelerating-mamba-2-with-high-performance-forward-pass-kernels-in-nvidia-cuda-tile/). The unmodified `vendor/mamba2.py` is from NVIDIA/TileGym commit `3a871364f1fbd59ef30b1afcc5a79fc3634888ce`. Its NVIDIA copyright and SPDX notice are retained; `vendor/TileGym-LICENSE` contains the upstream MIT license. The separate TileGym skills subtree is not redistributed here.
- VeriTile dependency, commit `16abbb7e77bf111bf7ede620ec6bf91744e7325b`: MIT, copyright 2026 Zenan Li. The required notice is included in [`vendor/VeriTile-LICENSE`](vendor/VeriTile-LICENSE).
- Mathlib and Lean: mathematical library and proof checker, pinned through Lake and `lean-toolchain`; their respective upstream licenses apply.

No upstream author has reviewed or endorsed this artifact.
