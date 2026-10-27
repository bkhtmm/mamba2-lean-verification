# Third-party sources and acknowledgments

- NVIDIA and Jane Street: Mamba-2 cuTile implementation and [technical article](https://developer.nvidia.com/blog/accelerating-mamba-2-with-high-performance-forward-pass-kernels-in-nvidia-cuda-tile/). The unmodified `vendor/mamba2.py` is from NVIDIA/TileGym commit `3a871364f1fbd59ef30b1afcc5a79fc3634888ce`. Its NVIDIA copyright and SPDX notice are retained; `vendor/TileGym-LICENSE` contains the upstream MIT license. The separate TileGym skills subtree is not redistributed here.
- VeriInfra-Lab/VeriTile, commit `16abbb7e77bf111bf7ede620ec6bf91744e7325b`, MIT, copyright 2026 Zenan Li. This project imports its Triton-style semantics, scan, execution, memory and audit definitions. The scan proof follows the `foldl_realAdd_coe` / `scan2d_axis1_sum` technique in `bench/tritonbench_g/chunked_cumsum_fwd/ChunkedCumsumFwd.lean`; its MIT license is included as `vendor/VeriTile-LICENSE`. Dependency source is obtained by Lake, rather than copied into this archive.
- Mathlib and Lean: mathematical library and proof checker, pinned through Lake and `lean-toolchain`; their respective upstream licenses apply.
- RESOLVE, [arXiv:2610.05683v1](https://arxiv.org/html/2610.05683v1), by Ashkan Vedadi Gargary, Guido Martínez, Sebastian Burckhardt, Gabriel Ebner, Abhinav Jangda, Madan Musuvathi and Tyler Sorensen. Methodological inspiration only. We did not execute RESOLVE, its instrumentation, or Kuiper. No RESOLVE/Kuiper source is redistributed.

No upstream author has reviewed or endorsed this artifact.
