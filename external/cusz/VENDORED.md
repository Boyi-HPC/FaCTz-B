# Vendored cuSZ Source

This directory contains the CUDA compression libraries used by cpSZ-GPU's GPU
Huffman bridge.

- Upstream: https://github.com/szcompressor/cuSZ
- Revision: `d22d9c44b791da798ee8c4e8d68f3a1fb17613f1`
- License: `LICENSE` in this directory
- Imported components: `cmake`, `codec`, `portable`, `psz`, and `utils`

Examples, tests, Python bindings, the top-level cuSZ API/CLI, and optional
lossless-codec sources are not included because cpSZ-GPU does not build or link
them. `src/Makefile` builds this snapshot with those components disabled and
installs the local libraries under `external/cusz/build-cuda/install`.

The imported codec and kernel source files are kept at the revision above. The
vendored CMake files add `PSZ_BUILD_MAIN_LIBRARY` so this repository can build
only the required libraries without pulling in the unused LC source. Other
cpSZ-GPU-specific integration lives in `src/gpu_huffman_bridge.cu` and
`src/Makefile`.
