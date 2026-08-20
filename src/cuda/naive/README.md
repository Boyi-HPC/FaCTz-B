# Naive CUDA cpSZ-Offline

This directory contains the unoptimized CUDA pipeline and its dedicated GPU
Huffman bridge. Shared CUDA options and customized Lorenzo support live in the
parent `src/cuda/` directory. Build products are written to the repository-level
`build/` directory.

From this directory:

```bash
make build
make run
```

`make run` defaults to the included 2400 by 3600 float32 U/V fields and an
error bound of 0.1. Override inputs and dimensions as Make variables:

```bash
make run U=/path/to/u.dat V=/path/to/v.dat R1=2400 R2=3600 EB=0.1
```

Useful build overrides are `CUDACXX`, `CUDA_ARCHITECTURES`, `BUILD_TYPE`,
`BUILD_DIR`, and `JOBS`. For example:

```bash
make build CUDACXX=/path/to/nvcc CUDA_ARCHITECTURES=80 JOBS=8
```

The executable is generated as `build/bin/cpszg_2d_no_opt_speed`. Runtime
debug arrays are isolated under `build/runs/naive`.
