# Naive and V1 CUDA Validation

Date: 2026-08-23

## Scope and environment

- Repository: `/home/boyi/FaCTz-B`
- No repository-level `AGENTS.md` was present.
- GPU: NVIDIA GeForce RTX 3080, compute capability 8.6, 10 GiB.
- CUDA compiler: nvcc 12.6.85; driver: 610.88.
- Input: `data/uf.dat` and `data/vf.dat`, float32, `2400 x 3600`, 8,640,000 values per component.
- Parameter: relative error bound `0.1`.
- Paper assumption: `/home/boyi/FaCTz/ppopp27-summer-paper854.pdf`, the only PDF found in the two related repositories. Its title is *FaCTz: Fast Critical-Point and Topology-Aware GPU Compression for Scientific Vector Fields*.

## Build model

The two Makefiles are CMake wrappers, so the effective flags come from the CMake targets and their transitive dependencies.

- Naive target: `cpszg_2d_no_opt_speed`
  - Direct sources: `naive/cpszg_2d_no_opt_speed.cu` and `naive/gpu_huffman_bridge_no_opt_speed.cu`.
  - Shared CUDA sources: `cusz/lproto_c.cu` and `cusz/lproto_x.cu` through `factz_lorenzo`.
- V1 target: `cpszg_2d_v1`
  - Direct sources: `V1/cpszg_2d_v1.cu`, `V1/gpu_huffman_u2_bridge.cu`, and the shared naive Huffman bridge.
  - V1-only helpers: `V1/fused_ebzero_kernel.cuh` and `V1/tile_uniform_eb.cuh`.
  - Shared CUDA sources: the same `factz_lorenzo` library as naive.
- Effective CUDA flags: `-O3 -DNDEBUG -std=c++17 --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations`.
- Definitions: `PSZ_USE_CUDA=1`, plus `PHF_USE_CUDA` and `_PORTABLE_USE_CUDA` from the vendored cuSZ/PHF targets.
- Architectures: the pre-existing project setting `70;75;80;86`; the test GPU runs the `sm_86` image.
- Main dependencies: CUDA runtime, FTK, vendored cuSZ core/memory/eval, and PHF.

Commands that were executed successfully:

```bash
cd /home/boyi/FaCTz-B/src/cuda/naive && make -j8
cd /home/boyi/FaCTz-B/src/cuda/V1 && make -j8

cmake -S /home/boyi/FaCTz-B -B /home/boyi/FaCTz-B/build-validation-cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES='70;75;80;86' \
  -DFTK_DIR=/home/boyi/FaCTz-B/external/ftk/install/lib/cmake/FTK \
  -DFACTZ_BUILD_CUDA=ON -DFACTZ_BUILD_CUDA_NAIVE=ON -DFACTZ_BUILD_CUDA_V1=ON
cmake --build /home/boyi/FaCTz-B/build-validation-cmake --parallel 8 \
  --target cpszg_2d_no_opt_speed cpszg_2d_v1

cd /home/boyi/FaCTz-B/build/validation/naive
/home/boyi/FaCTz-B/build-validation-cmake/bin/cpszg_2d_no_opt_speed uf.dat vf.dat 2400 3600 0.1
cd /home/boyi/FaCTz-B/build/validation/v1
/home/boyi/FaCTz-B/build-validation-cmake/bin/cpszg_2d_v1 uf.dat vf.dat 2400 3600 0.1

cd /home/boyi/FaCTz-B/src/cuda/V1
make clean BUILD_DIR=/home/boyi/FaCTz-B/build-validation-cmake
```

Both clean-CMake executables exited with status 0. `make clean` also exited successfully. Compilation emitted existing FTK host/device and ignored-`fread` warnings, but no errors.

## Correctness

Each version was run in a separate directory so its `.out` and `.cucpsz` files could not overwrite the other version's files.

| Check | Naive | V1 |
|---|---:|---:|
| In-memory versus file decompression | U/V bit-identical | U/V bit-identical |
| Decompressed EB verification errors | U=0, V=0 | U=0, V=0 |
| Original critical points | 20,929 | 20,929 |
| Decompressed critical points | 20,929 | 20,929 |
| TP / FP / FN | 20,929 / 0 / 0 | 20,929 / 0 / 0 |

Floating-point comparison against the original field:

| Version/component | Max absolute error | RMSE | Program PSNR |
|---|---:|---:|---:|
| Naive U | 15.5392151 | 0.1686684 | 66.2260 |
| Naive V | 15.4406738 | 0.1373586 | 68.2285 |
| V1 U | 0.0624847 | 0.0012557 | 108.7888 |
| V1 V | 0.0622330 | 0.0007691 | 113.2656 |

The two reconstructed fields are not numerically identical. Naive versus V1 has max absolute differences of 15.5371704 (U) and 15.4447327 (V), with RMSE 0.1686735 and 0.1373616. This is expected because V1 changes the predictor and uses a conservative tile-common bound. The relevant algorithmic invariant is preserved: both satisfy the requested bound according to the program's checks and both preserve all critical points.

The actual `.cucpsz` sizes are 17,113,928 bytes (naive) and 27,006,658 bytes (V1), giving comparable on-disk ratios of 4.039x and 2.559x from 69,120,000 raw bytes. Naive prints 4.311x because that calculation excludes its 1,080,000-byte land mask; V1's printed 2.559x includes the complete estimated file. The on-disk values should therefore be used for cross-version comparison.

## Performance method and results

One complete run of each version was used as warm-up. Five subsequent process runs were alternated on an otherwise idle RTX 3080. Times below are the program's existing internal timings: CUDA events for GPU stages and decompression, while Huffman includes the bridge's host histogram/book construction and encode wall time. File I/O, CPU critical-point verification, and quality analysis are outside `COMPRESS_TIME`.

Compression total times in milliseconds:

| Run | Naive | V1 |
|---:|---:|---:|
| 1 | 21.946 | 13.292 |
| 2 | 21.821 | 12.676 |
| 3 | 22.651 | 15.884 |
| 4 | 22.539 | 13.215 |
| 5 | 21.188 | 14.897 |
| Median | **21.946** | **13.292** |

Median compression speedup is `21.946 / 13.292 = 1.651x`. Median throughput is approximately 2.93 GiB/s for naive and 4.84 GiB/s for V1.

| Median stage (ms) | Naive | V1 | Naive / V1 |
|---|---:|---:|---:|
| Derive EB | 4.202 | 3.822 | 1.10x |
| Land handling | 0.240 | 0.261 | 0.92x |
| Zero-EB handling | 1.304 | 0.234 | 5.57x |
| Tile-uniform EB | - | 0.561 | added cost |
| Lorenzo compression | 10.129 | 2.060 | 4.92x |
| Huffman | 6.249 | 6.286 | 0.99x |
| Tile-EB packing | - | 0.106 | added cost |
| Total | **21.946** | **13.292** | **1.65x** |

Decompression times were naive `[5.961728, 5.498752, 5.629024, 5.432224, 5.441536]` ms and V1 `[3.708160, 3.723264, 3.841056, 3.717120, 4.130816]` ms. Their medians are 5.498752 and 3.723264 ms, so V1 decompression is **1.477x** faster.

## Kernel-level analysis

Labels below distinguish source facts, measurements, and CUDA-based interpretation.

### 1. Error-bound derivation

- **Fact:** `derive_eb_offline_v2` keeps the same body, but V1 changes `BLOCKSIZE_Y` from 8 to 16. The launch changes from `(32,8)` and grid `(120,400)` to `(32,16)` and `(120,172)`. Both use shared-memory patches padded by one x element.
- **Measured:** NCU reports 33 registers/thread for both. Static shared memory doubles from 5.28 to 10.56 KiB/block, but achieved occupancy remains 97.41% versus 97.71%. Raw kernel duration is 784.96 versus 770.62 us; the five-run stage medians show a modest 1.10x gain.
- **Interpretation:** the taller block covers more useful rows per two-row halo and launches fewer total threads, reducing overlap/reloads. The cost is twice the shared memory and 512-thread blocks. The data supports only a modest improvement, not a major optimization claim.
- **Inactive code:** V1 contains `derive_eb_offline_v3`, but active calls still launch v2; v3 must not be credited as a V1 optimization.

### 2. Land handling

- **Fact:** V1 precomputes `land_id` and `land_eb` on the CPU instead of evaluating `log2` in the device lambda. It still launches a full-field Thrust `for_each`.
- **Measured:** median time changes from 0.240 to 0.261 ms, so this test does not support a performance gain.
- **Interpretation:** removing repeated arithmetic is reasonable, but this pass is dominated by scanning memory and launch variation on this input.

### 3. Zero-EB classification and replacement

- **Fact:** naive performs one `thrust::sequence`, four `copy_if` calls, and four `transform` calls. V1 replaces these nine full-array operations with `kernel_fused_ebzero`, which gathers U/V value-index pairs, rewrites IDs, and replaces small EBs in one pass.
- **Measured:** stage median falls from 1.304 to 0.234 ms (5.57x). NCU measures the fused kernel at 220.13 us, 64.27% DRAM throughput, 70.60% achieved occupancy, 24 registers/thread, and no static shared memory.
- **Interpretation:** fewer launches and full-field rereads explain the gain. The tradeoff is two `atomicAdd` counters; heavy zero-EB density could contend, and compacted side-stream order is not guaranteed even though index/value pairs remain aligned.

### 4. Tile-common error bound

- **Fact:** V1 launches one `kernel_uniformize_tile_eb` for U and one for V. Each 32x32 block min-reduces 1,024 integer IDs in shared memory and broadcasts the minimum back to all valid elements.
- **Measured:** one kernel takes 284.03 us in NCU; it uses 4.10 KiB shared memory, 16 registers/thread, and reaches 60.59% achieved occupancy versus 66.67% theoretical. The two-kernel event median is 0.561 ms.
- **Interpretation:** the common bound enables tiled Lorenzo and reduces EB metadata to one byte per tile, but 1,024-thread blocks and ten reduction barriers limit flexibility. Taking a tile minimum tightens many bounds: V1 outliers rise from 10,624/16,247 to 372,462/379,656 for U/V, which explains much of its lower compression ratio.

### 5. Lorenzo compression

- **Fact:** naive's row kernel launches only 75 blocks of 32 threads. One thread loops serially over all 3,600 values in a row. V1 launches 8,475 blocks of 32x32 threads; values are quantized and two-dimensional Lorenzo differences are formed from a shared-memory tile.
- **Measured:** for one component, NCU reports 5.41 ms and 2.31% achieved occupancy for naive, versus 814.62 us and 65.34% for V1 (6.64x raw-kernel speedup). The two-component stage median improves 4.92x. V1 uses 4.22 KiB shared memory/block and 24 registers/thread.
- **Interpretation:** V1 removes the long per-row dependency chain and exposes thousands of resident blocks. Costs include 1,024-thread blocks, shared-memory barriers, independent tile boundaries, and many more outliers under the tighter common bound.

### 6. Lorenzo decompression

- **Fact:** naive again scans each row serially in one thread. V1 performs shared-memory prefix sums in x and then y inside each tile.
- **Measured:** NCU reports 1.40 ms and 2.32% occupancy for naive versus 424.96 us and 64.12% for V1 for one component, a 3.29x raw-kernel improvement. Full decompression improves 1.477x because entropy decode, scatters, and setup remain in the measured interval.
- **Interpretation:** parallel prefix sums trade additional synchronization and shared memory for much wider parallelism.

### 7. Error-bound payload and Huffman path

- **Fact:** naive Huffman-codes four full arrays: U/V quantization codes and U/V per-element EB IDs. V1 Huffman-codes only U/V and stores 8,475 raw EB bytes per component, reducing the EB side channel from 8,640,000 IDs to `n/1024` IDs.
- **Measured:** Huffman medians are effectively unchanged (6.249 versus 6.286 ms). V1's U/V code blobs and special-value streams are much larger, so eliminating two EB Huffman streams does not reduce total time or file size on this dataset.
- **Interpretation:** metadata reduction succeeds, but conservative tile bounds move cost into quantization-code entropy and outliers.

## Paper-to-code mapping

The current V1 is a partial prototype of the paper's block-wise mode, not an exact implementation of the paper artifact.

| Paper item | Current code correspondence | Status |
|---|---|---|
| Section 3.3, four-step block-wise safe bound | Per-triangle analytic bound, incident-cell minimum, component EB IDs, then 32x32 tile minimum | Matches conceptually |
| Section 4.1, 32x16 derivation kernel with two-vertex halo and x padding | `derive_eb_offline_v2`, `(32,16)`, `[TileDim_X+1]` shared arrays | Matches |
| Section 4.1, one fused special/tile kernel | Current V1 has a Thrust land pass, one atomic zero-EB kernel, and two separate tile-reduction kernels | Partial; not the paper kernel |
| Section 4.1, `__ballot_sync`, bit masks, CUB scan, no flag arrays | CPU-built land mask plus value/index arrays and atomic counters | Missing/different |
| Section 4.1, EB side channel `n/1024` and parallel-prefix Lorenzo | 8,475 bytes/component and tiled 32x32 Lorenzo | Matches |
| Section 4.1, roughly 8x parallel Lorenzo claim | NCU gives 6.64x per compression kernel and event stage gives 4.92x on RTX 3080 | Direction supported; magnitude differs |
| Section 4, pipeline entirely on device | CPU land-bitpack construction, host file I/O, and host Huffman book/timing remain | Does not match |
| Section 4, default ANS backend | Current executable uses Huffman only | Missing |
| Section 4.3, concurrent U/V histograms and codebooks | Current U2 bridge processes U then V on one stream and one reusable buffer | Missing |
| Figure 7, EB/EP/LP/HF stage split | Program exposes derive, land/zero/uniform, Lorenzo, and Huffman timings | Comparable stage structure |
| Table 1, Ocean dataset | Resolution 2400x3600 and 20,929 critical points match this test | Matches input |

Paper Table 2 reports FaCTz-B Ocean results on an A100, while this test uses an RTX 3080 and a partial prototype. Those published numbers must not be used as expected values for this run.

## Changed files

- `CMakeLists.txt`: added `FACTZ_BUILD_CUDA_V1`.
- `src/cuda/CMakeLists.txt`: conditionally adds the V1 subdirectory.
- `src/cuda/V1/Makefile`: V1 CMake-wrapper build/run/clean entry points.
- `src/cuda/V1/CMakeLists.txt`: V1 executable and Huffman bridge targets.
- `src/cuda/V1/fused_ebzero_kernel.cuh`: compatible fused zero-EB helper required by the V1 source.
- `src/cuda/V1/tile_uniform_eb.cuh`: compatible 32x32 tile-common EB helper.
- `src/cuda/V1/gpu_huffman_u2_bridge.cu`: missing U2-only Huffman bridge required by V1.
- `src/cuda/NAIVE_V1_ANALYSIS.md`: this validation report.

The pre-existing modified `src/cuda/naive/cpszg_2d_no_opt_speed.cu`, `.vscode/`, and `src/cuda/naive/run_result.txt` were not changed as part of this task.

## Limitations and remaining work

- Results cover one dataset, one error bound, and one GPU; tile size and input sensitivity remain untested.
- GPU clocks were not locked. Five-run medians limit noise, but V1 derive-EB showed visible run-to-run variation.
- Nsight Systems captured CUDA API calls but reported no CUDA kernel data for this driver/tool combination. NCU reports were therefore used for targeted kernels.
- NCU replays kernels and heavily perturbs event timing; only its per-kernel metrics are used, while speedups come from unprofiled runs.
- Current V1 omits several paper optimizations listed above. Calling it a reproduction of the full paper implementation would be inaccurate.
- Existing unchecked `fread` calls mean corrupt/truncated `.cucpsz` files are not robustly rejected; this was outside the requested build/performance scope.
