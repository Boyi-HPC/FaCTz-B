# V1.2 构建、运行与优化结论

## 测试环境与命令

环境与 V1.1 报告相同：RTX 3080、CUDA 12.6.85、Ocean `2400 x 3600`、`max_relative_eb=0.1`、Release、SM 86。论文为 `/home/boyi/FaCTz/ppopp27-summer-paper854.pdf`。

主 `.cu` 与 experiments 版本的 SHA-256 一致。原目录同样没有保留 V1.2 当时的中间 fused helper，本目录按其“ID + flag + bitmask”调用契约重建；paired tile-reduction helper 来自最终 FaCTz 源码。

```bash
make -C src/cuda/V1.2
make -C src/cuda/V1.2 run
make -C src/cuda/V1.2 clean
```

Make/CMake target 为 `cpszg_2d_v1_2`。独立 Make 构建和全新 CMake 全版本构建均成功。

## 正确性

V1.2 的 `.cucpsz` 由统一 GPU 解压器恢复，再由统一 verifier 检查：20,929 个原始 critical points 全部匹配，FP/FN/type mismatch 都为 0。U/V 最大绝对误差分别为 0.0624847412 和 0.0622329712，与 V1/V1.1 完全相同。

## 性能

一次 warm-up 后运行 5 次，runs 已排序。

| 版本 | 5 次 GPU compression total (ms) | 中位数 | speedup |
|---|---|---:|---:|
| V1.1 | 12.194, 12.292, 12.484, 12.571, 12.608 | 12.484 ms | - |
| V1.2 | 5.701, 5.909, 6.244, 6.400, 6.674 | 6.244 ms | 1.999x |

GPU-resident 解压 CUDA-event 时间为 3.699, 3.735, 3.753, 3.787, 3.906 ms，中位数 3.753 ms。

| 阶段中位数 | V1.1 | V1.2 | 变化 |
|---|---:|---:|---:|
| derive EB | 2.811 | 2.906 | +0.095 ms |
| land | 0.214 | 0.192 | -0.022 ms |
| zero-EB | 0.797 | 0.754 | -0.043 ms |
| tile EB reduction | 0.527 | 0.210 | -0.317 ms |
| Lorenzo | 2.017 | 0.765 | -1.252 ms |
| Huffman kernels | 5.883 | 1.342 | -4.541 ms |

## 源码能够确认的优化

1. `derive_eb_offline_v2` 不再写两个完整的 float EB 数组，只写 U/V exponent ID，减少约 `2 * n * sizeof(float)` 的输出和后续存储。
2. `kernel_reduce_tile_eb_pair` 在一次 32x32 tile launch 中同时归约 U/V，通过 warp shuffle 和少量 shared memory 输出每 tile 两个 ID，不再对 U/V 分别做完整 shared-memory broadcast。
3. tile Lorenzo 直接读取 `tile_eq_dEb_U/V`，不再读取完整的逐点 float EB 数组。Ocean 只需每个分量 8,475 bytes 的 tile EB payload。
4. 新 Huffman bridge 并行执行 U/V histogram，并发构建两个 CPU codebook，再串行发射会饱和 GPU 的 encode。日志把 kernel-only 与 wall/E2E 明确分开。
5. 压缩 executable 删除了内嵌解压和大段 debug/rightness 路径；正确性改由独立解压器和 verifier 测试。这降低了程序职责，但不会计入 GPU kernel total 的 speedup。

关键位置：`cpszg_2d_v1.2.cu:263`、`:1083`、`:1115`、`:1144`，`tile_uniform_eb.cuh:8`，`../gpu_huffman_bridge.cu:670`。

## Profiling 证据

`kernel_reduce_tile_eb_pair` 的 NCU basic 指标：duration 165.28 us、achieved occupancy 57.37%、memory throughput 55.40%、DRAM throughput 16.45%、20 registers/thread。它一次写出 U/V 两个 tile ID，实测 tile 阶段从 0.527 ms 降至 0.210 ms。

最有力的运行证据是 Lorenzo 和 Huffman 阶段分别下降约 62% 和 77%，共同解释了接近 2x 的总 speedup。源码和阶段计时都支持这一结论。

## 与论文对应

- Section 4.1 的 `n -> n/1024` tile-bound side channel 和 parallel prefix-sum Lorenzo 与 V1.2 直接对应。
- Section 4.3 的 U/V Huffman concurrency 与新 bridge 对应。
- Section 4.1 所说的“single fused kernel replaces four passes”尚未实现；V1.2 仍将 land、zero-EB 和 tile reduction 分开，这部分要到 V1.3 才对应。

## 限制

- Huffman kernel-only 计时不包含 CPU codebook 和 host/device copy，因此不能作为应用端 wall-time speedup。
- 只测试一个输入；tile minimum 的收益和压缩率代价依赖场结构。
- V1.2 executable 本身只压缩，正确性依赖本次新增并实际运行的 `factz_decpszg_2d` 与 `factz_verify_cpszg_2d`。
