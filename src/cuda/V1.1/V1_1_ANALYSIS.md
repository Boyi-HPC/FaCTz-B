# V1.1 构建、运行与优化结论

## 测试环境

- GPU: NVIDIA GeForce RTX 3080, compute capability 8.6
- CUDA: nvcc 12.6.85
- Nsight Compute: 2025.1.0
- 数据: Ocean `uf.dat/vf.dat`, `2400 x 3600`, 共 8,640,000 个二维向量
- 参数: `max_relative_eb=0.1`
- 构建类型: Release, `CMAKE_CUDA_ARCHITECTURES=86`
- 计时边界: 程序输出的 CUDA kernel 阶段之和，不包含文件 I/O 和 Huffman CPU 建树时间
- 论文: `/home/boyi/FaCTz/ppopp27-summer-paper854.pdf`

主 `.cu` 与 `FaCTz/experiments/cuda/cpszg_2d_v1.1.cu` 的 SHA-256 一致。原 experiments 目录没有保留该阶段的 `fused_ebzero_kernel.cuh`，本目录中的 helper 是根据 V1.1 调用参数和 bitmask/scan 数据流重建的兼容实现。

## 构建和运行

```bash
make -C src/cuda/V1.1
make -C src/cuda/V1.1 run
make -C src/cuda/V1.1 clean
```

Make target 和 CMake target 均为 `cpszg_2d_v1_1`。独立 Make 构建、全新 CMake 全版本构建和实际运行均成功。

## 正确性

V1.1 自带的内存解压与文件解压结果对 U/V 都是 bit-identical。统一 verifier 的结果为：

| 指标 | 结果 |
|---|---:|
| 原始/解压 critical points | 20,929 / 20,929 |
| matched / FP / FN / type mismatch | 20,929 / 0 / 0 / 0 |
| U 最大绝对误差 | 0.0624847412 |
| V 最大绝对误差 | 0.0622329712 |
| overall max relative EB | 1.8089997539e-4 |

结论：V1.1 在该输入上保持全部 critical points，且与 V1 的恢复误差相同。

## 性能

先运行一次 warm-up，再独立运行 5 次。下面的 runs 已排序。

| 版本 | 5 次 GPU compression total (ms) | 中位数 | 相对上一版 |
|---|---|---:|---:|
| V1 | 12.418, 13.467, 14.082, 14.198, 31.961 | 14.082 ms | - |
| V1.1 | 12.194, 12.292, 12.484, 12.571, 12.608 | 12.484 ms | 1.128x |

V1.1 阶段中位数：derive EB 2.811 ms、land 0.214 ms、zero-EB 0.797 ms、tile uniform 0.527 ms、Lorenzo 2.017 ms、Huffman 5.883 ms、tile payload pack 0.140 ms。

V1.1 内置 GPU 解压的 5 次中位数为 3.875 ms；V1 为 3.781 ms，因此 V1.1 解压没有加速，约慢 2.5%。

## 源码能够确认的变化

1. zero-EB 位置由稀疏 `uint32_t` index 数组改成完整 bitmask。`kernel_compact_zeroeb_values` 根据 bitmask 和 exclusive-scan offset 按行优先顺序收集原值，解压时执行反向 restore。
2. 压缩流程为 U/V 分别生成 bitmask、执行逐元素 Thrust scan，再执行两个 compact kernel。
3. 文件中 zero-EB side stream 从 `index + value` 改为 `bitmask + value`。Ocean 上文件由 V1 的 27,006,658 bytes 增至 28,179,902 bytes，compression ratio 从 2.559 降至 2.453。
4. derive-EB、tile uniform、Lorenzo 和旧 Huffman bridge 没有实质算法变化。因此 1.128x 总体差异主要来自本批次 derive-EB 波动，不能归因于 V1.1 新增 bitmask 逻辑。

关键位置：`cpszg_2d_v1.1.cu:556`、`:1133`、`:1172`、`:1240`，以及 `fused_ebzero_kernel.cuh`。

## Profiling 证据

Nsight Compute `basic` 对新增的 zero-EB compact kernel 得到：

| kernel | duration | achieved occupancy | DRAM throughput | registers/thread |
|---|---:|---:|---:|---:|
| compact U | 77.60 us | 56.93% | 14.53% | 16 |
| compact V | 77.50 us | 56.94% | 15.65% | 16 |

该 kernel 本身不重，但 V1.1 还需要两个长度为 `n` 的 scan。实测 zero-EB 阶段 0.797 ms，高于 V1 的 0.216 ms；bitmask 的主要价值是稳定、可逆的 side-stream 表达，而不是这一版的速度。

## 与论文对应

论文 Section 4.1 描述了 bitmask、population count、scan 和 ordered side stream。V1.1 只完成了这一思路的中间形态：它仍扫描两个长度为 `n` 的逐元素 offset 数组，而论文实现按 mask word 计数并扫描，后者在 V1.3 才出现。论文没有给出 V1.1 这样的版本编号，因此不能把论文最终性能直接归到该中间版本。

## 限制

- 单一 Ocean 数据集和单一误差参数，不代表其他尺寸或 zero-EB 密度。
- V1 的 5 次结果有一次 31.961 ms 离群值，因此使用中位数；V1.1 的代码变化本身不支持“derive EB 被优化”的结论。
- NCU replay 会显著放大程序内计时，报告只使用 NCU 的 per-kernel 指标，不使用 profiling run 的总时间。
