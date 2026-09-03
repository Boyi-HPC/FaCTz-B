
# V1.1 构建、运行与优化结论

初始测试日期：2026-08-24；当前代码复测日期：2026-09-02

> **2026-09-02 更新：** 当前代码使用 `RADIUS=512` 和
> `LORENZO_TILE_DIM=16`。本次按新的统一条件重新编译 V1.0/V1.1，
> 每版正式运行 100 次并比较程序打印的 `COMPRESS_TIME` 算术平均值。
> 本节结果优先于后面的历史 5 次测试；两批测试的代码、tile/radius 和
> 统计方法不同，不能混用。

## 2026-09-02 当前代码 100 次复测

### 测试条件与方法

- Git commit：`2ec421759b9dd4d69649ecd4cb08260d8c431e0a`。
- GPU：NVIDIA GeForce RTX 3080，Compute Capability 8.6；CUDA 12.6.85。
- 构建：Release，`-O3 -DNDEBUG`，CUDA 架构 `sm_86`；两版均由当前源码重新编译。
- 输入：`uf.dat`、`vf.dat`，float32，`2400 x 3600`，U/V 各 8,640,000 个值。
- 参数：`max_relative_eb=0.1`，V1.0 和 V1.1 均为 `RADIUS=512`、`16x16` tile。
- 每版先预热 5 次，随后各正式运行 100 次；预热不计入结果。
- 奇数 pair 按 V1.0→V1.1，偶数 pair 按 V1.1→V1.0；每版位于位置 1/2 各 50 次。
- 每次调用均是新进程，没有注入进程内 CUDA 预热，因此保留了直接反复运行 CLI
  时的首次 kernel、GPU 频率和系统调度波动。
- 本节严格使用 100 次 `COMPRESS_TIME` 的算术平均值，所有有效样本均保留，
  没有删除较慢样本。
- 原始日志、运行脚本和 CSV 位于
  `src/cuda/V1.1/benchmark_v1_v11_100runs_20260902T223532Z/`。

程序的 `COMPRESS_TIME` 是已标记阶段的内部时间之和，不是从输入文件读取到
`.cucpsz` 落盘的端到端墙钟时间。CUDA 阶段主要使用 event；Huffman 使用 host
chrono，包含其计时区间内的 histogram、D2H 同步、CPU book 构建和 encode。

### 100 次算术平均结果

下表的差值定义为 `V1.1 - V1.0`。负数表示 V1.1 更快，正数表示 V1.1 更慢。

| 阶段            |   V1.0 平均（ms） |   V1.1 平均（ms） |   V1.1−V1.0（ms） |       V1.1 耗时变化 |
| --------------- | ----------------: | ----------------: | -----------------: | ------------------: |
| derive_eb       |           2.25433 |           2.09552 |           -0.15881 |             -7.045% |
| land_data       |           0.15904 |           0.16406 |           +0.00502 |             +3.156% |
| eb_zero         |           0.21757 |           0.77269 | **+0.55512** | **+255.145%** |
| uniform_eb      |           0.29337 |           0.28935 |           -0.00402 |             -1.370% |
| lorenzo         |           1.43061 |           1.39484 |           -0.03577 |             -2.500% |
| huffman         |           3.15694 |           3.12114 |           -0.03580 |             -1.134% |
| tile_eb         |           0.09894 |           0.08815 |           -0.01079 |            -10.906% |
| **total** | **7.61083** | **7.92570** | **+0.31487** |   **+4.137%** |
| 吞吐率          |     8.51270 GiB/s |     8.19290 GiB/s |     -0.31980 GiB/s |             -3.757% |

按平均总时间计算：

```text
V1.0 平均 total = 7.61083 ms
V1.1 平均 total = 7.92570 ms
V1.1 多用时间   = 0.31487 ms（相对 V1.0 增加 4.137%）
V1.0 相对 V1.1 加速 = 7.92570 / 7.61083 = 1.04137x
```

因此，在当前 RTX 3080 和当前代码条件下，**V1.1 没有比 V1.0 更快；V1.0
平均快约 1.041x**。如果从 V1.1 自身时间看，V1.0 减少约 3.973% 的耗时。

### 为什么 V1.1 总体更慢

决定性差异是 `eb_zero`：

```text
V1.0：0.21757 ms
V1.1：0.77269 ms
差值 ：+0.55512 ms
V1.1 用时是 V1.0 的 3.551x
```

V1.0 的融合 kernel 在一次扫描中直接把 zero-EB 的值和下标紧凑写出。V1.1 为了
生成确定的 bitmask/有序 side stream，在融合 kernel 后增加两次长度为 `n` 的
`thrust::exclusive_scan`，随后再运行 U/V 两个 compact kernel。因此 V1.1 的
`eb_zero` 虽然仍是融合流程的一部分，但计时区间包含更多全数组操作。

V1.1 在 derive、uniform、Lorenzo、Huffman 和 tile pack 上合计挽回约
0.24519 ms，扣除 land 增加的 0.00502 ms 后，其他阶段净挽回约 0.24017 ms；
仍不足以抵消 `eb_zero` 增加的 0.55512 ms，最终 total 净增 0.31487 ms。
阶段均值之和与 total 相差约 0.00008 ms，来自日志只打印三位小数。

除 `eb_zero` 外的差异都很小，而且 derive 是每个新进程中的第一个主要计时
kernel，波动较大。V1.0/V1.1 的 total 样本标准差分别为 0.66827/0.88938 ms，
范围分别为 6.825–10.995 ms 和 7.101–13.845 ms。100 个配对中 V1.0 总时间
较短 69 次，V1.1 较短 31 次；但是 `eb_zero` 在 100/100 个配对中均为 V1.0
更快。因此不能把约 1%–2.5% 的其他阶段均值差异单独解释成确定的算法优化，
而 zero-EB 的退化方向和源码变化是一致且稳定的。

### 正确性与压缩文件

200 个正式运行以及 10 个预热运行全部满足：

- dEb U/V 重建均为 `error count=0`；
- U/V 内存解压与文件解压均报告 `OK (bit-identical)`；
- 临界点均为 `orig=20929, decomp=20929, TP=20929, FP=0, FN=0`；
- 每个日志都恰好包含一组完整的 `COMPRESS_TIME`。

两版的 special/outlier/zero-EB 数量一致，但 V1.1 将 zero-EB 位置从逐项 index
改为完整 bitmask，文件布局不同：

| 指标                 |              V1.0 |              V1.1 |               V1.1 变化 |
| -------------------- | ----------------: | ----------------: | ----------------------: |
| `.cucpsz` 文件大小 |      29,284,800 B |      30,458,044 B | +1,173,244 B（+4.006%） |
| 压缩率               |          2.360269 |          2.269351 |                 -3.852% |
| U outlier / zero-EB  | 719,663 / 123,344 | 719,663 / 123,344 |                    相同 |
| V outlier / zero-EB  | 788,103 / 123,345 | 788,103 / 123,345 |                    相同 |

### 结果文件

- 中文摘要：`benchmark_v1_v11_100runs_20260902T223532Z/RESULT_ZH.md`
- 每次正式运行：`benchmark_v1_v11_100runs_20260902T223532Z/measured_results.csv`
- 每版平均值：`benchmark_v1_v11_100runs_20260902T223532Z/averages.csv`
- 版本差值：`benchmark_v1_v11_100runs_20260902T223532Z/comparison.csv`
- 完整日志：`benchmark_v1_v11_100runs_20260902T223532Z/logs/`

## 2026-08-24 历史测试说明

以下内容保留原报告的 5 次测试、profiling 和论文对应关系。它记录的是当时的
代码与测试条件，不能用来替换上面的当前 100 次平均结果。

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

| 指标                              |               结果 |
| --------------------------------- | -----------------: |
| 原始/解压 critical points         |    20,929 / 20,929 |
| matched / FP / FN / type mismatch | 20,929 / 0 / 0 / 0 |
| U 最大绝对误差                    |       0.0624847412 |
| V 最大绝对误差                    |       0.0622329712 |
| overall max relative EB           |    1.8089997539e-4 |

结论：V1.1 在该输入上保持全部 critical points，且与 V1 的恢复误差相同。

## 性能

先运行一次 warm-up，再独立运行 5 次。下面的 runs 已排序。

| 版本 | 5 次 GPU compression total (ms)        |    中位数 | 相对上一版 |
| ---- | -------------------------------------- | --------: | ---------: |
| V1   | 12.418, 13.467, 14.082, 14.198, 31.961 | 14.082 ms |          - |
| V1.1 | 12.194, 12.292, 12.484, 12.571, 12.608 | 12.484 ms |     1.128x |

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

| kernel    | duration | achieved occupancy | DRAM throughput | registers/thread |
| --------- | -------: | -----------------: | --------------: | ---------------: |
| compact U | 77.60 us |             56.93% |          14.53% |               16 |
| compact V | 77.50 us |             56.94% |          15.65% |               16 |

该 kernel 本身不重，但 V1.1 还需要两个长度为 `n` 的 scan。实测 zero-EB 阶段 0.797 ms，高于 V1 的 0.216 ms；bitmask 的主要价值是稳定、可逆的 side-stream 表达，而不是这一版的速度。

## 与论文对应

论文 Section 4.1 描述了 bitmask、population count、scan 和 ordered side stream。V1.1 只完成了这一思路的中间形态：它仍扫描两个长度为 `n` 的逐元素 offset 数组，而论文实现按 mask word 计数并扫描，后者在 V1.3 才出现。论文没有给出 V1.1 这样的版本编号，因此不能把论文最终性能直接归到该中间版本。

## 限制

- 单一 Ocean 数据集和单一误差参数，不代表其他尺寸或 zero-EB 密度。
- V1 的 5 次结果有一次 31.961 ms 离群值，因此使用中位数；V1.1 的代码变化本身不支持“derive EB 被优化”的结论。
- NCU replay 会显著放大程序内计时，报告只使用 NCU 的 per-kernel 指标，不使用 profiling run 的总时间。
