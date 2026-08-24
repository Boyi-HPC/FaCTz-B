# V1.3 构建、运行与优化结论

## 构建和测试

测试环境：RTX 3080、CUDA 12.6.85、Ocean `2400 x 3600`、`max_relative_eb=0.1`、Release、SM 86。论文为 `/home/boyi/FaCTz/ppopp27-summer-paper854.pdf`。

压缩、解压两个 `.cu` 均与 experiments 中的 V1.3 版本 SHA-256 一致；融合 helper 与最终 FaCTz 源码一致。历史解压源码为 `decpszg_2d_v1.3.cu`，SHA-256 是 `1967411ca2c1e00ec05d667842e97ac429904a4d67f47683b8fbe63a11f72bad`。

```bash
make -C src/cuda/V1.3
make -C src/cuda/V1.3 run
make -C src/cuda/V1.3 clean
```

targets `cpszg_2d_v1_3` 和 `decpszg_2d_v1_3` 已通过独立 Make 构建、全新 CMake 全版本构建和实际 GPU 运行。

现在 V1.3 同时提供：

- `cpszg_2d_v1_3`：压缩器；
- `decpszg_2d_v1_3`：该版本原有的独立解压器。

`make` 会构建两个 target，`make run` 会顺序执行压缩和解压，输出 `U.dat.cucpsz`、`U.dat.out` 和 `V.dat.out`。也可以分别执行 `make compress` 和 `make decompress`。

## 正确性

V1.3 历史解压器和 verifier 得到：原始/解压 critical points 都为 20,929，matched=20,929，FP=0，FN=0，type mismatch=0。U/V 的最大相对误差（最大绝对误差除以原始值域）分别为 `1.8089997539e-04` 和 `1.7568627708e-04`。历史解压器的 U、V 输出还与通用解压器输出逐字节一致，说明新增 target 使用的确实是兼容的 V1.3 文件格式和解码路径。

实际验证命令：

```bash
build/bin/factz_verify_cpszg_2d \
  data/uf.dat data/vf.dat \
  /tmp/factz-v13-historical-dec/U.dat.out \
  /tmp/factz-v13-historical-dec/V.dat.out \
  2400 3600
```

## 性能

一次 warm-up 后运行 5 次，runs 已排序。

| 版本 | 5 次 GPU compression total (ms) | 中位数 | V1.2/V1.3 |
|---|---|---:|---:|
| V1.2 | 5.701, 5.909, 6.244, 6.400, 6.674 | 6.244 ms | 0.993x |
| V1.3 | 6.096, 6.204, 6.286, 6.294, 6.528 | 6.286 ms | -0.7% |

通用解压器测得的 GPU-resident 解压 5 次 CUDA-event 中位数为 3.740 ms，V1.2 为 3.753 ms，属于基本持平。

新增的历史 V1.3 解压器也在一次 warm-up 后正式运行 5 次：

| 计时范围 | 5 次结果（s） | 中位数 |
|---|---|---:|
| 文件读取 + 解码流程 + D2H，不含输出文件写入 | 0.398857, 0.358659, 0.332097, 0.335668, 0.361292 | 0.358659 s |

历史解压器按进程测量完整路径，包含文件读取、内存分配、Huffman/Lorenzo 解码和 D2H；3.740 ms 则是预加载数据后的 GPU-resident CUDA-event 时间。两者计时边界不同，不能据此计算 speedup。历史解压器另外报告输出文件写入中位数 `0.042416 s`。

| V1.3 阶段 | 中位数 |
|---|---:|
| derive EB | 3.601 ms |
| fused special classification + tile min | 0.251 ms |
| mask word count + scan | 0.220 ms |
| ordered zero-value pack | 0.108 ms |
| Lorenzo | 0.745 ms |
| Huffman kernels | 1.324 ms |

V1.2 的 land + zero-EB + tile reduction 中位数合计 1.156 ms；V1.3 对应三个融合/紧凑阶段合计 0.579 ms，子流水线约快 2.00x，减少约 0.577 ms。但 V1.3 的 derive-EB 本批次高 0.695 ms，最终总时间没有快于 V1.2。因此结论应是“融合局部生效，总体持平”，而不是“V1.3 整体加速”。

## 源码能够确认的优化

1. `kernel_classify_specials_and_tile_eb` 以一个 32x8 block 负责一个 32x32 tile，在一次 field sweep 中完成 land 检测、zero-EB 分类/ID 替换、三个 bitmask 输出，以及 U/V tile minimum。
2. 使用 `__ballot_sync` 每 warp 产生 32-bit mask，避免每元素 atomic；仅由 lane 0 将 ballot 合并到线性 bitmask。
3. 使用 warp shuffle 和 8 个 shared-memory warp minima 完成 U/V tile reduction。
4. `kernel_mask_word_counts_pair_and_land` 对每个 32-bit word 做 `__popc`，只扫描 `ceil(n/32)` 个 word offset，不再扫描两个长度为 `n` 的逐元素 flag 数组。
5. `kernel_compact_zeroeb_values_pair_by_word` 由一个 warp 处理一个 mask word，用 lower-bit popcount 算稳定 rank，并在同一 kernel 中收集 U/V。

关键位置：`fused_ebzero_kernel.cuh:24`、`:86`、`:129`、`:179`，`cpszg_2d_v1.3.cu:987`、`:1014`、`:1050`。

## Profiling 证据

融合 classification kernel 的 NCU basic 指标：duration 231.39 us、achieved occupancy 96.05%、DRAM throughput 50.51%、34 registers/thread、block 256 threads。高 occupancy 和单次约 0.23 ms 与正常运行的 0.251 ms 一致，支持融合 pass 工作正常；34 registers/thread 是它相对简单 kernel 的资源代价。

## 与论文对应

V1.3 是四个中间版本里最直接对应论文 Section 4.1 的版本：single fused kernel、`__ballot_sync` bitmask、warp/block tile minimum、移除三次 full-field reread，以及 bitmask population count + scan 都能在源码中逐项找到。论文的版本命名并不包含 V1.3，因此这里只做实现特征对应，不声称论文结果就是该 executable 的结果。

## 限制

- derive-EB 在 V1.2/V1.3 间没有对应的实质优化，阶段差异受 GPU 频率和进程级波动影响。
- 5 次总时间显示融合节省被其他阶段波动抵消；需要更多轮次或固定 GPU clocks 才能区分小于 1% 的差异。
- NCU replay 序列化并重放 kernel，不能使用 profile run 内打印的总时间。
- 历史解压器的端到端计时包含初始化和 I/O；若要和 GPU-resident 指标比较，需要在该文件内部增加相同边界的 CUDA-event 计时。
