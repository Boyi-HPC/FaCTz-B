# V1.4 构建、运行与优化结论

## 构建和运行

测试环境：RTX 3080、CUDA 12.6.85、nvCOMP 5、Ocean `2400 x 3600`、`max_relative_eb=0.1`、Release、SM 86。论文为 `/home/boyi/FaCTz/ppopp27-summer-paper854.pdf`。

主 `.cu` 与 experiments 版本的 SHA-256 一致；融合、Huffman 和 ANS helper/bridge 来自最终 FaCTz 源码。

```bash
make -C src/cuda/V1.4
make -C src/cuda/V1.4 run CODEC=hf
make -C src/cuda/V1.4 run CODEC=ans
make -C src/cuda/V1.4 clean
```

target `cpszg_2d_v1_4` 已通过独立 Make 和全新 CMake 全版本构建。V1.4 通过 `Findnvcomp.cmake` 使用 `NVCOMP_ROOT` 或本机 `$HOME/.local/nvcomp-cu12/nvidia/libnvcomp`。

## 正确性

HF 和 ANS 两条路径均成功压缩、统一解压并通过 verifier：20,929 个 critical points 全匹配，FP/FN/type mismatch 都为 0。二者恢复 U/V 的最大误差与 V1.3 完全相同，确认 entropy codec 切换是 lossless 的，不改变拓扑和误差。

## 性能和压缩率

一次 warm-up 后各运行 5 次，runs 已排序。

| 路径 | 5 次 GPU compression total (ms) | 中位数 | ratio | 文件大小 |
|---|---|---:|---:|---:|
| V1.3 HF | 6.096, 6.204, 6.286, 6.294, 6.528 | 6.286 | 2.453 | 28,179,902 B |
| V1.4 HF | 6.219, 6.306, 6.353, 6.512, 6.544 | 6.353 | 2.453 | 28,179,902 B |
| V1.4 ANS | 18.422, 18.525, 18.601, 24.069, 27.954 | 18.601 | 2.393 | 28,884,256 B |

V1.4-HF 与 V1.3 基本持平，约慢 1.1%。当前 V1.4-ANS 压缩比 HF 慢 2.93x，并且文件大 2.5%。

GPU-resident 解压 CUDA-event 中位数：HF 3.879 ms，ANS 2.876 ms。ANS 解压快 1.35x，这是本次 ANS 路径明确成立的收益。

V1.4-HF 阶段中位数为 derive 3.651 ms、special classification 0.268 ms、word scan 0.224 ms、zero pack 0.107 ms、Lorenzo 0.716 ms、Huffman 1.357 ms。V1.4 没有改变 V1.3 的前端，所以 HF 总时间持平符合源码。

## 源码能够确认的变化

1. CLI 增加可选 `hf|ans`，但未指定时当前源码默认 `Huffman`。
2. ANS 对 16-bit quantization code 先减 radius、做 zigzag mapping，再拆成 U/V 的 low/high byte 四个 plane。
3. 四个 byte plane 交给 nvCOMP ANS，输出文件 header 记录 codec，统一解压器据此选择 Huffman 或 ANS workspace。
4. ANS one-shot timed 路径在每次调用中分配 byte planes、创建 events/configuration、压缩并下载 blob；源码同时存在 persistent workspace API，但该 V1.4 executable 调用的是 one-shot timed API。

关键位置：`cpszg_2d_v1.4.cu:1143`、`:1276`、`:1458`，`../gpu_ans_bridge.cu:69`、`:808`。

## Profiling 证据

ANS 前端 `split_u16_pair_kernel` 的 NCU basic 指标为：duration 104.26 us、achieved occupancy 79.78%、DRAM throughput 89.80%、20 registers/thread。正常运行中 ANS 阶段中位数是 13.267 ms，因此 byte split 只占很小部分，当前瓶颈在 nvCOMP ANS compression/configuration 路径，而不是 zigzag/byte-plane kernel。这是 profiling 支持的结论。

## 与论文对应及差异

- 论文 Section 4.3 和 Figure 5 对 zigzag 后拆 low/high byte plane 的描述与当前源码直接对应。
- 论文称 ANS 为默认 codec，并报告 ANS 通常比 Huffman 更快；当前 V1.4 默认是 HF，而且 Ocean 实测 ANS 压缩更慢、ratio 更低。这是明确的不一致，不能强行用论文最终结果解释当前中间版本。
- 合理推测：论文结果更接近使用 persistent ANS workspace、完整 batching 和进一步调优后的最终实现；当前 one-shot V1.4 的分配/configuration 和 nvCOMP 路径尚未达到论文状态。该句是源码与结果形成的推测，不是论文直接事实。

## 限制

- ANS 前两次正式运行明显较慢，5 次中位数虽可抗离群，但稳定 benchmark 应固定 GPU clocks 并增加到至少 20 次。
- 只测试 Ocean；论文所述 high-byte plane 优势可能随 quantization-code 分布改变。
- 当前报告没有逐个展开 nvCOMP 内部 proprietary kernel，只对可控 byte-split kernel 和 ANS 总阶段计时。
