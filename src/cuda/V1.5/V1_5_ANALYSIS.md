# V1.5 GPU-resident 构建、运行与性能结论

## 版本来源和构建

V1.5 是 FaCT `apps` 中实际使用的 offline compressor/decompressor，放入独立版本目录后保持源码不变：

- `cpszg_2d.cu` 与 `/home/boyi/FaCTz/apps/cpszg_2d.cu` SHA-256 一致：`12f058446ce24fc053df4c48c3365a42696b09a2842b4b049b4e8e1e36772825`；
- `decpszg_2d.cu` 与 `/home/boyi/FaCTz/apps/decpszg_2d.cu` SHA-256 一致：`8ad7b7d27ccad94f3e54c44bf6c68431ac3d3493cb7bac6f64f46b4ad2b23464`；
- `fused_ebzero_kernel.cuh` 与 FaCT app 使用的 helper SHA-256 一致。

生成的 targets：

- `cpszg_2d_v1_5`；
- `decpszg_2d_v1_5`；
- 复用公共 `factz_verify_cpszg_2d` verifier。

```bash
make -C src/cuda/V1.5
make -C src/cuda/V1.5 run CODEC=hf
make -C src/cuda/V1.5 verify CODEC=hf
make -C src/cuda/V1.5 run CODEC=ans
make -C src/cuda/V1.5 verify CODEC=ans
make -C src/cuda/V1.5 clean
```

测试环境：RTX 3080、CUDA 12.6.85、Ocean `2400 x 3600`、`max_relative_eb=0.1`、Release、SM 86。HF/ANS 使用相同输入、GPU、文件格式和数值参数。

## 正确性

HF 和 ANS 均完成压缩、独立 V1.5 解压和 verifier 检查：

| 指标 | HF | ANS |
|---|---:|---:|
| 原始/decompressed critical points | 20,929 / 20,929 | 20,929 / 20,929 |
| matched | 20,929 | 20,929 |
| FP / FN / type mismatch | 0 / 0 / 0 | 0 / 0 / 0 |
| U max relative error | `1.8089997539e-04` | `1.8089997539e-04` |
| V max relative error | `1.7568627708e-04` | `1.7568627708e-04` |

HF 和 ANS 恢复出的 U/V 文件逐字节相同。V1.3 历史解压器也能读取 V1.5-HF container，输出与 V1.5 解压器逐字节相同并通过 verifier；V1.3 不支持 ANS。

## 测试方法和边界

- V1.4 先运行一个独立 warm-up 进程，再运行 5 个正式进程；其程序只报告各阶段 CUDA-event 时间之和。
- V1.5 每个进程在正式数据前执行同进程 synthetic warm-up，再运行正式数据；同时报告 kernel sum 和 GPU-resident E2E。
- compression kernel sum 只包含 derive EB、special classification、word scan、zero pack、Lorenzo 和 entropy kernels。
- V1.5 GPU-resident compression E2E 边界为 device inputs/workspace ready 到 compressed device segments ready。
- V1.5 GPU-resident decompression E2E 边界为 compressed device segments ready 到 recovered U/V ready on GPU。

V1.4 和 V1.5 的 warm-up 位置不完全相同，因此 kernel-sum speedup 是“两个 executable 当前 benchmark 路径”的结果，不等价于同一个 kernel 的纯代码 speedup。

## Compression 结果

5 次结果已排序，speedup=`V1.4 median / V1.5 median`。

| Codec | V1.4 GPU kernel sum（ms） | V1.5 GPU kernel sum（ms） | 中位数 V1.4→V1.5 | speedup |
|---|---|---|---:|---:|
| HF | 6.145, 6.169, 6.482, 6.487, 6.781 | 2.430, 2.464, 2.735, 2.776, 4.959 | 6.482→2.735 | **2.37x** |
| ANS | 16.491, 17.034, 17.850, 19.698, 29.658 | 3.728, 3.798, 3.808, 3.820, 3.840 | 17.850→3.808 | **4.69x** |

阶段中位数：

| 阶段（ms） | V1.4 HF | V1.5 HF | V1.4 ANS | V1.5 ANS |
|---|---:|---:|---:|---:|
| derive EB | 3.544 | 0.730 | 3.722 | 0.756 |
| special classification | 0.240 | 0.227 | 0.270 | 0.226 |
| word scan | 0.258 | 0.034 | 0.291 | 0.034 |
| zero pack | 0.111 | 0.077 | 0.125 | 0.077 |
| Lorenzo | 0.774 | 0.349 | 0.725 | 0.350 |
| entropy | 1.334 | 1.356 | 12.604 | 2.338 |

V1.5 自身的完整 GPU-resident E2E：

| Codec | 5 次 CUDA-event（ms） | 中位数 | 文件大小 | ratio |
|---|---|---:|---:|---:|
| HF | 4.774, 5.141, 5.176, 5.358, 6.078 | 5.176 | 28,179,902 B | 2.452812 |
| ANS | 4.009, 4.022, 4.042, 4.055, 4.061 | 4.042 | 28,884,256 B | 2.392999 |

ANS 的 V1.5 resident compression E2E 比 HF 快 `1.28x`，但文件大约大 `2.5%`。ANS 的 kernel sum 比 HF 高，是因为 HF kernel sum 不包含 CPU codebook 构造造成的 GPU idle/调用时间；GPU-resident E2E 包含了这段真实流水线等待，因此它才适合比较最终 codec 路径。

## Decompression 结果

V1.5 5 次 GPU-resident CUDA-event：

| Codec | 5 次结果（ms） | 中位数 | HF/ANS speedup |
|---|---|---:|---:|
| HF | 3.705, 3.731, 3.747, 3.748, 3.761 | 3.747 | 1.00x |
| ANS | 2.683, 2.704, 2.710, 2.732, 2.746 | 2.710 | **1.38x** |

V1.4 报告中的统一解压器与 V1.5 `decpszg_2d.cu` 本来就是同一份 app 源码，所以不存在新的 V1.4→V1.5 decoder kernel 差异；V1.5 的意义是把实际 app decoder 纳入明确的版本 target。

作为历史路径参考，同一个 V1.5-HF 文件还使用 V1.3 standalone decoder 运行了 5 次：

| 路径 | 边界 | 5 次结果（ms） | 中位数 |
|---|---|---|---:|
| V1.3 historical | file read + allocations/H2D + decode + D2H，不含写盘 | 309.553, 311.059, 357.290, 358.800, 390.915 | 357.290 |
| V1.5 HF reconstructed comparable path | file read + prepare + resident wall + host alloc + D2H，不含写盘 | 46.853, 48.167, 48.399, 63.550, 66.146 | 48.399 |

按应用当前报告边界，观察到约 `7.38x`；但 V1.5 在同进程先 warm-up，而 V1.3 timer 包含该进程第一次 CUDA 工作，因此不能把 7.38x 全部归因于 decoder kernels。

## 性能差异的原因

### 源码直接确认

1. app 完整保留 V1.4 函数体并重命名为 `sz_compress_cp_preserve_2d_offline_gpu_legacy`；main 实际调用新的 GPU-resident 函数。
2. `CpCompressionWorkspace2D` 在正式计时前分配 field buffers、bitmasks、zero-EB arrays、CUB scratch、events 和 Huffman/ANS workspace。
3. V1.4 在 derive、classification、scan 和 zero pack 后逐段同步；V1.5 连续提交这些阶段，等 codec 已经排空 stream 后统一读取 events。
4. V1.4 使用 `thrust::exclusive_scan`；V1.5 使用带预分配 scratch 的 `cub::DeviceScan::ExclusiveSum`。
5. V1.4 ANS 使用 one-shot timed API，每次创建 byte planes/events/configuration 并下载 blobs；V1.5 使用 persistent ANS workspace 和 device-resident output。
6. V1.5 Huffman/ANS compressed blobs 保持在 GPU，D2H 和文件写入移到 GPU-resident E2E 之后。
7. V1.5 decoder 使用 word-level zero-EB popcount/scan，只扫描 `ceil(n/32)` 个 mask words，并成对恢复 U/V；V1.3 对 U/V 分别展开并扫描 `n` 个 flags。
8. V1.5 decoder 预分配 codec、CUB 和输出 workspace，移除 U/V 之间显式的 `cudaDeviceSynchronize`，D2H/写盘独立计时。

### 运行结果支持

- Huffman kernel 中位数 `1.334→1.356 ms`，基本不变，支持“HF 底层 kernel 没有实质重写，收益主要在流水线外围”的结论。
- word scan `0.258→0.034 ms`，支持预分配 CUB scratch 和减少 Thrust 调度/同步开销有效。
- ANS entropy `12.604→2.338 ms`，支持 persistent workspace、同进程 warm-up 和 device-resident API 消除了 V1.4 one-shot 路径的大量成本。
- V1.5 ANS decompression resident E2E 比 HF 快 1.38x，但其 prepare 时间更高；当把文件读取、配置、host allocation 和 D2H 算入时，HF/ANS 总路径接近。

### 不能过度解释

- derive/classification/Lorenzo 使用的核心 kernel 和 launch geometry 与 V1.4 相同。derive `3.544→0.730 ms` 和 Lorenzo `0.774→0.349 ms` 的差异很大程度受到 V1.5 同进程 warm-up、GPU clocks 和旧路径同步位置影响，不能声称这些 kernel 获得了对应倍数的代码优化。
- V1.5 的主要定位是 GPU residency、持久 workspace 和可靠 benchmark boundary，而不是新的拓扑保持算法。

## 限制

- 每组只有 5 次，未固定 GPU clocks，HF 和 V1.4 ANS 均出现离群值；更严格结果应至少运行 20 次并固定 clocks。
- V1.4 缺少与 V1.5 完全相同的 GPU-resident E2E boundary，因此这里只报告 kernel-sum speedup，并明确保留 warm-up 差异。
- V1.3 historical decoder 没有同进程 warm-up 或独立 CUDA-event boundary，7.38x 不能作为纯 decoder-kernel speedup。
- 仅测试 Ocean 数据集；codec 和 special-value 分布变化可能改变结果。
