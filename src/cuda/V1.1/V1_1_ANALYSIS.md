# V1.1 Bitmask 验证、构建与性能结论

### V1.1 的版本定位

V1.1 的首要目的不是在当前实现中立即超过 V1.0 的速度或压缩率，而是先验证 zero-EB bitmask 路径的正确性，并建立后续融合 kernel 和压缩率优化所需的确定性数据表示。本版本重点验证以下链路：

- bitmask 中的每个 bit 能否准确表示对应网格位置是否为 zero-EB；
- bitmask、前缀和与按行主序排列的 zero-EB value side stream 能否一一对应；
- bitmask 写入文件、读回和解压回填后，是否仍能保持 U/V 位级一致和临界点不变。

为了先获得一条容易核对的参考路径，V1.1 当前采用“fused kernel 生成 flag/bitmask → U/V 各一次 `exclusive_scan` → U/V 各一次 compact kernel”的实现（[调用链](cpszg_2d_v1.1.cu#L978-L1006)）。这一实现刻意优先保证 bitmask 语义和 side stream 顺序正确，它是后续优化的正确性基线，而不是最终的高性能紧凑实现。

后续工作可在不改变已验证文件语义的前提下，将 flag 生成、块内排名和 value compact 融合，减少当前两次全长前缀和与两次额外全数组遍历。同时，已验证的 bitmask 格式为后续对位置模式做无损压缩，或在稀疏 index 与压缩 bitmask 之间自适应选择，提供了稳定输入。

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

V1.0 的融合 kernel 在一次扫描中通过 `atomicAdd` 直接获得紧凑输出位置，并写出 zero-EB 的 value 和 index（[源码](../V1/fused_ebzero_kernel.cuh#L27-L44)）。它在 fused kernel 返回时已经完成 zero-EB side stream 的紧凑化，但由于原子加的执行顺序不固定，输出顺序也不是确定的行主序。

V1.1 的目标是验证确定的 bitmask 和有序 value side stream。它在 fused kernel 中生成逐元素 flag、bitmask 和 count，然后对 U/V 的 `n` 个 flag 各执行一次 `thrust::exclusive_scan`，最后再启动 U/V 两个 compact kernel（[调用链](cpszg_2d_v1.1.cu#L978-L1006)，[compact kernel](cpszg_2d_v1.1.cu#L412-L425)）。前缀和为每个 zero-EB 位置生成确定的行主序 rank，compact kernel 再按该 rank 写入 value。因此，V1.1 当前的 `eb_zero` 计时不仅包含融合 kernel，还包含两次全长前缀和与两次额外全数组遍历。

这一源码路径差异直接对应到测量结果：V1.1 的 `eb_zero` 由 `0.21757 ms` 增加到 `0.77269 ms`，净增 `0.55512 ms`，且 100 个配对中都是 V1.0 更快。这说明退化方向与新增全数组操作一致，是当前参考实现的稳定成本，而不是偶然的均值波动。

除 `eb_zero` 外，两版在其余计时阶段走的是同一类计算路径：`derive_eb` 调用同一个 `derive_eb_offline_v2`，`land_data` 执行相同的逐点处理，`uniform_eb` 使用相同的 tile 最小 EB kernel，Lorenzo 使用同一套二维 tile 接口，Huffman 链接同一个 `factz_v1_huffman`，Tile-EB 也使用相同的 pack kernel（[V1.0 调用段](../V1/cpszg_2d_v1.cu#L880-L1039)，[V1.1 对应调用段](cpszg_2d_v1.1.cu#L940-L1112)，[共用 Huffman 目标](../CMakeLists.txt#L52-L68)）。这些阶段没有像 zero-EB 路径那样新增或删除全数组扫描，因此其均值差异没有对应的源码机制可以认定为优化。尤其 `derive_eb` 是每个新进程中的第一个主要计时 kernel，容易同时受到首次启动、GPU 频率和调度状态影响；Tile-EB 的绝对耗时很小，相对百分比也容易被放大。

数据与这一判断一致：V1.1 在 `derive_eb`、`uniform_eb`、Lorenzo、Huffman 和 Tile-EB 上测得的负差值合计为 `0.24519 ms`，再扣除 `land_data` 增加的 `0.00502 ms`，表面上的其他阶段净抵消量为 `0.24017 ms`。这个数字只表示本轮样本均值对 total 的抵消量，不表示这些相同路径获得了确定的算法加速。它仍不足以抵消 `eb_zero` 增加的 `0.55512 ms`，所以 total 最终净增 `0.31487 ms`。按日志中已四舍五入的三位小数阶段值直接求和，会与 total 相差约 `0.00008 ms`。

从样本分布看，V1.0/V1.1 的 total 样本标准差分别为 `0.66827/0.88938 ms`，范围分别为 `6.825–10.995 ms` 和 `7.101–13.845 ms`；100 个配对中，V1.0 total 较短 69 次，V1.1 较短 31 次，说明总时间仍叠加了运行波动。`eb_zero` 则在 `100/100` 个配对中均为 V1.0 更快，并且退化方向与新增全数组操作一致。因此，本轮结果能明确归因的是 zero-EB 参考路径的额外工作，而不是把其余阶段约 `1%–2.5%` 的小幅均值变化，或 Tile-EB 在极小基数上的相对变化，解释成新的算法优化。

#### 对应代码

V1.0：在一个 fused kernel 中通过 `atomicAdd` 得到紧凑输出位置，直接写出
`value + index`（[查看源码](../V1/fused_ebzero_kernel.cuh#L27-L44)）：

```cpp
const size_t i =
    static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
if (i >= n) return;

if (eq_eb_U[i] == Eq{0}) {
    const uint32_t out = atomicAdd(zero_U_count, 1u);
    zero_U_values[out] = data_U[i];
    zero_U_indices[out] = static_cast<uint32_t>(i);
    eq_eb_U[i] = replacement_id;
}
if (eb_U[i] <= threshold) eb_U[i] = replacement_eb;

if (eq_eb_V[i] == Eq{0}) {
    const uint32_t out = atomicAdd(zero_V_count, 1u);
    zero_V_values[out] = data_V[i];
    zero_V_indices[out] = static_cast<uint32_t>(i);
    eq_eb_V[i] = replacement_id;
}
if (eb_V[i] <= threshold) eb_V[i] = replacement_eb;
```

V1.1 第一步：fused kernel 生成逐元素 flag、bitmask 和 count
（[查看源码](fused_ebzero_kernel.cuh#L25-L45)）：

```cpp
const size_t i =
    static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
if (i >= n) return;

const bool zero_u = eq_eb_U[i] == Eq{0};
const bool zero_v = eq_eb_V[i] == Eq{0};
zero_U_flags[i] = zero_u;
zero_V_flags[i] = zero_v;

if (zero_u) {
    atomicOr(zero_U_mask + (i >> 5), 1u << (i & 31u));
    atomicAdd(zero_U_count, 1u);
    eq_eb_U[i] = replacement_id;
}
if (eb_U[i] <= threshold) eb_U[i] = replacement_eb;

if (zero_v) {
    atomicOr(zero_V_mask + (i >> 5), 1u << (i & 31u));
    atomicAdd(zero_V_count, 1u);
    eq_eb_V[i] = replacement_id;
}
if (eb_V[i] <= threshold) eb_V[i] = replacement_eb;
```

V1.1 第二步：对 U/V 的 `n` 个 flags 分别执行前缀和，再分别启动 compact
kernel（[查看调用点](cpszg_2d_v1.1.cu#L995-L1006)）：

```cpp
thrust::exclusive_scan(thrust::device,
    ebIsZero_U_indices, ebIsZero_U_indices + num_elements,
    ebIsZero_U_indices);
thrust::exclusive_scan(thrust::device,
    ebIsZero_V_indices, ebIsZero_V_indices + num_elements,
    ebIsZero_V_indices);

kernel_compact_zeroeb_values<T><<<zeb_grid, ZEB_BLK>>>(
    dU, d_zeroeb_mask_U, ebIsZero_U_indices,
    ebIsZero_U_data, num_elements);
kernel_compact_zeroeb_values<T><<<zeb_grid, ZEB_BLK>>>(
    dV, d_zeroeb_mask_V, ebIsZero_V_indices,
    ebisZero_V_data, num_elements);
```

V1.1 的 compact kernel 再遍历一次完整数组，用 mask 判断当前元素是否入选，用
前缀和结果决定紧凑输出位置（[查看源码](cpszg_2d_v1.1.cu#L412-L425)）：

```cpp
size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
if (i >= n) return;
if ((mask[i >> 5] >> (i & 31u)) & 1u) {
    zeroeb_values[offsets[i]] = data[i];
}
```

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

当前 V1.1 直接保存两个未压缩的完整 bitmask，每个分量均占 `8,640,000 / 8 = 1,080,000 bytes`。V1.0 对 zero-EB 位置保存逐项 `uint32_t` index，两个分量合计为：

```text
(123,344 + 123,345) * 4 = 986,756 bytes
```

因此，未压缩 bitmask 相对稀疏 index 恰好增加：

```text
2 * 1,080,000 - 986,756 = 1,173,244 bytes
```

这与表中完整文件的增量完全一致。所以，当前文件变大不是 bitmask 验证失败，而是 V1.1 暂时将未压缩完整 bitmask 作为参考格式的可量化代价。本版本已完成的是 bitmask 位置语义、有序 value side stream 和文件回填链路的正确性验证；性能与压缩率收益需由后续 fused kernel、bitmask 无损压缩或稀疏/稠密表示自适应选择来实现。
