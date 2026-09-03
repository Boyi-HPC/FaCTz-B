# Naive CUDA cpSZ-Offline

本目录包含未经优化的 CUDA 流程及其专用 GPU Huffman 桥接代码。共享的 CUDA
配置和定制 Lorenzo 支持位于上一级 `src/cuda/` 目录中。编译产物会写入仓库
顶层的 `build/` 目录。

在本目录中执行：

```bash
make build
make run
```

`make run` 默认使用项目提供的 `2400 × 3600` float32 U/V 场，误差界为 `0.1`。
可以通过 Make 变量指定其他输入、尺寸和误差界：

```bash
make run U=/path/to/u.dat V=/path/to/v.dat R1=2400 R2=3600 EB=0.1
```

常用的编译配置变量包括 `CUDACXX`、`CUDA_ARCHITECTURES`、`BUILD_TYPE`、
`BUILD_DIR` 和 `JOBS`。例如：

```bash
make build CUDACXX=/path/to/nvcc CUDA_ARCHITECTURES=80 JOBS=8
```

生成的可执行文件为 `build/bin/cpszg_2d_no_opt_speed`。程序运行时产生的调试
数组会单独保存在 `build/runs/naive` 目录中。

## 压缩流水线

Naive 版本的主压缩函数是
[`sz_compress_cp_preserve_2d_offline_gpu`](./cpszg_2d_no_opt_speed.cu#L734)。
它处理一个 `r1 × r2` 的二维向量场，令

```text
n = r1 * r2
U[i] = 水平方向分量
V[i] = 垂直方向分量
```

整体流程按执行顺序分为：

1. `derive_eb`
2. `exception process`
   1. `land_data`
   2. `eb_zero`
3. `lorenzo`
4. `huffman`

主数据流和三条旁路数据流如下：

```text
                     ┌─ land bitpack ───────────────────────────────┐
U, V ── derive_eb ── exception process ── Lorenzo ── Huffman ── .cucpsz
          │                 │                 │          │
          │                 └─ zero-EB 原值/下标 ────────┤
          └─ dEb_U/V、eq_dEb_U/V                        │
                                            └─ outlier 原值/下标 ───┘
```

这里有两类容易混淆的 ID：

| 数组                       | 含义                                   | 产生阶段      |
| -------------------------- | -------------------------------------- | ------------- |
| `eq_dEb_U`, `eq_dEb_V` | EB 的指数 ID，用来重建每个元素的误差界 | `derive_eb` |
| `eq_U`, `eq_V`         | 数据经过 Lorenzo 预测和量化后的量化码  | `lorenzo`   |

另外有三类不进入普通量化码的数据：

| 旁路数据             | 保存内容                                          | 解压时机                     |
| -------------------- | ------------------------------------------------- | ---------------------------- |
| land bitpack         | `U[i] == 0 && V[i] == 0` 的位置，每个元素 1 bit | 最后把 U、V 都恢复成精确的 0 |
| zero-EB 列表         | EB ID 为 0 的元素下标和原始浮点值                 | Lorenzo 解压后覆盖           |
| Lorenzo outlier 列表 | 无法在量化半径内表示的元素下标和原始浮点值        | Lorenzo 解压前放入预测缓冲区 |

## 1. derive_eb

实现位置：
[`derive_eb_offline_v2`](./cpszg_2d_no_opt_speed.cu#L284)，调用位置：
[`RUN_DERIVE_EB`](./cpszg_2d_no_opt_speed.cu#L810)。

### 输入

- `dU`, `dV`：设备端的原始二维向量场，按行展开成长度为 `n` 的数组。
- `r1`, `r2`：行数和列数。
- `max_pwr_eb`：允许使用的最大相对误差界。
- 固定最小量化单位 `threshold = 2^-20`。

### 处理过程

一个 `32 × 8` CUDA block 先把 U、V 加载到 shared memory。相邻 block
之间保留 2 个元素的重叠区域，因为每个输出点的 EB 依赖周围三角形。

```cpp
__shared__ T buf_U[TileDim_Y][TileDim_X + 1];
__shared__ T buf_V[TileDim_Y][TileDim_X + 1];
__shared__ T per_cell_eb_L[TileDim_Y][TileDim_X + 1];
__shared__ T per_cell_eb_U[TileDim_Y][TileDim_X + 1];

int row = blockIdx.y * (blockDim.y - 2) + threadIdx.y;
int col = blockIdx.x * (blockDim.x - 2) + threadIdx.x;

if (row < r1 && col < r2) {
    buf_U[threadIdx.y][threadIdx.x] = dU[row * r2 + col];
    buf_V[threadIdx.y][threadIdx.x] = dV[row * r2 + col];
}
__syncthreads();
```

每个网格单元被分成上下两个三角形。函数
[`gpu_max_eb_to_keep_position_and_type`](./cpszg_2d_no_opt_speed.cu#L207)
根据三个顶点的 `(U,V)` 计算仍能保持拓扑位置和类型的最大相对 EB：

```cpp
per_cell_eb_U[localRow][localCol] =
    gpu_max_eb_to_keep_position_and_type(
        U00, U01, U11, V00, V01, V11);

per_cell_eb_L[localRow][localCol] =
    gpu_max_eb_to_keep_position_and_type(
        U00, U10, U11, V00, V10, V11);
```

一个顶点会影响周围多个三角形，因此代码取与该顶点相关的 6 个三角形
误差界的最小值。这样选择的是所有局部约束中最严格的一个：

```cpp
T localmin = max_pwr_eb;
localmin = min(localmin, per_cell_eb_U[localRow][localCol]);
localmin = min(localmin, per_cell_eb_L[localRow][localCol]);
localmin = min(localmin, per_cell_eb_U[localRow + 1][localCol]);
localmin = min(localmin, per_cell_eb_L[localRow][localCol + 1]);
localmin = min(localmin, per_cell_eb_U[localRow + 1][localCol + 1]);
localmin = min(localmin, per_cell_eb_L[localRow + 1][localCol + 1]);
```

随后把相对 EB 分别乘以 `|U[i]|` 和 `|V[i]|`，得到两个分量各自的绝对
EB。绝对 EB 再向下量化到 `threshold × 4^id`：

```cpp
T threshold = (T)(1.0 / (1 << 20));
T temp = local_relative_eb * fabs(component_value);

if (temp <= threshold) {
    temp = 0;
    id = 0;
}
if (temp > threshold) {
    id = log2(temp / threshold) / 2.0;
    temp = (T)(1ULL << (2 * id)) * threshold;
}

dEb_component[i] = temp;
eq_dEb_component[i] = id;
```

数学上近似为：

```text
id = floor(log2(raw_eb / threshold) / 2)
stored_eb = 2^(2*id) * threshold = 4^id * threshold
```

例如 `threshold = 2^-20`，若 `raw_eb = 0.01`，则 `id = 6`，实际使用的
EB 是 `2^12 × 2^-20 = 2^-8 = 0.00390625`。它不大于原始允许值，因此不会
放宽拓扑约束。

### 输出

- `dEb_U[i]`, `dEb_V[i]`：Lorenzo 直接使用的逐元素浮点 EB。
- `eq_dEb_U[i]`, `eq_dEb_V[i]`：对应的逐元素 EB 指数 ID；主函数中
  `eq_dEb_U` 的变量名是 `eq_dEb`。
- `id == 0` 表示该位置的 EB 太小，必须进入后面的 `eb_zero` 异常流程。

当前 kernel 签名中的 `dEb`（调用处为 `eb_gpu`）没有在 kernel 内写入；真正
参与后续处理的是上面四个 U/V 数组。边界写零的条件也只实际覆盖首行或首列，
详见[边界分支](./cpszg_2d_no_opt_speed.cu#L371)。

### Naive 性能特征

每个元素要读取邻域、计算两个三角形约束、做多次 `sqrt`、除法和 `log2`，
同时还保存两份浮点 EB 和两份 EB ID。后续阶段仍会再次完整扫描这些数组。

## 2. exception process

异常处理位于 `derive_eb` 和 Lorenzo 之间。它先处理陆地位置，再处理零 EB
位置。两者用途不同：land bitpack 同时描述 U/V 都为零的位置；zero-EB
列表分别描述 U 或 V 无法使用普通 EB 量化的位置。

### 2.1 land_data

实现位置：[`land bitpack 构建`](./cpszg_2d_no_opt_speed.cu#L798) 和
[`DEAL_WITH_LAND_DATA`](./cpszg_2d_no_opt_speed.cu#L829)。

#### 输入

- 原始 `U`, `V` 和设备数组 `dU`, `dV`。
- `max_pwr_eb` 与 `threshold`。

#### 处理过程

首先在 CPU 上为所有 `U[i] == 0 && V[i] == 0` 的位置建立 bitpack：

```cpp
size_t land_bitpack_bytes = (num_elements + 7) / 8;
uint8_t* h_bp = new uint8_t[land_bitpack_bytes]();

for (size_t i = 0; i < num_elements; i++) {
    if (U[i] == (T)0 && V[i] == (T)0)
        h_bp[i / 8] |= (uint8_t)(1u << (i % 8));
}
```

`i / 8` 选择字节，`i % 8` 选择该字节中的 bit。例如 `i = 13` 时使用第 1
个字节的第 5 bit，对应掩码 `0010 0000`，即 `0x20`。bitpack 只需要
`ceil(n/8)` 字节。

然后 Thrust 对全部元素做一次扫描。对于陆地位置，把 U、V 的 EB 都替换为
由 `max_pwr_eb` 离散化得到的最大 EB：

```cpp
thrust::for_each(idx_first, idx_last, [=] __device__(size_t i) {
    if (dU[i] == 0 && dV[i] == 0) {
        int id = log2(max_pwr_eb / threshold) / 2.0;
        eq_dEb[i]   = id;
        eq_dEb_V[i] = id;
        dEb_U[i] = (T)(1ULL << (2 * id)) * threshold;
        dEb_V[i] = (T)(1ULL << (2 * id)) * threshold;
    }
});
```

这些位置的精确答案已知为 `(0,0)`，所以主量化路径可以使用较宽松的 EB；
解压最后再根据 bitpack 把 U、V 强制恢复为精确的 0。

#### 输出

- `d_land_bitpack`：每个元素 1 bit 的陆地标记，最终写入压缩文件。
- 修改后的 `dEb_U/V` 和 `eq_dEb_U/V`。

#### Naive 性能特征

bitpack 先在 CPU 构造，再复制到 GPU；异常处理又对 `dU/dV` 做一次完整的
Thrust 扫描。写文件前 bitpack 还会从 GPU 复制回 CPU。

### 2.2 eb_zero

实现位置：[`DEAL_WITH_EBZERO`](./cpszg_2d_no_opt_speed.cu#L856)。

#### 为什么要单独保存

`derive_eb` 产生 `eq_dEb_component[i] == 0`，表示该位置允许的 EB 小到无法用
普通量化路径安全表示。Naive 版本先保存该位置的原始值和下标，再临时把它的
EB 改成最大 EB，让统一的 Lorenzo kernel 能继续运行。解压时再用保存的原始值
覆盖 Lorenzo 结果，从而保持这些位置精确。

例如：

```text
eq_dEb_U       = [2, 0, 3, 0]
dU             = [7.1, 8.2, 9.3, 10.4]

zero_U_indices = [1, 3]
zero_U_values  = [8.2, 10.4]
zero_U_count   = 2
```

#### 输入

- `dU`, `dV`：需要保留异常原值的数据。
- `eq_dEb_U/V`：作为 `copy_if` 的筛选 stencil。
- `dEb_U/V`：之后要替换的小 EB。

#### 处理过程

先生成所有线性下标，再分别 compact U/V 的原始值和下标：

```cpp
thrust::sequence(thrust::device,
                 data_indices, data_indices + r1 * r2);

end_it = thrust::copy_if(thrust::device,
    dU, dU + r1 * r2, eq_dEb,
    ebIsZero_U_data, IsZero<T>());
thrust::copy_if(thrust::device,
    data_indices, data_indices + r1 * r2, eq_dEb,
    ebIsZero_U_indices, IsZero<T>());
zero_eb_U_count = end_it - ebIsZero_U_data;

end_it = thrust::copy_if(thrust::device,
    dV, dV + r1 * r2, eq_dEb_V,
    ebisZero_V_data, IsZero<T>());
thrust::copy_if(thrust::device,
    data_indices, data_indices + r1 * r2, eq_dEb_V,
    ebIsZero_V_indices, IsZero<T>());
zero_eb_V_count = end_it - ebisZero_V_data;
```

值和下标使用同一个 stencil，`copy_if` 保持相同顺序，所以第 `k` 个 value
与第 `k` 个 index 对应。之后替换浮点 EB 和 EB ID：

```cpp
int id = log2(max_pwr_eb / threshold) / 2.0;
T eb_back = (T)(1ULL << (2 * id)) * threshold;

thrust::transform(thrust::device,
    dEb_U, dEb_U + n, dEb_U,
    ReplaceLessThreshold(eb_back, threshold));
thrust::transform(thrust::device,
    dEb_V, dEb_V + n, dEb_V,
    ReplaceLessThreshold(eb_back, threshold));
thrust::transform(thrust::device,
    eq_dEb_U, eq_dEb_U + n, eq_dEb_U,
    ReplaceZero(id));
thrust::transform(thrust::device,
    eq_dEb_V, eq_dEb_V + n, eq_dEb_V,
    ReplaceZero(id));
```

#### 输出

- `ebIsZero_U_data`, `ebIsZero_U_indices`, `zero_eb_U_count`。
- `ebisZero_V_data`, `ebIsZero_V_indices`, `zero_eb_V_count`。
- 已把 `<= threshold` 的浮点 EB 和 ID 0 替换掉的 `dEb_U/V`、
  `eq_dEb_U/V`。

zero-EB 列表不会送入 Huffman；它们作为 `(index, original_value)` 旁路数组
直接写入文件。解压时在 Lorenzo 完成后通过 `thrust::scatter` 覆盖回去，见
[`zero-EB 恢复`](./cpszg_2d_no_opt_speed.cu#L1287)。

#### Naive 性能特征

这一段至少包含 9 次顶层全数组操作：1 次 `sequence`、4 次 `copy_if` 和 4 次
`transform`。其中每次 `copy_if` 内部还可能包含不止一个 GPU kernel，因此
它通常是 Naive 版本明显的内存流量和 kernel-launch 开销来源。

## 3. lorenzo

调用位置：[`RUN_QUANTIZATION_LORENZO`](./cpszg_2d_no_opt_speed.cu#L926)，
kernel 位置：
[`KERNEL_c_lorenzo_row1d__eb_list`](../cusz/detail/lproto_c.cuhip.inl#L330)，
launch wrapper：
[`GPU_PROTO_c_lorenzo_row1d__eb_list`](../cusz/detail/lproto_c.cuhip.inl#L684)。

虽然输入是二维数组，这个 Naive 主路径实际使用的是**逐行一维 Lorenzo**：
各行之间并行，一行内部从左到右串行，预测器只使用左边已经重建的值。

### 输入

- U 路：`dU`, `dEb_U`；V 路：`dV`, `dEb_V`。
- `RADIUS = 512`：可编码量化码的有效半径。
- `r1`, `r2`。

### 处理过程

每个 CUDA thread 负责一整行：

```cpp
size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
if (row >= r1) return;

double prev = 0.0;
for (size_t j = 0; j < r2; j++) {
    size_t id = row * r2 + j;
    double cur  = (double)in_data[id];
    double eb   = (double)eb_list[id];
    double pred = prev;
    // quantize or save as outlier ...
}
```

首列没有左邻居，因此总是作为 outlier 保存原始值。其余元素计算预测差并尝试
量化：

```cpp
double diff  = cur - pred;
double qdiff = fabs(diff) / eb + 1.0;

if (qdiff < 2 * radius) {
    qdiff = (diff > 0.0) ? qdiff : -qdiff;
    int qi = (int)(qdiff / 2) + radius;
    double decomp = pred + (qi - radius) * 2.0 * eb;

    if (fabs(decomp - cur) < eb) {
        out_eq[id] = (Eq)qi;
        prev = decomp;
        continue;
    }
}
```

预测状态使用 `decomp` 而不是 `cur`，保证压缩端和解压端使用完全相同的左邻
值，避免误差沿一行失配。若量化码超出半径或重建误差检查失败，就保存为
outlier：

```cpp
out_eq[id] = 0;
uint32_t idx = atomicAdd(out_cn, 1);
out_cidx[idx] = (uint32_t)id;
out_cval[idx] = in_data[id];
prev = cur;
```

例如 `pred = 10.0`、`cur = 10.25`、`eb = 0.1`，可得到 `qi = 513`，重建值
为 `10.2`，误差为 `0.05 < 0.1`，因此在 `eq` 中保存 513，而不是保存原始
浮点数。

主函数先压缩 U，再同步，然后压缩 V：

```cpp
GPU_PROTO_c_lorenzo_row1d__eb_list(
    dU, dim3(r2, r1, 1), eq_U,
    ot_val_U, ot_idx_U, ot_num_U, dEb_U, RADIUS, &lrz_time, 0);
cudaDeviceSynchronize();

GPU_PROTO_c_lorenzo_row1d__eb_list(
    dV, dim3(r2, r1, 1), eq_V,
    ot_val_V, ot_idx_V, ot_num_V, dEb_V, RADIUS, &lrz_time, 0);
```

### 输出

- `eq_U`, `eq_V`：长度都是 `n` 的 Lorenzo 量化码数组。
- `ot_idx_U/V`, `ot_val_U/V`, `ot_num_U/V`：两个分量各自的 outlier 旁路。

outlier 和 zero-EB 是不同集合。outlier 表示 Lorenzo 预测差无法安全编码；
zero-EB 表示 derive 阶段给出的允许误差太小。解压时 outlier 必须在 Lorenzo
之前预填充，zero-EB 必须在 Lorenzo 之后覆盖。

### Naive 性能特征

- 一行只有一个 thread，行内 `r2` 次迭代存在严格的数据依赖；列方向没有并行。
- U 和 V 顺序执行，不能互相重叠。
- outlier 通过一个全局计数器 `atomicAdd` 分配位置。
- 每个元素都从全局内存读取一个浮点 `eb_list`。

## 4. huffman

调用位置：[`RUN_HF`](./cpszg_2d_no_opt_speed.cu#L958)，桥接实现：
[`gpu_huffman_bridge_no_opt_speed.cu`](./gpu_huffman_bridge_no_opt_speed.cu#L14)。

### 输入

Huffman 对下面 4 个长度均为 `n` 的符号数组分别编码：

```text
eq_U, eq_V, eq_dEb_U, eq_dEb_V
```

通常 `eq_U/V` 是 `uint16_t`，`eq_dEb_U/V` 是 `uint8_t`，所以主函数调用
`run_gpu_huffman_u2_u1_arrays`：

```cpp
run_gpu_huffman_u2_u1_arrays(
    reinterpret_cast<uint16_t*>(eq_U),
    reinterpret_cast<uint16_t*>(eq_V),
    reinterpret_cast<uint8_t*>(eq_dEb_U),
    reinterpret_cast<uint8_t*>(eq_dEb_V),
    num_elements, stream,
    hf_lens, hf_encode_ms, hf_decode_ms, hf_blobs);
```

### 单个数组的处理过程

以 `uint8_t` 的 EB ID 数组为例：

```cpp
GPU_histogram_generic<uint8_t>::kernel(
    d_symbols, n, d_hist, bklen,
    grid_dim, block_dim, shmem_use, r_per_block, stream);

cudaMemcpyAsync(h_hist, d_hist, bklen * sizeof(uint32_t),
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);

if (h_hist[0] == 0) h_hist[0] = 1;
phf::high_level<uint8_t>::build_book(&hf_buf, h_hist, bklen, stream);
phf::high_level<uint8_t>::encode(
    &hf_buf, d_symbols, n,
    &encoded, &encoded_len, hf_header, stream);
```

步骤为：GPU 统计直方图，直方图复制到 CPU，构造 Huffman codebook，再在 GPU
上编码。`h_hist[0] = 1` 是对单一符号 Huffman 树会产生空 bitstream 的规避。
编码后的 blob 会复制到 host，因为同一个 `hf_buf` 随后会被下一数组重用。

4 个数组当前顺序调用，不是同时编码：

```cpp
hf_encode_decode_u2("eq_U",     eq_U,     ...);
hf_encode_decode_u2("eq_V",     eq_V,     ...);
hf_encode_decode_u1("eq_dEb_U", eq_dEb_U, ...);
hf_encode_decode_u1("eq_dEb_V", eq_dEb_V, ...);
```

每个 helper 还会立刻执行一次 decode，用于验证和统计解压时间；decode 的结果
不会写入压缩文件，`hf_encode_ms` 只统计 histogram、build book 和 encode。

### 输出

- `hf_blobs[0..3]`：4 个 host 端 Huffman blob。
- `hf_lens[0..3]`：对应 blob 的字节数。
- `hf_encode_ms[0..3]`, `hf_decode_ms[0..3]`：性能计时。

最后 [`write_cucpsz`](./cpszg_2d_no_opt_speed.cu#L400) 按以下顺序写文件：

```text
CucpszHeader
4 个 Huffman blob：eq_U, eq_V, eq_dEb_U, eq_dEb_V
land bitpack
U outlier indices + values
U zero-EB indices + values
V outlier indices + values
V zero-EB indices + values
```

对应的核心写文件代码是：

```cpp
for (int i = 0; i < 4; i++)
    fwrite(hf_blobs[i], 1, hf_lens[i], f);

fwrite(land_bitpack, 1, land_bitpack_bytes, f);
fwrite(ot_idx_U, sizeof(uint32_t), ot_count_U, f);
fwrite(ot_val_U, sizeof(float), ot_count_U, f);
fwrite(zeroeb_idx_U, sizeof(uint32_t), zeroeb_count_U, f);
fwrite(zeroeb_val_U, sizeof(float), zeroeb_count_U, f);
// V 同样写入 outlier 和 zero-EB 数组
```

### Naive 性能特征

每个数组都独立经历 histogram、D2H 同步、codebook 构造和 encode，四个数组
串行处理。尤其是逐元素的 `eq_dEb_U/V` 也各自具有 `n` 个符号，因此会产生
额外的两轮全数组 Huffman 工作和两个完整的压缩 blob。

## 各阶段输入输出汇总

| 阶段          | 主要输入                       | 主要输出                                     |
| ------------- | ------------------------------ | -------------------------------------------- |
| `derive_eb` | `dU`, `dV`, `max_pwr_eb` | `dEb_U/V`, `eq_dEb_U/V`                  |
| `land_data` | U/V 为 0 的位置、EB 数组       | land bitpack、修正后的 EB/ID                 |
| `eb_zero`   | ID 为 0 的位置、原始 U/V       | zero-EB values/indices/count、修正后的 EB/ID |
| `lorenzo`   | `dU/V`, `dEb_U/V`          | `eq_U/V`、outlier values/indices/count     |
| `huffman`   | `eq_U/V`, `eq_dEb_U/V`     | 4 个 Huffman blob 及长度                     |

解压按相反方向恢复 Huffman 符号和浮点 EB，预填 Lorenzo outlier，执行逐行
Lorenzo 解压，再覆盖 zero-EB 原值，最后按照 land bitpack 将陆地位置恢复成
`(U,V) = (0,0)`。对应代码见
[`解压恢复顺序`](./cpszg_2d_no_opt_speed.cu#L1520)。
