# Naive 与 V1 CUDA 验证

### 测试设置与方法

- 代码版本：`5f9f8ab`。
- 构建：Release，CUDA 12.6.85，`-O3 -DNDEBUG`。
- GPU：NVIDIA GeForce RTX 3080，驱动 610.88。
- 输入：`uf.dat` 和 `vf.dat`，float32，`2400 x 3600`，U/V 各 8,640,000 个值。
- 参数：最大相对误差界 `0.1`；Naive 和 V1 均使用 `RADIUS=512`；V1 使用 `16x16` tile。
- 两个版本各预热一次，然后交替运行 20 次；奇数轮先运行 Naive，偶数轮先运行 V1，以减小温度、频率和运行顺序造成的偏差。
- GPU 时钟未锁定，部分阶段存在调度/频率离群值，因此以下比较采用 20 次运行的中位数，而不是单次结果或均值。
- 原始日志和复现脚本位于 `build/benchmarks/naive-v1-20260902/`。

### 核心阶段结果

下表中的“加速比”定义为 `Naive 时间 / V1 时间`；大于 1 表示 V1 更快。

| 阶段                    | Naive 中位数（ms） |   V1 中位数（ms） |       V1 相对变化 |           加速比 |
| ----------------------- | -----------------: | ----------------: | ----------------: | ---------------: |
| 推导 EB                 |             4.2915 |            3.9680 |            -7.54% |           1.082x |
| 陆地区域处理            |             0.2015 |            0.1765 |           -12.41% |           1.142x |
| Zero-EB 处理            |             1.2235 |            0.2160 |           -82.35% | **5.664x** |
| Tile 统一 EB            |                  - |            0.3005 |    新增 0.3005 ms |                - |
| Lorenzo 压缩            |            10.0975 |            1.8285 |           -81.89% | **5.522x** |
| Huffman（程序现有口径） |             5.1955 |            3.6550 |           -29.65% |           1.421x |
| Tile-EB 打包            |                  - |            0.1310 |    新增 0.1310 ms |                - |
| 核心阶段合计            |  **21.0795** | **10.2875** | **-51.20%** | **2.049x** |
| 解压                    |             5.4433 |            2.9654 |           -45.52% | **1.836x** |

按中位数计算，V1 的核心压缩阶段节省约 10.792 ms，内部吞吐量由约
3.054 GiB/s 提高到 6.257 GiB/s。主要收益不是 Huffman，而是：

1. 计入 Tile 统一 EB 后，二维 tile-Lorenzo 路径净节省约 7.969 ms，是最大的性能来源；
2. `kernel_fused_ebzero` 节省约 1.008 ms；
3. 当前口径下的 Huffman 阶段节省约 1.541 ms；
4. `uniform_eb` 和 `tile_eb` 合计新增约 0.426 ms，约占 V1 核心总时间的 4.1%。

### Zero-EB 融合的实际收益

Naive 的 zero-EB 路径执行一次 `thrust::sequence`、四次 `copy_if` 和四次
`transform`，需要多次扫描完整数组（[源码](../naive/cpszg_2d_no_opt_speed.cu#L863-L875)）。
V1 的 `kernel_fused_ebzero` 在一次扫描中完成 U/V 分类、值和下标收集、计数以及
EB/ID 替换（[调用点](cpszg_2d_v1.cu#L929-L933)，
[kernel 源码](fused_ebzero_kernel.cuh#L27-L44)）。20 次测试中：

```text
Naive：1.2235 ms
V1   ：0.2160 ms
节省 ：1.0075 ms（82.35%）
加速 ：5.664x
```

因此，kernel fusion 对 zero-EB 的提升明确而且稳定。需要注意，V1 的计时不包含
kernel 前的两个计数器清零以及 kernel 后的两个计数器 D2H 拷贝，所以这是核心
GPU 操作的比较，不是完整的 zero-EB 端到端时间。

#### 对应代码

Naive：一次 `sequence`、四次 `copy_if` 和四次 `transform`

```cpp
thrust::sequence(thrust::device, data_indices, data_indices + r1 * r2);
end_it = thrust::copy_if(thrust::device, dU, dU + r1 * r2, eq_dEb, ebIsZero_U_data, IsZero<T>());
end_idx = thrust::copy_if(thrust::device, data_indices, data_indices + r1 * r2, eq_dEb, ebIsZero_U_indices, IsZero<T>());
end_it = thrust::copy_if(thrust::device, dV, dV + r1 * r2, eq_dEb_V, ebisZero_V_data, IsZero<T>());
end_idx = thrust::copy_if(thrust::device, data_indices, data_indices + r1 * r2, eq_dEb_V, ebIsZero_V_indices, IsZero<T>());
thrust::transform(thrust::device, dEb_U, dEb_U + r1 * r2, dEb_U, ReplaceLessThreshold(eb_back, threshold));
thrust::transform(thrust::device, dEb_V, dEb_V + r1 * r2, dEb_V, ReplaceLessThreshold(eb_back, threshold));
thrust::transform(thrust::device, eq_dEb, eq_dEb + r1 * r2, eq_dEb, ReplaceZero(id));
thrust::transform(thrust::device, eq_dEb_V, eq_dEb_V + r1 * r2, eq_dEb_V, ReplaceZero(id));
```

V1：一个 fused kernel 直接收集值和下标并完成替换

```cpp
kernel_fused_ebzero<T, Eq2><<<zeb_grid, ZEB_BLK>>>
   (dU, dV, dEb_U, dEb_V, eq_dEb, eq_dEb_V,
    ebIsZero_U_data, ebIsZero_U_indices, d_zeb_U_cnt,
    ebisZero_V_data, ebIsZero_V_indices, d_zeb_V_cnt,
    eb_back, threshold, (Eq2)id, num_elements);
```

具体kernel：

```cpp
const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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

### 二维 Lorenzo 的实际收益

Naive 的逐行 Lorenzo 由少量线程沿每行串行推进；V1 使用 `16x16` tile 和二维并行前缀运算。由于 tile-2D Lorenzo 依赖预先生成的 Tile 统一 EB，实际收益应将`uniform_eb` 计入 V1 策略开销。20 次测试中：

```text
Naive row-1D Lorenzo：10.0975 ms
V1 Tile uniform EB      ： 0.3005 ms
V1 tile-2D Lorenzo  ： 1.8285 ms
V1 策略合计          ： 2.1290 ms
净节省               ： 7.9685 ms（78.92%）
实际加速             ： 4.743x
```

若只比较 Lorenzo kernel，原始结果仍是节省 8.2690 ms、加速 5.522x；计入必要的Tile uniform EB 后，二维 tile-Lorenzo 策略的净收益为 7.9685 ms，实际加速为 4.743x。这是 V1 总加速的主要来源。不过，这并不是只替换一个等价 kernel：V1 同时使用tile 公共误差界，改变了误差界分布、量化码和 outlier 数量。因此该结果表示当前两条完整 Lorenzo 路径的性能差，而不是完全相同输入下单个 kernel 的微基准。

#### 对应代码

Naive：U/V 分别调用 row-1D Lorenzo

```cpp
psz::cuhip::GPU_PROTO_c_lorenzo_row1d__eb_list<T, Eq1>(dU, dim3(r2, r1, 1), eq_U, ot_val_U, ot_idx_U, ot_num_U, dEb_U, RADIUS, &lrz_time, 0);
psz::cuhip::GPU_PROTO_c_lorenzo_row1d__eb_list<T, Eq1>(dV, dim3(r2, r1, 1), eq_V, ot_val_V, ot_idx_V, ot_num_V, dEb_V, RADIUS, &lrz_time, 0);
```

Naive row-1D kernel 的关键结构：每个线程负责一行，并在 `for (j)` 中串行处理该行全部列

```cpp
size_t r1 = data_len3.y, r2 = data_len3.x;
size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
if (row >= r1) return;

double prev = 0.0;

for (size_t j = 0; j < r2; j++) {
    size_t id = row * r2 + j;
    double cur = (double)in_data[id];
    double eb = (double)eb_list[id];
    double pred = prev;

    if (j == 0) {
        out_eq[id] = 0;
        uint32_t idx = atomicAdd(out_cn, 1);
        out_cidx[idx] = (uint32_t)id;
        out_cval[idx] = in_data[id];
        prev = cur;
        continue;
    }

    if (eb > 0.0) {
        double diff = cur - pred;
        double qdiff = fabs(diff) / eb + 1.0;
        if (qdiff < (double)(2 * radius)) {
            qdiff = (diff > 0.0) ? qdiff : -qdiff;
            int qi = (int)(qdiff / 2) + radius;
            double decomp = pred + (double)(qi - radius) * 2.0 * eb;
            if (fabs(decomp - cur) < eb) {
                out_eq[id] = (Eq)qi;
                prev = decomp;
                continue;
            }
        }
    }

    out_eq[id] = 0;
    uint32_t idx = atomicAdd(out_cn, 1);
    out_cidx[idx] = (uint32_t)id;
    out_cval[idx] = in_data[id];
    prev = cur;
}
```

Naive 的启动配置也直接体现了并行度限制：每个 block 只有32个线程，每个线程对应一整行

```cpp
constexpr int ROWS_PER_BLOCK = 32;
size_t r1 = data_len3.y;
dim3 grid((r1 + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
dim3 block(ROWS_PER_BLOCK, 1, 1);

psz::KERNEL_c_lorenzo_row1d__eb_list<T, Eq>
    <<<grid, block, 0, (cudaStream_t)stream>>>(in_data, data_len3, out_eq, out_ot_val, out_ot_idx, out_ot_num, EFF_RADIUS, eb_list);
```

V1 前置步骤：先把每个 `16x16` tile 内的 EB 统一为最小 EB；U/V 的两次调用共同构成上文计入的 `0.3005 ms` Tile 统一 EB 开销（[调用点](cpszg_2d_v1.cu#L944-L957)）。

```cpp
dim3 ublk(LORENZO_TILE_DIM, LORENZO_TILE_DIM, 1);
dim3 ugrid((r2 + LORENZO_TILE_DIM - 1) / LORENZO_TILE_DIM,
           (r1 + LORENZO_TILE_DIM - 1) / LORENZO_TILE_DIM, 1);

cudaEventRecord(_ce0);
kernel_uniformize_tile_eb<T, Eq2><<<ugrid, ublk>>>(
    dEb_U, eq_dEb, (int)r1, (int)r2, threshold);
kernel_uniformize_tile_eb<T, Eq2><<<ugrid, ublk>>>(
    dEb_V, eq_dEb_V, (int)r1, (int)r2, threshold);
cudaEventRecord(_ce1);
cudaEventSynchronize(_ce1);
cudaEventElapsedTime(&ms_uni, _ce0, _ce1);
```

Tile 统一 EB kernel 先在 shared memory 中归约 tile 的最小 EB ID，再将该 ID 和对应 EB 广播回 tile 内所有有效位置（[kernel 源码](tile_uniform_eb.cuh#L7-L43)）：

```cpp
constexpr int tile_elements = LORENZO_TILE_DIM * LORENZO_TILE_DIM;
__shared__ unsigned int tile_ids[tile_elements];

const int x = blockIdx.x * LORENZO_TILE_DIM + threadIdx.x;
const int y = blockIdx.y * LORENZO_TILE_DIM + threadIdx.y;
const int tid = threadIdx.y * LORENZO_TILE_DIM + threadIdx.x;
const bool in_bounds = x < r2 && y < r1;
const size_t i = in_bounds ? static_cast<size_t>(y) * r2 + x : 0;

tile_ids[tid] = in_bounds
    ? static_cast<unsigned int>(eq_eb[i])
    : 0xffffffffu;
__syncthreads();

for (int stride = tile_elements / 2; stride > 0; stride >>= 1) {
    if (tid < stride && tile_ids[tid + stride] < tile_ids[tid]) {
        tile_ids[tid] = tile_ids[tid + stride];
    }
    __syncthreads();
}

if (in_bounds) {
    const Eq tile_id = static_cast<Eq>(tile_ids[0]);
    eq_eb[i] = tile_id;
    eb[i] = static_cast<T>(1ULL << (2 * static_cast<unsigned int>(tile_id)))
        * threshold;
}
```

完成 Tile 统一 EB 后，V1 对 U/V 分别调用 tile-2D Lorenzo；在当前配置中，每个`16x16` block 并行处理一个 tile：

```cpp
psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__eb_list<T, Eq1>(dU, dim3(r2, r1, 1), eq_U, ot_val_U, ot_idx_U, ot_num_U, dEb_U, RADIUS, &lrz_time, 0);
psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__eb_list<T, Eq1>(dV, dim3(r2, r1, 1), eq_V, ot_val_V, ot_idx_V, ot_num_V, dEb_V, RADIUS, &lrz_time, 0);
```

V1 tile-2D kernel 的关键结构：每个线程只处理一个元素，先把量化值写入 shared memory，然后并行计算二维 Lorenzo 差分
`current - left - up + upper_left`：

```cpp
SETUP_ND_GPU_CUDA;
__shared__ T buf[TileDim][TileDim + 1];

uint32_t y = threadIdx.y, x = threadIdx.x;
auto data = [&](auto dx, auto dy) -> T& {
    return buf[t().y + dy][t().x + dx];
};

auto id = gid2();
if (check_boundary2()) {
    auto ebx2_r = 0.5 / eb_list[id];
    data(0, 0) = round(in_data[id] * ebx2_r);
}
__syncthreads();

T delta = data(0, 0) -
          ((x > 0 ? data(-1, 0) : 0) +
           (y > 0 ? data(0, -1) : 0) -
           (x > 0 && y > 0 ? data(-1, -1) : 0));

bool quantizable = fabs(delta) < radius;
T candidate = delta + radius;
if (check_boundary2()) {
    out_eq[id] = quantizable * static_cast<Eq>(candidate);
    if (!quantizable) {
        auto cur_idx = atomicAdd(out_cn, 1);
        out_cidx[cur_idx] = id;
        out_cval[cur_idx] = candidate;
    }
}
```

### Huffman 与 Tile-EB 的拆分

V1 的 Huffman 编码器本身没有获得可确认的加速。真正变化的是输入工作量：

- Naive 对 `eq_U`、`eq_V`、`eq_dEb_U`、`eq_dEb_V` 四个完整数组执行 Huffman；
- 本次测试采用的 V1 tile-uniform 分支只对 `eq_U`、`eq_V` 两个完整数组执行 Huffman；
- 该分支将每点 EB ID 改为每个 `16x16` tile 一个原始 ID，再执行两个很小的打包 kernel。

20 次运行的细分中位数为：

| 比较项                                     | Naive（ms） | V1（ms） | 结论                         |
| ------------------------------------------ | ----------: | -------: | ---------------------------- |
| 仅 U/V Huffman                             |      3.5800 |   3.6550 | V1 慢约 2.1%，基本可视为持平 |
| Naive 的两个逐点 EB Huffman                |      1.5100 |        - | V1 删除了这两次完整数组编码  |
| V1 的 Tile-EB pack                         |           - |   0.1310 | 新增，但远小于 1.5100 ms     |
| 完整日志中的 Huffman                       |      5.1955 |   3.6550 | V1 表面快 1.421x             |
| Naive Huffman 对比 V1 Huffman+pack         |      5.1955 |   3.7795 | V1 快 1.375x                 |
| Naive Huffman 对比 V1 uniform+Huffman+pack |      5.1955 |   4.0965 | V1 快 1.268x                 |

所以，对“V1 的 Huffman 是否变快”的准确回答是：

> 对相同类别的 U/V 数组，Huffman 本身没有变快；V1 的 Huffman 阶段总时间下降，是因为它不再 Huffman 编码两个长度为 8,640,000 的逐点 EB 数组。即使计入`tile_eb`，再把 `uniform_eb` 也算入这一策略的成本，V1 仍分别保留约 1.375x和 1.268x 的时间优势。

这个比较仍有计时口径限制：`huffman` 包含直方图、直方图 D2H 同步、CPU 码书构建和 encode，但不包含 Huffman buffer 构造、编码 blob D2H 和内部验证 decode；`tile_eb` 只包含两个 pack kernel，不包含临时 `cudaMalloc`、host `new`、D2H 和`cudaFree`。此外，Naive bridge 在计时区间中包含调试直方图扫描和打印。因此这些数字适合解释当前程序的核心阶段，不应称为严格的 Huffman 端到端微基准。

#### 对应代码

Naive：当前 `Eq1=uint16_t`、`Eq2=uint8_t` 路径把 U/V 量化码和 U/V 逐点 EB ID
四个完整数组一起交给 Huffman（[调用点](../naive/cpszg_2d_no_opt_speed.cu#L976-L983)）：

```cpp
if (sizeof(Eq1) == sizeof(uint16_t) && sizeof(Eq2) == sizeof(uint8_t)) {
    run_gpu_huffman_u2_u1_arrays(
        reinterpret_cast<uint16_t*>(eq_U),
        reinterpret_cast<uint16_t*>(eq_V),
        reinterpret_cast<uint8_t*>(eq_dEb),
        reinterpret_cast<uint8_t*>(eq_dEb_V),
        num_elements, stream,
        hf_lens, hf_encode_ms, hf_decode_ms, hf_blobs);
}
```

Naive bridge 随后依次编码这四个数组（[bridge 源码](../naive/gpu_huffman_bridge_no_opt_speed.cu#L196-L212)）：

```cpp
hf_encode_decode_u2("eq_U",     eq_U,     n, cuda_stream, hf_buf_u2,
                    &out_lens[0], &out_encode_ms[0], &out_decode_ms[0],
                    out_h_blobs ? &out_h_blobs[0] : nullptr);
hf_encode_decode_u2("eq_V",     eq_V,     n, cuda_stream, hf_buf_u2,
                    &out_lens[1], &out_encode_ms[1], &out_decode_ms[1],
                    out_h_blobs ? &out_h_blobs[1] : nullptr);
hf_encode_decode_u1("eq_dEb_U", eq_dEb_U, n, cuda_stream, hf_buf_u1,
                    &out_lens[2], &out_encode_ms[2], &out_decode_ms[2],
                    out_h_blobs ? &out_h_blobs[2] : nullptr);
hf_encode_decode_u1("eq_dEb_V", eq_dEb_V, n, cuda_stream, hf_buf_u1,
                    &out_lens[3], &out_encode_ms[3], &out_decode_ms[3],
                    out_h_blobs ? &out_h_blobs[3] : nullptr);
```

V1：在本次测试采用的 tile-uniform 分支中，只有 U/V 两个完整量化码数组进入Huffman；EB ID 则转入独立的 Tile-EB 打包路径
（[调用点](cpszg_2d_v1.cu#L1019-L1042)）：

```cpp
if (debug_options.use_tile_uniform_eb && sizeof(Eq1) == sizeof(uint16_t)) {
    size_t eq_hf_lens[2] = {0, 0};
    float eq_hf_encode_ms[2] = {0.0f, 0.0f};
    float eq_hf_decode_ms[2] = {0.0f, 0.0f};
    uint8_t* eq_hf_blobs[2] = {nullptr, nullptr};

    run_gpu_huffman_u2_arrays(
        reinterpret_cast<uint16_t*>(eq_U),
        reinterpret_cast<uint16_t*>(eq_V),
        num_elements, stream,
        eq_hf_lens, eq_hf_encode_ms, eq_hf_decode_ms, eq_hf_blobs);

    hf_lens[0] = eq_hf_lens[0];
    hf_lens[1] = eq_hf_lens[1];
    hf_encode_ms[0] = eq_hf_encode_ms[0];
    hf_encode_ms[1] = eq_hf_encode_ms[1];
    hf_decode_ms[0] = eq_hf_decode_ms[0];
    hf_decode_ms[1] = eq_hf_decode_ms[1];
    hf_blobs[0] = eq_hf_blobs[0];
    hf_blobs[1] = eq_hf_blobs[1];

    pack_tile_eb_payloads<Eq2>(
        eq_dEb, eq_dEb_V, r1, r2,
        hf_blobs[2], hf_blobs[3],
        hf_lens[2], hf_lens[3], ms_tile_eb_pack);
}
```

`run_gpu_huffman_u2_arrays` 内部只有 U/V 两次编码，不再接收 EB ID 数组
（[bridge 源码](gpu_huffman_u2_bridge.cu#L95-L115)）：

```cpp
encode_decode_u2(
    "eq_U", eq_U, n, cuda_stream, buffer, &out_lens[0],
    &out_encode_ms[0], &out_decode_ms[0],
    out_h_blobs ? &out_h_blobs[0] : nullptr);
encode_decode_u2(
    "eq_V", eq_V, n, cuda_stream, buffer, &out_lens[1],
    &out_encode_ms[1], &out_decode_ms[1],
    out_h_blobs ? &out_h_blobs[1] : nullptr);
```

Tile-EB pack kernel 为每个 tile 读取一个已经统一的 EB ID，并写入紧凑数组；因此每个分量的元素数从 `r1*r2` 降为 `tile_rows*tile_cols`
（[kernel 源码](cpszg_2d_v1.cu#L365-L379)）：

```cpp
int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
int tile_rows = (r1 + tile_dim - 1) / tile_dim;
if (tile_x >= tile_cols || tile_y >= tile_rows) return;

int x = tile_x * tile_dim;
int y = tile_y * tile_dim;
tile_eq_dEb[tile_y * tile_cols + tile_x] =
    eq_dEb[(size_t)y * r2 + x];
```
### 时间收益与空间代价

| 指标                |            Naive |      V1（16x16） |    变化 |
| ------------------- | ---------------: | ---------------: | ------: |
| U/V outlier 总数    |           26,871 |        1,507,766 |  56.11x |
| U/V Huffman payload |  9,818,120 bytes | 14,101,564 bytes | +43.63% |
| EB payload          |  4,027,240 bytes |     67,500 bytes | -98.32% |
| 特殊值 payload      |  2,188,480 bytes | 14,035,640 bytes |   6.41x |
| 完整文件            | 17,113,928 bytes | 29,284,800 bytes | +71.12% |
| 实际文件压缩率      |         4.038816 |         2.360269 | -41.56% |

Tile-EB 确实极大压缩了 EB 元数据，但 tile 最小误差界使大量位置采用更严格的 EB，从而扩大 U/V Huffman payload，并使 outlier 总数增加到 Naive 的约 56 倍。最终 V1虽然核心压缩约快 2.05x，但文件大 71.12%，压缩率由 4.039 降至 2.360。

#### 各项数据的计算方式与变化原因

表中前几项不是若干个独立文件，而是同一个 `.cucpsz` 文件内的不同 payload；`outlier 总数`则是记录数，不是字节数。完整文件由文件头、U/V 量化码 Huffman payload、EB payload、共享 land bitpack 和 U/V 特殊值 payload 组成。

##### 1. U/V outlier 总数

```text
Naive = 10,624 (U) + 16,247 (V)
      = 26,871

V1    = 719,663 (U) + 788,103 (V)
      = 1,507,766

V1 / Naive = 1,507,766 / 26,871
           = 56.11x
```

V1 先对每个 `16x16` tile 的 256 个逐点 EB ID 取最小值，再将该值广播回整个 tile（[Tile uniform kernel](tile_uniform_eb.cuh#L26-L42)）：

```text
EB_tile = min(EB_i), i ∈ tile
```

二维 Lorenzo 随后先计算量化整数，再计算二维差分：

```text
q     = round(x / (2 * EB_tile))
delta = q - left - up + upper_left
```

当 `|delta| >= RADIUS` 时，该位置无法放入普通量化码，会转入 outlier 侧通道（[二维量化与 outlier 判定](../cusz/detail/lproto_c.cuhip.inl#L138-L160)）。当前有效 `RADIUS=512`。EB 越小，`q` 及其差分的幅值通常越大，因此更容易越过固定的 `±512` 范围。

这个对比同时将预测器从 row-1D 换成了 tile-2D，tile 边界和二维差分也会改变码分布。因此，outlier 的精确增量是“Tile 最小 EB + 二维 Lorenzo”的共同结果，不能仅凭这一组数据将百分比完全归因于某一项。

##### 2. U/V Huffman payload

```text
Naive = 4,569,620 (U) + 5,248,500 (V)
      = 9,818,120 bytes

V1    = 7,009,528 (U) + 7,092,036 (V)
      = 14,101,564 bytes

变化 = (14,101,564 - 9,818,120) / 9,818,120
     = +43.63%
```

两边送入 Huffman 的 U/V 元素数没有增加，仍然是每个分量 `8,640,000` 个 `uint16_t` 码。变大的是实际编码 blob，原因是更小的 tile EB 以及不同的二维 Lorenzo 差分使量化码分布更分散、符号熵更高，所以平均每个符号需要更多 bit。

outlier 位置在 U/V 量化码数组中会留下零标记，这部分本身有利于 Huffman 压缩，但不足以抵消其余符号分布变宽带来的增长。因此，Huffman 字节数不能仅由 outlier 数量推出，而是由完整码频分布决定。

##### 3. EB payload

Naive 保留两个逐点 `uint8_t` EB ID 数组，并分别对其做 Huffman 压缩：

```text
Naive EB = 2,015,272 (U) + 2,011,968 (V)
         = 4,027,240 bytes
```

V1 每个 `16x16` tile 只保存一个原始 `uint8_t` EB ID（[Tile-EB 打包](cpszg_2d_v1.cu#L397-L424)）：

```text
tile 行数       = ceil(2400 / 16) = 150
tile 列数       = ceil(3600 / 16) = 225
每分量 tile 数 = 150 * 225 = 33,750

V1 EB = 33,750 (U) + 33,750 (V)
      = 67,500 bytes

变化 = (67,500 - 4,027,240) / 4,027,240
     = -98.32%
```

这里减少的是“EB 描述数量”：从每点一个 ID 改为每 256 点共享一个 ID。

##### 4. 特殊值 payload

特殊值 payload 同时包含 outlier 和 zero-EB 记录。每条记录都保存一个 `uint32_t index` 和一个 `float value/code`，因此占 `4 + 4 = 8 bytes`（[特殊值文件布局](cpszg_2d_v1.cu#L440-L465)、[写入代码](cpszg_2d_v1.cu#L512-L526)）。

```text
Naive U = (10,624 + 123,344) * 8 = 1,071,744 bytes
Naive V = (16,247 + 123,345) * 8 = 1,116,736 bytes
Naive special                       = 2,188,480 bytes

V1 U = (719,663 + 123,344) * 8 = 6,744,056 bytes
V1 V = (788,103 + 123,345) * 8 = 7,291,584 bytes
V1 special                         = 14,035,640 bytes

V1 / Naive = 14,035,640 / 2,188,480
           = 6.41x
```

两版 zero-EB 数量完全相同：U 为 `123,344`，V 为 `123,345`。因此 special 增加的字节全部来自新增 outlier：

```text
新增 outlier = 1,507,766 - 26,871
             = 1,480,895

新增 special = 1,480,895 * 8
             = 11,847,160 bytes
```

special 只增至 6.41 倍，而outlier 增至 56.11 倍，是因为 special 还包含两版共有且数量不变的 `246,689` 条 zero-EB 记录。

##### 5. 完整文件

Naive 的完整文件为：

```text
       88  header
+ 9,818,120  U/V Huffman
+ 4,027,240  EB Huffman
+ 1,080,000  land bitpack
+ 2,188,480  special
------------
 17,113,928  bytes
```

V1 的完整文件为：

```text
       96  header
+14,101,564  U/V Huffman
+    67,500  raw Tile-EB
+ 1,080,000  land bitpack
+14,035,640  special
------------
 29,284,800  bytes
```

两者的净变化可以精确闭合：

```text
U/V Huffman  + 4,283,444 bytes
EB           - 3,959,740 bytes
special      +11,847,160 bytes
header       +         8 bytes
land bitpack +         0 bytes
--------------------------------
总计         +12,170,872 bytes
```

```text
12,170,872 / 17,113,928 = +71.12%
```

V1 的文件头比 Naive 多 8 bytes，是因为新增了 `lorenzo_variant` 和 `lorenzo_tile_dim` 两个 `uint32_t` 字段（[V1 文件头](cpszg_2d_v1.cu#L440-L461)）。`land bitpack` 为 U/V 共享且只保存一次，每个网格位置占 1 bit：

```text
8,640,000 / 8 = 1,080,000 bytes
```

因此，文件净增长的主要来源是 outlier 特殊值侧通道，而不是 EB 元数据或文件头。

##### 6. 实际文件压缩率

原始 U/V 数据的总字节数固定为：

```text
2400 * 3600 * 2 个分量 * sizeof(float)
= 2400 * 3600 * 2 * 4
= 69,120,000 bytes
```

所以：

```text
Naive = 69,120,000 / 17,113,928 = 4.038816
V1    = 69,120,000 / 29,284,800 = 2.360269

变化 = 2.360269 / 4.038816 - 1
     = -41.56%
```

文件大小与压缩率互为倒数，所以“文件增大 71.12%”和“压缩率下降 41.56%”同时成立，两个百分比的绝对值不会相同。

综合来看，Tile-EB 节省了 `3,959,740 bytes` EB 元数据，但 U/V Huffman 增加了 `4,283,444 bytes`，outlier 特殊值又增加了 `11,847,160 bytes`。后两项超过 EB 节省量，最终使 V1 文件增大 `12,170,872 bytes`。

20 次运行中，两版均满足：

- 内存解压与文件解压 U/V 位级一致；
- 解压 EB 验证错误数为 0；
- 原始和解压场均有 20,929 个临界点，`TP=20,929、FP=0、FN=0`。

#### 为什么 V1 的重建质量明显更高

V1 的 U/V 最大绝对误差由 Naive 的 `15.5392/15.4407` 降到 `0.9899/0.2500`，PSNR 由 `66.226/68.229 dB` 提高到 `92.260/96.202 dB`。这一质量提升主要不是二维 Lorenzo 预测器本身带来的，而是 V1 在量化前额外执行了 Tile-uniform EB。

`derive_eb` 首先将逐点误差界离散为（[EB 离散化代码](cpszg_2d_v1.cu#L323-L352)）：

```text
threshold = 2^-20
EB(id)    = 2^(2*id) * threshold
          = 4^id * 2^-20
```

因此 EB ID 每降低一级，实际 EB 就缩小 4 倍。Naive 直接使用每个点自己的 EB；V1 则对每个 `16x16` tile 取最小 EB ID，并将它广播给块内全部 256 个点：

```text
EB_V1(i) = min(EB_Naive(j)), j ∈ i 所在的 tile
```

这意味着，tile 内只要有一个低幅值点或拓扑约束较严的点，其余位置也必须使用同样小的 EB。量化步长为 `2*EB`，所以更小的 EB 会让重建值落在更密的量化网格上，从而同时降低局部误差、整体 MSE 和视觉失真。

实测 EB 上限与最大误差几乎一一对应：

| 路径 | 分量 | 最大 EB ID | 对应 EB | 实测最大绝对误差 |
| ---- | ---- | -----------: | ------: | -------------------: |
| Naive | U | 12 | 16 | 15.5392 |
| Naive | V | 12 | 16 | 15.4407 |
| V1 | U | 10 | 1 | 0.9899 |
| V1 | V | 9 | 0.25 | 0.2500 |

zero-EB 点也会进一步收紧周围点的误差界。这些点在 Tile uniform 之前会被临时替换为 `id=8`，即 `EB=0.0625`（[zero-EB 替换代码](fused_ebzero_kernel.cuh#L30-L44)）。该点自身会在解压后用原值精确回填，但它仍参与 tile 最小值归约，因而可能将同一 tile 的普通点一并限制到 `EB <= 0.0625`。

PSNR 的变化也是这一过程的直接结果。程序使用（[PSNR 计算](../../../include/utils.hpp#L95-L98)）：

```text
PSNR = 20*log10(range / RMSE)
```

两版对同一原始数据计算，因此 `range` 相同。V1 的 U/V RMSE 分别约缩小 `20.03x/25.04x`，对应的 PSNR 增量为：

```text
U: 20*log10(20.03) = 26.03 dB
V: 20*log10(25.04) = 27.97 dB
```

这正好对应 `66.226 -> 92.260 dB` 和 `68.229 -> 96.202 dB`。

需要特别注意，outlier 只表示 Lorenzo 差分超出固定码域，不表示该点被错误重建。V1 会把完整 delta code 写入特殊值侧通道，解压时再恢复相同的量化整数；因此 outlier 增多主要损害文件大小，不会通过截断 delta 直接降低重建质量。二维 tile-Lorenzo 对已量化整数进行可逆的二维差分和前缀恢复，主要改变执行方式、码分布和 outlier 数量，不是这次数量级误差下降的直接原因。

因此，相同的 `max_relative_eb=0.1` 并不表示 Naive 和 V1 处于相同率失真点：V1 用更大文件换取了更小实际 EB 和更高重建质量。若要评价算法整体优劣，还应补充“相同质量”或“相同文件大小”下的参数扫描，不能只比较单一 `max_relative_eb=0.1` 配置。

### 计时范围说明

程序打印的 `COMPRESS_TIME` 是 CUDA event 时间与 host chrono 时间的相加，不是从输入到 `.cucpsz` 文件落盘的完整端到端时间。它没有完整包含初始分配/H2D、CPUland-bitpack 构建、部分计数器拷贝、Huffman buffer/blob 拷贝以及文件写入。外层进程墙钟时间还混入了解压、文件 I/O、质量计算和 CPU 临界点检查，因此也不能直接作为压缩时间。当前报告用现有阶段计时回答各个优化步骤的核心性能差异，并明确列出其边界。
