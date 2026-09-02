# Naive 与 V1 CUDA 验证

初始测试日期：2026-08-23；当前代码复测日期：2026-09-02

> **2026-09-02 更新：** 当前代码已经从本报告最初测试时的
> `LORENZO_TILE_DIM=32`、`RADIUS=4096` 改为 `LORENZO_TILE_DIM=16`、
> `RADIUS=512`。下面先给出当前代码的重新测试结果；后续章节保留
> 2026-08-23 的历史测试，用于记录当时版本，不能与当前结果混用。

## 2026-09-02 当前代码复测

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

| 阶段 | Naive 中位数（ms） | V1 中位数（ms） | V1 相对变化 | 加速比 |
|---|---:|---:|---:|---:|
| 推导 EB | 4.2915 | 3.9680 | -7.54% | 1.082x |
| 陆地区域处理 | 0.2015 | 0.1765 | -12.41% | 1.142x |
| Zero-EB 处理 | 1.2235 | 0.2160 | -82.35% | **5.664x** |
| Tile 统一 EB | - | 0.3005 | 新增 0.3005 ms | - |
| Lorenzo 压缩 | 10.0975 | 1.8285 | -81.89% | **5.522x** |
| Huffman（程序现有口径） | 5.1955 | 3.6550 | -29.65% | 1.421x |
| Tile-EB 打包 | - | 0.1310 | 新增 0.1310 ms | - |
| 核心阶段合计 | **21.0795** | **10.2875** | **-51.20%** | **2.049x** |
| 解压 | 5.4433 | 2.9654 | -45.52% | **1.836x** |

按中位数计算，V1 的核心压缩阶段节省约 10.792 ms，内部吞吐量由约
3.054 GiB/s 提高到 6.257 GiB/s。主要收益不是 Huffman，而是：

1. 二维 tile-Lorenzo 节省约 8.269 ms，是最大的性能来源；
2. `kernel_fused_ebzero` 节省约 1.008 ms；
3. 当前口径下的 Huffman 阶段节省约 1.541 ms；
4. `uniform_eb` 和 `tile_eb` 合计新增约 0.426 ms，约占 V1 核心总时间的 4.1%。

### Zero-EB 融合的实际收益

Naive 的 zero-EB 路径执行一次 `thrust::sequence`、四次 `copy_if` 和四次
`transform`，需要多次扫描完整数组。V1 的 `kernel_fused_ebzero` 在一次扫描中
完成 U/V 分类、值和下标收集、计数以及 EB/ID 替换。20 次测试中：

```text
Naive：1.2235 ms
V1   ：0.2160 ms
节省 ：1.0075 ms（82.35%）
加速 ：5.664x
```

因此，kernel fusion 对 zero-EB 的提升明确而且稳定。需要注意，V1 的计时不包含
kernel 前的两个计数器清零以及 kernel 后的两个计数器 D2H 拷贝，所以这是核心
GPU 操作的比较，不是完整的 zero-EB 端到端时间。

### 二维 Lorenzo 的实际收益

Naive 的逐行 Lorenzo 由少量线程沿每行串行推进；V1 使用 `16x16` tile 和二维
并行前缀运算。20 次测试中：

```text
Naive：10.0975 ms
V1   ： 1.8285 ms
节省 ： 8.2690 ms（81.89%）
加速 ： 5.522x
```

这是 V1 总加速的主要来源。不过，这并不是只替换一个等价 kernel：V1 同时使用
tile 公共误差界，改变了误差界分布、量化码和 outlier 数量。因此该结果表示当前
两条完整 Lorenzo 路径的性能差，而不是完全相同输入下单个 kernel 的微基准。

### Huffman 与 Tile-EB 的拆分

V1 的 Huffman 编码器本身没有获得可确认的加速。真正变化的是输入工作量：

- Naive 对 `eq_U`、`eq_V`、`eq_dEb_U`、`eq_dEb_V` 四个完整数组执行 Huffman；
- V1 只对 `eq_U`、`eq_V` 两个完整数组执行 Huffman；
- V1 将每点 EB ID 改为每个 `16x16` tile 一个原始 ID，再执行两个很小的打包 kernel。

20 次运行的细分中位数为：

| 比较项 | Naive（ms） | V1（ms） | 结论 |
|---|---:|---:|---|
| 仅 U/V Huffman | 3.5800 | 3.6550 | V1 慢约 2.1%，基本可视为持平 |
| Naive 的两个逐点 EB Huffman | 1.5100 | - | V1 删除了这两次完整数组编码 |
| V1 的 Tile-EB pack | - | 0.1310 | 新增，但远小于 1.5100 ms |
| 完整日志中的 Huffman | 5.1955 | 3.6550 | V1 表面快 1.421x |
| Naive Huffman 对比 V1 Huffman+pack | 5.1955 | 3.7795 | V1 快 1.375x |
| Naive Huffman 对比 V1 uniform+Huffman+pack | 5.1955 | 4.0965 | V1 快 1.268x |

所以，对“V1 的 Huffman 是否变快”的准确回答是：

> 对相同类别的 U/V 数组，Huffman 本身没有变快；V1 的 Huffman 阶段总时间下降，
> 是因为它不再 Huffman 编码两个长度为 8,640,000 的逐点 EB 数组。即使计入
> `tile_eb`，再把 `uniform_eb` 也算入这一策略的成本，V1 仍分别保留约 1.375x
> 和 1.268x 的时间优势。

这个比较仍有计时口径限制：`huffman` 包含直方图、直方图 D2H 同步、CPU 码书构建
和 encode，但不包含 Huffman buffer 构造、编码 blob D2H 和内部验证 decode；
`tile_eb` 只包含两个 pack kernel，不包含临时 `cudaMalloc`、host `new`、D2H 和
`cudaFree`。此外，Naive bridge 在计时区间中包含调试直方图扫描和打印。因此这些
数字适合解释当前程序的核心阶段，不应称为严格的 Huffman 端到端微基准。

### 时间收益与空间代价

| 指标 | Naive | V1（16x16） | 变化 |
|---|---:|---:|---:|
| U/V outlier 总数 | 26,871 | 1,507,766 | 56.11x |
| U/V Huffman payload | 9,818,120 bytes | 14,101,564 bytes | +43.63% |
| EB payload | 4,027,240 bytes | 67,500 bytes | -98.32% |
| 特殊值 payload | 2,188,480 bytes | 14,035,640 bytes | 6.41x |
| 完整文件 | 17,113,928 bytes | 29,284,800 bytes | +71.12% |
| 实际文件压缩率 | 4.038816 | 2.360269 | -41.56% |

Tile-EB 确实极大压缩了 EB 元数据，但 tile 最小误差界使大量位置采用更严格的 EB，
从而扩大 U/V Huffman payload，并使 outlier 总数增加到 Naive 的约 56 倍。最终 V1
虽然核心压缩约快 2.05x，但文件大 71.12%，压缩率由 4.039 降至 2.360。

20 次运行中，两版均满足：

- 内存解压与文件解压 U/V 位级一致；
- 解压 EB 验证错误数为 0；
- 原始和解压场均有 20,929 个临界点，`TP=20,929、FP=0、FN=0`。

V1 的实际数值误差也明显小于 Naive：U/V 最大绝对误差由
`15.5392/15.4407` 降到 `0.9899/0.2500`，PSNR 由 `66.226/68.229 dB`
提高到 `92.260/96.202 dB`。这说明当前比较的率失真点并不相同。若要评价算法整体
优劣，还应补充“相同质量”或“相同文件大小”下的参数扫描，而不能只比较单一
`max_relative_eb=0.1` 配置。

### 计时范围说明

程序打印的 `COMPRESS_TIME` 是 CUDA event 时间与 host chrono 时间的相加，不是从
输入到 `.cucpsz` 文件落盘的完整端到端时间。它没有完整包含初始分配/H2D、CPU
land-bitpack 构建、部分计数器拷贝、Huffman buffer/blob 拷贝以及文件写入。外层
进程墙钟时间还混入了解压、文件 I/O、质量计算和 CPU 临界点检查，因此也不能直接
作为压缩时间。当前报告用现有阶段计时回答各个优化步骤的核心性能差异，并明确列出
其边界。

## 2026-08-23 历史测试：范围与环境

- 仓库：`/home/boyi/FaCTz-B`
- 仓库顶层没有 `AGENTS.md`。
- GPU：NVIDIA GeForce RTX 3080，计算能力 8.6，10 GiB。
- CUDA 编译器：nvcc 12.6.85；驱动：610.88。
- 输入：`data/uf.dat` 和 `data/vf.dat`，float32，`2400 x 3600`，每个分量包含 8,640,000 个值。
- 参数：相对误差界 `0.1`。
- 论文依据：`/home/boyi/FaCTz/ppopp27-summer-paper854.pdf`，这是在两个相关仓库中找到的唯一 PDF。论文标题为 *FaCTz: Fast Critical-Point and Topology-Aware GPU Compression for Scientific Vector Fields*。

## 构建模型

两个 Makefile 都是 CMake 的封装，因此实际生效的编译选项来自 CMake 目标及其传递依赖。

- Naive 目标：`cpszg_2d_no_opt_speed`
  - 直接源文件：`naive/cpszg_2d_no_opt_speed.cu` 和 `naive/gpu_huffman_bridge_no_opt_speed.cu`。
  - 共享 CUDA 源文件：通过 `factz_lorenzo` 使用 `cusz/lproto_c.cu` 和 `cusz/lproto_x.cu`。
- V1 目标：`cpszg_2d_v1`
  - 直接源文件：`V1/cpszg_2d_v1.cu`、`V1/gpu_huffman_u2_bridge.cu`，以及与 naive 共用的 Huffman bridge。
  - 仅 V1 使用的辅助文件：`V1/fused_ebzero_kernel.cuh` 和 `V1/tile_uniform_eb.cuh`。
  - 共享 CUDA 源文件：与 naive 使用相同的 `factz_lorenzo` 库。
- 实际生效的 CUDA 编译选项：`-O3 -DNDEBUG -std=c++17 --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations`。
- 宏定义：`PSZ_USE_CUDA=1`，以及由内置 cuSZ/PHF 目标提供的 `PHF_USE_CUDA` 和 `_PORTABLE_USE_CUDA`。
- 架构：项目原有设置为 `70;75;80;86`；测试 GPU 运行 `sm_86` 镜像。
- 主要依赖：CUDA runtime、FTK、内置 cuSZ core/memory/eval 和 PHF。

以下命令均已成功执行：

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

两个由全新 CMake 构建生成的可执行文件均以状态码 0 退出，`make clean` 也成功退出。编译过程中出现了已有的 FTK host/device 警告和忽略 `fread` 返回值的警告，但没有错误。

## 正确性

两个版本分别在不同目录中运行，避免它们生成的 `.out` 和 `.cucpsz` 文件相互覆盖。

| 检查项 | Naive | V1 |
|---|---:|---:|
| 内存中解压结果与文件解压结果对比 | U/V 位级一致 | U/V 位级一致 |
| 解压误差界验证错误数 | U=0，V=0 | U=0，V=0 |
| 原始临界点数量 | 20,929 | 20,929 |
| 解压后临界点数量 | 20,929 | 20,929 |
| TP / FP / FN | 20,929 / 0 / 0 | 20,929 / 0 / 0 |

解压场与原始场之间的浮点比较结果：

| 版本/分量 | 最大绝对误差 | RMSE | 程序计算的 PSNR |
|---|---:|---:|---:|
| Naive U | 15.5392151 | 0.1686684 | 66.2260 |
| Naive V | 15.4406738 | 0.1373586 | 68.2285 |
| V1 U | 0.0624847 | 0.0012557 | 108.7888 |
| V1 V | 0.0622330 | 0.0007691 | 113.2656 |

两个版本重建出的场在数值上并不相同。Naive 与 V1 之间的最大绝对差分别为 15.5371704（U）和 15.4447327（V），RMSE 分别为 0.1686735 和 0.1373616。这是符合预期的，因为 V1 修改了预测器并采用更保守的 tile 公共误差界。相关算法不变量仍然得到保持：按照程序的检查结果，两个版本都满足请求的误差界，并且都保留了全部临界点。

实际 `.cucpsz` 文件大小分别为 17,113,928 bytes（naive）和 27,006,658 bytes（V1）。以 69,120,000 个原始字节计算，对应的可比磁盘压缩率分别为 4.039x 和 2.559x。Naive 打印的 4.311x 没有包含其 1,080,000-byte 陆地掩码，而 V1 打印的 2.559x 包含完整的估算文件大小。因此，跨版本比较应使用实际磁盘文件大小。

## 性能测试方法与结果

每个版本先完整运行一次作为预热，之后在一张无其他负载的 RTX 3080 上交替运行五次。下列时间来自程序已有的内部计时：GPU 阶段和解压使用 CUDA event；Huffman 时间包含 bridge 中的主机端直方图/码书构建以及编码墙钟时间。文件 I/O、CPU 临界点验证和质量分析不计入 `COMPRESS_TIME`。

压缩总时间，单位为毫秒：

| 运行次数 | Naive | V1 |
|---:|---:|---:|
| 1 | 21.946 | 13.292 |
| 2 | 21.821 | 12.676 |
| 3 | 22.651 | 15.884 |
| 4 | 22.539 | 13.215 |
| 5 | 21.188 | 14.897 |
| 中位数 | **21.946** | **13.292** |

压缩时间中位数对应的加速比为 `21.946 / 13.292 = 1.651x`。Naive 和 V1 的吞吐量中位数分别约为 2.93 GiB/s 和 4.84 GiB/s。

| 阶段时间中位数（ms） | Naive | V1 | Naive / V1 |
|---|---:|---:|---:|
| 推导 EB | 4.202 | 3.822 | 1.10x |
| 陆地区域处理 | 0.240 | 0.261 | 0.92x |
| Zero-EB 处理 | 1.304 | 0.234 | 5.57x |
| Tile 统一 EB | - | 0.561 | 新增开销 |
| Lorenzo 压缩 | 10.129 | 2.060 | 4.92x |
| Huffman | 6.249 | 6.286 | 0.99x |
| Tile-EB 打包 | - | 0.106 | 新增开销 |
| 总计 | **21.946** | **13.292** | **1.65x** |

Naive 的解压时间为 `[5.961728, 5.498752, 5.629024, 5.432224, 5.441536]` ms，V1 的解压时间为 `[3.708160, 3.723264, 3.841056, 3.717120, 4.130816]` ms。两者中位数分别为 5.498752 ms 和 3.723264 ms，因此 V1 解压速度提升了 **1.477x**。

## Kernel 级分析

以下标签用于区分源代码事实、实测结果以及基于 CUDA 的解释。

### 1. 误差界推导

- **事实：** `derive_eb_offline_v2` 的函数体保持不变，但 V1 将 `BLOCKSIZE_Y` 从 8 改为 16。启动配置由 `(32,8)`、grid `(120,400)` 变为 `(32,16)`、grid `(120,172)`。两个版本都使用在 x 方向增加一个元素作为 padding 的共享内存分块。
- **实测：** NCU 报告两个版本均使用 33 registers/thread。每个 block 的静态共享内存从 5.28 KiB 翻倍至 10.56 KiB，但实际占用率仍分别为 97.41% 和 97.71%。原始 kernel 执行时间分别为 784.96 us 和 770.62 us；五次运行的阶段时间中位数显示仅获得 1.10x 的小幅提升。
- **解释：** 更高的 block 可以在两行 halo 范围内覆盖更多有效行，并减少启动的总线程数，从而降低重叠计算和重复加载。代价是共享内存翻倍，并使用 512-thread block。数据只支持这是一个小幅改进，不能将其称为主要优化。
- **未启用代码：** V1 包含 `derive_eb_offline_v3`，但实际调用仍然启动 v2，因此不能把 v3 计入 V1 的优化效果。

### 2. 陆地区域处理

- **事实：** V1 在 CPU 上预先计算 `land_id` 和 `land_eb`，不再在 device lambda 中计算 `log2`，但它仍然会启动一次遍历完整场的 Thrust `for_each`。
- **实测：** 时间中位数从 0.240 ms 变为 0.261 ms，因此本次测试不支持这一改动带来了性能提升。
- **解释：** 去除重复计算是合理的，但在当前输入上，这一阶段主要受完整内存扫描和 kernel 启动波动支配。

### 3. Zero-EB 分类与替换

- **事实：** naive 执行一次 `thrust::sequence`、四次 `copy_if` 和四次 `transform`。V1 使用 `kernel_fused_ebzero` 替代这九次全数组操作，在一次遍历中收集 U/V 的值-下标对、重写 ID，并替换过小的 EB。
- **实测：** 阶段时间中位数从 1.304 ms 降至 0.234 ms（5.57x）。NCU 测得融合 kernel 的时间为 220.13 us、DRAM 吞吐利用率为 64.27%、实际占用率为 70.60%、使用 24 registers/thread，并且不使用静态共享内存。
- **解释：** 更少的 kernel 启动次数和更少的完整场重复读取解释了这一提升。代价是使用两个 `atomicAdd` 计数器；zero-EB 密度较高时可能发生竞争。虽然下标和值仍保持配对，但紧凑化附加数据流中的元素顺序无法保证。

### 4. Tile 公共误差界

- **事实：** V1 分别为 U 和 V 启动一次 `kernel_uniformize_tile_eb`。每个 32x32 block 在共享内存中对 1,024 个整数 ID 求最小值，并将最小值广播回所有有效元素。
- **实测：** NCU 测得单个 kernel 耗时 284.03 us；它使用 4.10 KiB 共享内存和 16 registers/thread，实际占用率为 60.59%，理论占用率为 66.67%。两个 kernel 的 CUDA event 时间中位数为 0.561 ms。
- **解释：** 公共误差界使 tiled Lorenzo 成为可能，并将 EB 元数据缩减为每个 tile 1 byte，但 1,024-thread block 和十次归约屏障限制了灵活性。取 tile 最小值会收紧许多位置的误差界：V1 的 U/V outlier 数量从 10,624/16,247 增加到 372,462/379,656，这解释了其压缩率下降的大部分原因。

### 5. Lorenzo 压缩

- **事实：** naive 的逐行 kernel 只启动 75 个、每个包含 32 个线程的 block。一个线程串行遍历一行中的全部 3,600 个值。V1 启动 8,475 个 32x32-thread block；数据值经过量化后，在共享内存 tile 中计算二维 Lorenzo 差分。
- **实测：** 对于一个分量，NCU 测得 naive 耗时 5.41 ms、实际占用率为 2.31%，而 V1 耗时 814.62 us、实际占用率为 65.34%，原始 kernel 加速比为 6.64x。两个分量合计的阶段时间中位数提升了 4.92x。V1 每个 block 使用 4.22 KiB 共享内存和 24 个寄存器/线程。
- **解释：** V1 消除了较长的逐行依赖链，并提供数千个可驻留 block。其代价包括 1,024-thread block、共享内存屏障、彼此独立的 tile 边界，以及更严格公共误差界所导致的大量新增 outlier。

### 6. Lorenzo 解压

- **事实：** naive 同样由一个线程串行扫描一整行。V1 则先在 tile 内沿 x 方向执行共享内存前缀和，再沿 y 方向执行前缀和。
- **实测：** 对于一个分量，NCU 测得 naive 耗时 1.40 ms、占用率为 2.32%，而 V1 耗时 424.96 us、占用率为 64.12%，原始 kernel 性能提升 3.29x。完整解压提升 1.477x，因为测量区间内还包含熵解码、scatter 和初始化操作。
- **解释：** 并行前缀和以增加同步和共享内存使用量为代价，换取了宽得多的并行度。

### 7. 误差界 payload 与 Huffman 路径

- **事实：** naive 对四个完整数组进行 Huffman 编码：U/V 量化码以及 U/V 的逐元素 EB ID。V1 只对 U/V 进行 Huffman 编码，并为每个分量存储 8,475 个原始 EB bytes，将 EB 附加数据从 8,640,000 个 ID 缩减为 `n/1024` 个 ID。
- **实测：** Huffman 时间中位数基本不变，分别为 6.249 ms 和 6.286 ms。V1 的 U/V 编码 blob 和特殊值数据流大得多，因此去掉两个 EB Huffman 数据流并没有减少该数据集上的总时间或文件大小。
- **解释：** 元数据缩减取得了预期效果，但保守的 tile 误差界把开销转移到了量化码熵和 outlier 上。

## 论文与代码的对应关系

当前 V1 是论文分块模式的部分原型，并不是论文 artifact 的精确实现。

| 论文内容 | 当前代码中的对应实现 | 状态 |
|---|---|---|
| 第 3.3 节，四步分块安全误差界 | 每个三角形的解析误差界、关联单元最小值、分量 EB ID，然后取 32x32 tile 最小值 | 概念上匹配 |
| 第 4.1 节，带双顶点 halo 和 x padding 的 32x16 推导 kernel | `derive_eb_offline_v2`、`(32,16)`、共享数组 `[TileDim_X+1]` | 匹配 |
| 第 4.1 节，一个融合的特殊值/tile kernel | 当前 V1 使用一次 Thrust 陆地区域处理、一个原子 zero-EB kernel 和两个独立 tile 归约 kernel | 部分匹配；不是论文中的 kernel |
| 第 4.1 节，`__ballot_sync`、bit mask、CUB scan、无 flag 数组 | CPU 构建的陆地 mask，加上值/下标数组和原子计数器 | 缺失/不同 |
| 第 4.1 节，EB 附加数据为 `n/1024`，以及并行前缀 Lorenzo | 每个分量 8,475 bytes，以及 tiled 32x32 Lorenzo | 匹配 |
| 第 4.1 节，约 8x 的并行 Lorenzo 加速 | 在 RTX 3080 上，NCU 给出的单个压缩 kernel 加速为 6.64x，CUDA event 阶段加速为 4.92x | 方向得到支持；幅度不同 |
| 第 4 节，完整流程均在 device 上执行 | 仍存在 CPU 端 land-bitpack 构建、主机文件 I/O 和主机 Huffman 码书构建/计时 | 不匹配 |
| 第 4 节，默认使用 ANS 后端 | 当前可执行文件仅使用 Huffman | 缺失 |
| 第 4.3 节，并发构建 U/V 直方图与码书 | 当前 U2 bridge 在一个 stream 上顺序处理 U 和 V，并复用一个缓冲区 | 缺失 |
| 图 7，EB/EP/LP/HF 阶段划分 | 程序提供 derive、land/zero/uniform、Lorenzo 和 Huffman 的计时 | 阶段结构可比 |
| 表 1，Ocean 数据集 | 分辨率 2400x3600、临界点数量 20,929，与本次测试一致 | 输入匹配 |

论文表 2 报告的是 A100 上 FaCTz-B 的 Ocean 结果，而本次测试使用 RTX 3080 和一个部分原型。因此，不能把论文发表的数据当作本次运行的预期值。

## 修改的文件

- `CMakeLists.txt`：添加 `FACTZ_BUILD_CUDA_V1`。
- `src/cuda/CMakeLists.txt`：按条件添加 V1 子目录。
- `src/cuda/V1/Makefile`：V1 的 CMake 封装 build/run/clean 入口。
- `src/cuda/V1/CMakeLists.txt`：V1 可执行文件和 Huffman bridge 目标。
- `src/cuda/V1/fused_ebzero_kernel.cuh`：V1 源代码所需的兼容融合 zero-EB 辅助实现。
- `src/cuda/V1/tile_uniform_eb.cuh`：兼容的 32x32 tile 公共 EB 辅助实现。
- `src/cuda/V1/gpu_huffman_u2_bridge.cu`：V1 所需但此前缺失的仅 U2 Huffman bridge。
- `src/cuda/NAIVE_V1_ANALYSIS.md`：本验证报告。

本任务未修改原先已有改动的 `src/cuda/naive/cpszg_2d_no_opt_speed.cu`、`.vscode/` 和 `src/cuda/naive/run_result.txt`。

## 局限性与后续工作

- 结果只覆盖一个数据集、一个误差界和一张 GPU；tile 大小及输入敏感性仍未测试。
- GPU 时钟未锁定。五次运行的中位数限制了噪声影响，但 V1 的 derive-EB 仍表现出明显的运行间波动。
- 在当前驱动/工具组合下，Nsight Systems 捕获了 CUDA API 调用，但没有报告 CUDA kernel 数据，因此使用 NCU 对目标 kernel 进行分析。
- NCU 会重放 kernel，并严重干扰 CUDA event 计时；因此这里只使用它的单 kernel 指标，而加速比来自未进行 profile 的运行。
- 当前 V1 缺少上文列出的若干论文优化，因此将其称为完整论文实现的复现并不准确。
- 现有代码没有检查部分 `fread` 调用的返回值，因此无法可靠拒绝损坏或被截断的 `.cucpsz` 文件；这一问题不在本次构建/性能任务的范围内。
