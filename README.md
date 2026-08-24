# FaCTz-B

FaCTz-B 是二维向量场 critical-point preserving 有损压缩的 CUDA 实验项目。本仓库保留了从 naive 到 V1.5 的多个实现，便于复现构建、正确性验证和性能演进。

## 依赖

- Linux x86_64、支持 CUDA 的 NVIDIA GPU；
- CMake 3.21 或更新版本；
- 支持 C++17 的编译器、`make`、`git`；
- CUDA Toolkit 和 `nvcc`；
- ZSTD 开发库；
- NVIDIA nvCOMP 5（构建 V1.4/V1.5 需要）；
- FTK。`build_script.sh` 会在首次构建时克隆指定版本到 `external/ftk` 并安装到仓库内部。

nvCOMP 根目录必须包含：

```text
include/nvcomp/ans.hpp
lib64/libnvcomp.so 或 lib64/libnvcomp.so.5
```

脚本默认查找 `$HOME/.local/nvcomp-cu12/nvidia/libnvcomp`；安装在其他位置时设置 `NVCOMP_ROOT`。

## 构建整个项目

首次使用先赋予脚本执行权限，然后构建：

```bash
cd /path/to/FaCTz-B
chmod +x build_script.sh
./build_script.sh
```

脚本会构建 CPU targets、naive、V1、V1.1、V1.2、V1.3、V1.4、V1.5，以及公共解压和验证工具。可执行文件位于 `build/bin/`。

常用操作：

```bash
./build_script.sh build       # 默认行为：配置并增量构建全部版本
./build_script.sh clean       # 调用 CMake clean target
./build_script.sh rebuild     # clean 后重新配置和构建
./build_script.sh help
```

可通过环境变量覆盖本机配置：

```bash
CUDA_ARCHITECTURES=86 BUILD_JOBS=8 ./build_script.sh

CUDACXX=/usr/local/cuda-12/bin/nvcc \
NVCOMP_ROOT=$HOME/.local/nvcomp-cu12/nvidia/libnvcomp \
BUILD_DIR=$PWD/build-release \
./build_script.sh
```

支持的变量包括 `BUILD_DIR`、`BUILD_TYPE`、`BUILD_JOBS`、`CUDACXX`、`CUDA_ARCHITECTURES` 和 `NVCOMP_ROOT`。FTK 已经安装且不想重复构建时可使用：

```bash
SKIP_FTK_BUILD=1 ./build_script.sh
```

## 输入格式

所有版本的公共压缩参数是：

```text
U.bin V.bin r1 r2 max_relative_eb
```

`U.bin` 和 `V.bin` 是无文件头的 `float32` 二进制数组，均应恰好包含 `r1 * r2` 个元素。仓库示例数据为：

```text
data/uf.dat data/vf.dat 2400 3600 0.1
```

V1.4/V1.5 还接受可选 codec 参数 `hf` 或 `ans`；不指定时默认为 `hf`。

## 运行各版本

每个版本目录中的 Makefile 都是根 CMake 构建的便捷封装。以下命令使用仓库自带 Ocean 数据：

| 版本 | 构建并运行命令 | 行为 |
|---|---|---|
| naive | `make -C src/cuda/naive run` | 压缩、内置解压和正确性输出 |
| V1 | `make -C src/cuda/V1 run` | 压缩、内置解压和正确性输出 |
| V1.1 | `make -C src/cuda/V1.1 run` | 压缩、内置解压和正确性输出 |
| V1.2 | `make -C src/cuda/V1.2 run` | 生成 Huffman `.cucpsz` |
| V1.3 | `make -C src/cuda/V1.3 run` | 压缩后调用 V1.3 独立解压器 |
| V1.4 HF | `make -C src/cuda/V1.4 run CODEC=hf` | 生成 Huffman `.cucpsz` |
| V1.4 ANS | `make -C src/cuda/V1.4 run CODEC=ans` | 生成 ANS `.cucpsz` |
| V1.5 HF | `make -C src/cuda/V1.5 run CODEC=hf` | GPU-resident 压缩和独立解压 |
| V1.5 ANS | `make -C src/cuda/V1.5 run CODEC=ans` | GPU-resident 压缩和独立解压 |

自定义输入时覆盖 Makefile 变量：

```bash
make -C src/cuda/V1.3 run \
  U=/path/U.bin V=/path/V.bin R1=2400 R2=3600 EB=0.1
```

`naive` 和 `V1` 会把 `.cucpsz`/`.out` 写在输入文件旁；V1.1 及之后版本默认写入 `build/runs/<version>/`。V1.4 的 HF/ANS 结果建议使用不同目录，避免覆盖：

```bash
make -C src/cuda/V1.4 run CODEC=hf  RUN_DIR=$PWD/build/runs/v1.4/hf
make -C src/cuda/V1.4 run CODEC=ans RUN_DIR=$PWD/build/runs/v1.4/ans
```

## 直接运行可执行文件

已经执行过 `build_script.sh`，不想再次触发构建时，可以直接调用 `build/bin` 下的程序：

```bash
# naive / V1 / V1.1 / V1.2 / V1.3
./build/bin/cpszg_2d_no_opt_speed data/uf.dat data/vf.dat 2400 3600 0.1
./build/bin/cpszg_2d_v1           data/uf.dat data/vf.dat 2400 3600 0.1
./build/bin/cpszg_2d_v1_1         data/uf.dat data/vf.dat 2400 3600 0.1
./build/bin/cpszg_2d_v1_2         data/uf.dat data/vf.dat 2400 3600 0.1
./build/bin/cpszg_2d_v1_3         data/uf.dat data/vf.dat 2400 3600 0.1

# V1.4 / V1.5：最后一个参数选择 entropy codec
./build/bin/cpszg_2d_v1_4 data/uf.dat data/vf.dat 2400 3600 0.1 hf
./build/bin/cpszg_2d_v1_5 data/uf.dat data/vf.dat 2400 3600 0.1 ans
```

压缩文件名由程序使用 `U` 输入名加 `.cucpsz` 得到，例如 `data/uf.dat.cucpsz`。

独立解压命令：

```bash
# V1.3 历史 Huffman 解压器
./build/bin/decpszg_2d_v1_3 \
  input.cucpsz U.out V.out

# 通用 HF/ANS 解压器，可用于 V1.2 及后续兼容格式
./build/bin/factz_decpszg_2d \
  input.cucpsz U.out V.out hf

# V1.5 app 解压器
./build/bin/decpszg_2d_v1_5 \
  input.cucpsz U.out V.out ans
```

codec 必须与压缩时一致。

## 正确性验证

V1.5 Makefile 可以直接运行公共 verifier：

```bash
make -C src/cuda/V1.5 run CODEC=hf
make -C src/cuda/V1.5 verify CODEC=hf
```

也可以手动比较任意恢复结果：

```bash
./build/bin/factz_verify_cpszg_2d \
  data/uf.dat data/vf.dat \
  U.out V.out \
  2400 3600
```

verifier 报告最大绝对/相对误差、RMSE、PSNR，以及原始和恢复向量场的 critical-point 匹配结果。

## Targets 和分析文档

| 版本 | CMake target | 分析报告 |
|---|---|---|
| naive | `cpszg_2d_no_opt_speed` | `src/cuda/naive/README.md` |
| V1 | `cpszg_2d_v1` | `src/cuda/V1/NAIVE_V1_ANALYSIS.md` |
| V1.1 | `cpszg_2d_v1_1` | `src/cuda/V1.1/V1_1_ANALYSIS.md` |
| V1.2 | `cpszg_2d_v1_2` | `src/cuda/V1.2/V1_2_ANALYSIS.md` |
| V1.3 | `cpszg_2d_v1_3`, `decpszg_2d_v1_3` | `src/cuda/V1.3/V1_3_ANALYSIS.md` |
| V1.4 | `cpszg_2d_v1_4` | `src/cuda/V1.4/V1_4_ANALYSIS.md` |
| V1.5 | `cpszg_2d_v1_5`, `decpszg_2d_v1_5` | `src/cuda/V1.5/V1_5_ANALYSIS.md` |

各报告记录了对应版本的正确性、性能数据、kernel 优化和论文对应关系。
