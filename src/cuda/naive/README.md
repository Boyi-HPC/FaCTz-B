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
