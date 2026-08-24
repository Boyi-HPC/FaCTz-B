# CUDA Implementations

该目录保存 FaCTz-B 的共享 CUDA 组件和各阶段实现。完整构建、输入格式和运行命令请查看仓库根目录的 `README.md`。

```text
cuda/
|-- CMakeLists.txt       共享 CUDA targets 和版本开关
|-- cusz/                共享 Lorenzo 实现
|-- tools/               通用解压器和正确性 verifier
|-- naive/               未优化基线
|-- V1/                  tiled Lorenzo 与初始融合版本
|-- V1.1/                bitmask side-stream 中间版本
|-- V1.2/                paired tile EB 和并行 Huffman
|-- V1.3/                fused special classification 与独立解压器
|-- V1.4/                Huffman/ANS codec 切换
`-- V1.5/                最终 GPU-resident app 压缩/解压路径
```

各版本可使用统一脚本一次构建：

```bash
./build_script.sh
```

也可以只构建或运行某一个版本：

```bash
make -C src/cuda/V1.3
make -C src/cuda/V1.3 run

make -C src/cuda/V1.5 run CODEC=hf
make -C src/cuda/V1.5 verify CODEC=hf
```

版本目录下的 `*_ANALYSIS.md` 记录了构建验证、性能结果和 kernel 优化分析。
