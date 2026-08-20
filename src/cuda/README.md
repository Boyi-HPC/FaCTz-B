# CUDA Implementations

Shared CUDA build options and customized Lorenzo sources live in this
directory. Each compression pipeline keeps its implementation-specific source,
build target, and usage notes in a subdirectory.

```text
cuda/
|-- CMakeLists.txt        Shared CUDA targets
|-- cusz/                 Shared customized Lorenzo implementation
|-- lorenzo_tile_dim.h    Shared Lorenzo configuration
`-- naive/                Unoptimized CUDA baseline
```

Add future versions as sibling directories and link them against
`factz::cuda_options` and `factz::lorenzo`.
