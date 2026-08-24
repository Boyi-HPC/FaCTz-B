/*
 * nvCOMP ANS integration for cpSZ-GPU.
 *
 * nvCOMP is distributed by NVIDIA under its own license. This file contains
 * only cpSZ-GPU integration code and does not copy nvCOMP implementation code.
 */

#include "gpu_ans_bridge.hpp"

#include <cuda_runtime.h>
#include <nvcomp/ans.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#ifndef CPSZ_ANS_CHUNK_BYTES
#define CPSZ_ANS_CHUNK_BYTES (64 * 1024)
#endif
constexpr size_t kAnsChunkBytes = CPSZ_ANS_CHUNK_BYTES;
constexpr uint16_t kQuantRadius = 4096;
constexpr char kAnsU16LegacyMagic[8] = {'C', 'P', 'A', 'N', 'S', 'U', '2', '\0'};
constexpr char kAnsU16ZigzagMagic[8] = {'C', 'P', 'A', 'N', 'S', 'Z', '2', '\0'};

struct AnsU16FieldHeader {
    char magic[8];
    uint64_t element_count;
    uint64_t low_blob_bytes;
    uint64_t high_blob_bytes;
};

static_assert(sizeof(AnsU16FieldHeader) == 32, "Unexpected ANS U16 header size");

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_nvcomp(nvcompStatus_t status, const char* operation)
{
    if (status != nvcompSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed with nvCOMP status " +
            std::to_string(static_cast<int>(status)));
    }
}

nvcomp::ANSManager make_ans_manager(cudaStream_t stream)
{
    auto compress_opts = nvcompBatchedANSCompressDefaultOpts;
    compress_opts.data_type = NVCOMP_TYPE_UCHAR;
    auto decompress_opts = nvcompBatchedANSDecompressDefaultOpts;
    decompress_opts.data_type = NVCOMP_TYPE_UCHAR;
    return nvcomp::ANSManager(
        kAnsChunkBytes, compress_opts, decompress_opts, stream,
        nvcomp::NoComputeNoVerify, nvcomp::BitstreamKind::NVCOMP_NATIVE);
}

__global__ void split_u16_pair_kernel(
    const uint16_t* __restrict__ in_U,
    const uint16_t* __restrict__ in_V,
    uint8_t* __restrict__ low_U,
    uint8_t* __restrict__ high_U,
    uint8_t* __restrict__ low_V,
    uint8_t* __restrict__ high_V,
    size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int u_delta = static_cast<int>(in_U[i]) - kQuantRadius;
    const int v_delta = static_cast<int>(in_V[i]) - kQuantRadius;
    const uint16_t u = static_cast<uint16_t>(
        u_delta >= 0 ? 2 * u_delta : -2 * u_delta - 1);
    const uint16_t v = static_cast<uint16_t>(
        v_delta >= 0 ? 2 * v_delta : -2 * v_delta - 1);
    low_U[i] = static_cast<uint8_t>(u);
    high_U[i] = static_cast<uint8_t>(u >> 8);
    low_V[i] = static_cast<uint8_t>(v);
    high_V[i] = static_cast<uint8_t>(v >> 8);
}

__global__ void merge_u16_pair_kernel(
    const uint8_t* __restrict__ low_U,
    const uint8_t* __restrict__ high_U,
    const uint8_t* __restrict__ low_V,
    const uint8_t* __restrict__ high_V,
    uint16_t* __restrict__ out_U,
    uint16_t* __restrict__ out_V,
    bool zigzag,
    size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint16_t mapped_U = static_cast<uint16_t>(low_U[i]) |
                              (static_cast<uint16_t>(high_U[i]) << 8);
    const uint16_t mapped_V = static_cast<uint16_t>(low_V[i]) |
                              (static_cast<uint16_t>(high_V[i]) << 8);
    if (zigzag) {
        const int delta_U = (mapped_U & 1u)
            ? -static_cast<int>((mapped_U + 1u) >> 1)
            : static_cast<int>(mapped_U >> 1);
        const int delta_V = (mapped_V & 1u)
            ? -static_cast<int>((mapped_V + 1u) >> 1)
            : static_cast<int>(mapped_V >> 1);
        out_U[i] = static_cast<uint16_t>(static_cast<int>(kQuantRadius) + delta_U);
        out_V[i] = static_cast<uint16_t>(static_cast<int>(kQuantRadius) + delta_V);
    }
    else {
        out_U[i] = mapped_U;
        out_V[i] = mapped_V;
    }
}

void compress_byte_buffers(
    const std::vector<const uint8_t*>& inputs,
    const std::vector<size_t>& input_sizes,
    cudaStream_t stream,
    std::vector<uint8_t*>& device_outputs,
    std::vector<size_t>& output_sizes,
    cudaEvent_t kernel_start,
    cudaEvent_t kernel_stop)
{
    auto manager = make_ans_manager(stream);
    auto configs = manager.configure_compression(input_sizes);
    const size_t count = inputs.size();
    device_outputs.assign(count, nullptr);
    output_sizes.resize(count);
    std::vector<uint8_t*> output_ptrs(count);

    for (size_t i = 0; i < count; ++i) {
        check_cuda(
            cudaMalloc(&device_outputs[i], configs[i].max_compressed_buffer_size),
            "cudaMalloc ANS output");
        output_ptrs[i] = device_outputs[i];
    }

    size_t* d_output_sizes = nullptr;
    check_cuda(cudaMalloc(&d_output_sizes, count * sizeof(size_t)),
               "cudaMalloc ANS output sizes");
    check_cuda(cudaEventRecord(kernel_start, stream), "record ANS start");
    manager.compress(inputs.data(), output_ptrs.data(), configs, d_output_sizes);
    check_cuda(cudaEventRecord(kernel_stop, stream), "record ANS stop");
    check_cuda(cudaEventSynchronize(kernel_stop), "synchronize ANS compression");
    check_cuda(
        cudaMemcpy(output_sizes.data(), d_output_sizes, count * sizeof(size_t),
                   cudaMemcpyDeviceToHost),
        "copy ANS output sizes");

    for (size_t i = 0; i < count; ++i) {
        check_nvcomp(*configs[i].get_status(), "ANS compression");
        if (output_sizes[i] > configs[i].max_compressed_buffer_size) {
            throw std::runtime_error("ANS returned an invalid compressed size");
        }
    }
    check_cuda(cudaFree(d_output_sizes), "cudaFree ANS output sizes");
}

void copy_device_blob(
    uint8_t* d_blob, size_t blob_size,
    uint8_t*& h_blob, size_t& h_blob_size)
{
    h_blob_size = blob_size;
    h_blob = new uint8_t[blob_size ? blob_size : 1];
    if (blob_size != 0) {
        check_cuda(
            cudaMemcpy(h_blob, d_blob, blob_size, cudaMemcpyDeviceToHost),
            "copy ANS blob to host");
    }
}

AnsU16FieldHeader parse_u16_header(
    const uint8_t* blob, size_t blob_len, size_t expected_elements,
    bool& zigzag)
{
    if (blob_len < sizeof(AnsU16FieldHeader)) {
        throw std::runtime_error("Truncated ANS U16 field header");
    }
    AnsU16FieldHeader header{};
    std::memcpy(&header, blob, sizeof(header));
    zigzag = std::memcmp(
        header.magic, kAnsU16ZigzagMagic, sizeof(header.magic)) == 0;
    const bool legacy = std::memcmp(
        header.magic, kAnsU16LegacyMagic, sizeof(header.magic)) == 0;
    if (!zigzag && !legacy) {
        throw std::runtime_error("Invalid ANS U16 field magic");
    }
    if (header.element_count != expected_elements) {
        throw std::runtime_error("ANS U16 element count does not match .cucpsz dimensions");
    }
    if (header.low_blob_bytes > std::numeric_limits<size_t>::max() ||
        header.high_blob_bytes > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("ANS U16 substream is too large for this platform");
    }
    const size_t low_size = static_cast<size_t>(header.low_blob_bytes);
    const size_t high_size = static_cast<size_t>(header.high_blob_bytes);
    if (low_size > blob_len - sizeof(header) ||
        high_size != blob_len - sizeof(header) - low_size) {
        throw std::runtime_error("Invalid ANS U16 substream lengths");
    }
    return header;
}

void decompress_byte_buffers(
    const std::vector<const uint8_t*>& host_inputs,
    const std::vector<size_t>& input_sizes,
    const std::vector<uint8_t*>& device_outputs,
    const std::vector<size_t>& expected_output_sizes,
    cudaStream_t stream)
{
    const size_t count = host_inputs.size();
    std::vector<uint8_t*> device_inputs(count, nullptr);
    std::vector<const uint8_t*> input_ptrs(count);
    try {
        for (size_t i = 0; i < count; ++i) {
            check_cuda(cudaMalloc(&device_inputs[i], input_sizes[i]),
                       "cudaMalloc ANS compressed input");
            check_cuda(
                cudaMemcpyAsync(device_inputs[i], host_inputs[i], input_sizes[i],
                                cudaMemcpyHostToDevice, stream),
                "copy ANS compressed input");
            input_ptrs[i] = device_inputs[i];
        }

        auto manager = make_ans_manager(stream);
        auto configs = manager.configure_decompression(input_ptrs.data(), count);
        for (size_t i = 0; i < count; ++i) {
            if (configs[i].decomp_data_size != expected_output_sizes[i]) {
                throw std::runtime_error("ANS decompressed size does not match expected payload size");
            }
        }
        manager.decompress(device_outputs.data(), input_ptrs.data(), configs);
        check_cuda(cudaStreamSynchronize(stream), "synchronize ANS decompression");
        for (size_t i = 0; i < count; ++i) {
            check_nvcomp(*configs[i].get_status(), "ANS decompression");
        }
    }
    catch (...) {
        for (uint8_t* ptr : device_inputs) if (ptr) cudaFree(ptr);
        throw;
    }
    for (uint8_t* ptr : device_inputs) check_cuda(cudaFree(ptr), "cudaFree ANS input");
}

}  // namespace

struct GpuAnsU2Workspace {
    static constexpr size_t stream_count = 4;

    size_t n = 0;
    size_t stride = 0;
    cudaStream_t stream = nullptr;
    uint8_t* d_planes = nullptr;
    std::array<const uint8_t*, stream_count> inputs{};
    std::array<uint8_t*, stream_count> outputs{};
    size_t* d_output_sizes = nullptr;
    void* d_scratch = nullptr;
    size_t scratch_capacity = 0;
    size_t scratch_requested = 0;
    std::unique_ptr<nvcomp::ANSManager> manager;
    std::vector<nvcomp::CompressionConfig> configs;
    cudaEvent_t split_start = nullptr;
    cudaEvent_t split_stop = nullptr;
    cudaEvent_t compress_start = nullptr;
    cudaEvent_t compress_stop = nullptr;

    GpuAnsU2Workspace(size_t element_count, cudaStream_t user_stream)
        : n(element_count),
          stride((element_count + 255) & ~size_t(255)),
          stream(user_stream)
    {
        try {
            check_cuda(cudaMalloc(&d_planes, stream_count * stride),
                       "allocate persistent ANS byte planes");
            for (size_t i = 0; i < stream_count; ++i) {
                inputs[i] = d_planes + i * stride;
            }

            auto compress_opts = nvcompBatchedANSCompressDefaultOpts;
            compress_opts.data_type = NVCOMP_TYPE_UCHAR;
            auto decompress_opts = nvcompBatchedANSDecompressDefaultOpts;
            decompress_opts.data_type = NVCOMP_TYPE_UCHAR;
            manager = std::make_unique<nvcomp::ANSManager>(
                kAnsChunkBytes, compress_opts, decompress_opts, stream,
                nvcomp::NoComputeNoVerify,
                nvcomp::BitstreamKind::NVCOMP_NATIVE);

            // nvCOMP otherwise calls cudaMallocAsync on the first compress().
            // Reserve the measured ANS requirement plus alignment/version margin
            // so every device allocation stays outside the compression call.
            constexpr size_t scratch_bytes_per_element = 6;
            constexpr size_t scratch_margin = 64ull << 20;
            if (n > (std::numeric_limits<size_t>::max() - scratch_margin) /
                        scratch_bytes_per_element) {
                throw std::overflow_error("ANS scratch capacity overflow");
            }
            scratch_capacity = scratch_bytes_per_element * n + scratch_margin;
            check_cuda(cudaMalloc(&d_scratch, scratch_capacity),
                       "allocate persistent ANS scratch");
            manager->set_scratch_allocators(
                [this](size_t requested) -> void* {
                    scratch_requested = requested;
                    if (requested > scratch_capacity) {
                        throw std::runtime_error(
                            "Preallocated ANS scratch buffer is too small: requested " +
                            std::to_string(requested) + ", capacity " +
                            std::to_string(scratch_capacity));
                    }
                    return d_scratch;
                },
                [](void*, size_t) {});
            configs = manager->configure_compression(
                std::vector<size_t>(stream_count, n));
            for (size_t i = 0; i < stream_count; ++i) {
                check_cuda(cudaMalloc(
                               &outputs[i], configs[i].max_compressed_buffer_size),
                           "allocate persistent ANS output");
            }
            check_cuda(cudaMalloc(
                           &d_output_sizes, stream_count * sizeof(size_t)),
                       "allocate persistent ANS output sizes");
            check_cuda(cudaEventCreate(&split_start), "create ANS split start event");
            check_cuda(cudaEventCreate(&split_stop), "create ANS split stop event");
            check_cuda(cudaEventCreate(&compress_start), "create ANS compression start event");
            check_cuda(cudaEventCreate(&compress_stop), "create ANS compression stop event");
            check_cuda(cudaStreamSynchronize(stream), "finish persistent ANS setup");
        }
        catch (...) {
            release();
            throw;
        }
    }

    ~GpuAnsU2Workspace() { release(); }

    void release()
    {
        if (compress_start) cudaEventDestroy(compress_start);
        if (compress_stop) cudaEventDestroy(compress_stop);
        if (split_start) cudaEventDestroy(split_start);
        if (split_stop) cudaEventDestroy(split_stop);
        compress_start = compress_stop = split_start = split_stop = nullptr;
        if (d_output_sizes) cudaFree(d_output_sizes);
        d_output_sizes = nullptr;
        for (auto& output : outputs) {
            if (output) cudaFree(output);
            output = nullptr;
        }
        if (manager) manager->deallocate_gpu_mem();
        configs.clear();
        manager.reset();
        if (d_scratch) cudaFree(d_scratch);
        d_scratch = nullptr;
        if (d_planes) cudaFree(d_planes);
        d_planes = nullptr;
    }
};

struct GpuAnsU1Workspace {
    static constexpr size_t stream_count = 2;

    size_t n = 0;
    cudaStream_t stream = nullptr;
    std::array<const uint8_t*, stream_count> inputs{};
    std::array<uint8_t*, stream_count> outputs{};
    size_t* d_output_sizes = nullptr;
    void* d_scratch = nullptr;
    size_t scratch_capacity = 0;
    size_t scratch_requested = 0;
    std::unique_ptr<nvcomp::ANSManager> manager;
    std::vector<nvcomp::CompressionConfig> configs;
    cudaEvent_t compress_start = nullptr;
    cudaEvent_t compress_stop = nullptr;

    GpuAnsU1Workspace(size_t element_count, cudaStream_t user_stream)
        : n(element_count), stream(user_stream)
    {
        try {
            auto compress_opts = nvcompBatchedANSCompressDefaultOpts;
            compress_opts.data_type = NVCOMP_TYPE_UCHAR;
            auto decompress_opts = nvcompBatchedANSDecompressDefaultOpts;
            decompress_opts.data_type = NVCOMP_TYPE_UCHAR;
            manager = std::make_unique<nvcomp::ANSManager>(
                kAnsChunkBytes, compress_opts, decompress_opts, stream,
                nvcomp::NoComputeNoVerify,
                nvcomp::BitstreamKind::NVCOMP_NATIVE);

            constexpr size_t scratch_bytes_per_element = 4;
            constexpr size_t scratch_margin = 32ull << 20;
            if (n > (std::numeric_limits<size_t>::max() - scratch_margin) /
                        scratch_bytes_per_element) {
                throw std::overflow_error("ANS U1 scratch capacity overflow");
            }
            scratch_capacity = scratch_bytes_per_element * n + scratch_margin;
            check_cuda(cudaMalloc(&d_scratch, scratch_capacity),
                       "allocate persistent ANS U1 scratch");
            manager->set_scratch_allocators(
                [this](size_t requested) -> void* {
                    scratch_requested = requested;
                    if (requested > scratch_capacity) {
                        throw std::runtime_error(
                            "Preallocated ANS U1 scratch buffer is too small: requested " +
                            std::to_string(requested) + ", capacity " +
                            std::to_string(scratch_capacity));
                    }
                    return d_scratch;
                },
                [](void*, size_t) {});
            configs = manager->configure_compression(
                std::vector<size_t>(stream_count, n));
            for (size_t i = 0; i < stream_count; ++i) {
                check_cuda(cudaMalloc(
                               &outputs[i], configs[i].max_compressed_buffer_size),
                           "allocate persistent ANS U1 output");
            }
            check_cuda(cudaMalloc(
                           &d_output_sizes, stream_count * sizeof(size_t)),
                       "allocate persistent ANS U1 output sizes");
            check_cuda(cudaEventCreate(&compress_start),
                       "create ANS U1 compression start event");
            check_cuda(cudaEventCreate(&compress_stop),
                       "create ANS U1 compression stop event");
            check_cuda(cudaStreamSynchronize(stream),
                       "finish persistent ANS U1 setup");
        }
        catch (...) {
            release();
            throw;
        }
    }

    ~GpuAnsU1Workspace() { release(); }

    void release()
    {
        if (compress_start) cudaEventDestroy(compress_start);
        if (compress_stop) cudaEventDestroy(compress_stop);
        compress_start = compress_stop = nullptr;
        if (d_output_sizes) cudaFree(d_output_sizes);
        d_output_sizes = nullptr;
        for (auto& output : outputs) {
            if (output) cudaFree(output);
            output = nullptr;
        }
        if (manager) manager->deallocate_gpu_mem();
        configs.clear();
        manager.reset();
        if (d_scratch) cudaFree(d_scratch);
        d_scratch = nullptr;
    }
};

struct GpuAnsDecodeWorkspace {
    static constexpr size_t max_stream_count = 6;

    size_t n = 0;
    size_t stride = 0;
    size_t input_count = 0;
    bool include_u1_pair = false;
    bool zigzag = false;
    cudaStream_t stream = nullptr;
    std::array<uint8_t*, max_stream_count> d_inputs{};
    std::array<const uint8_t*, max_stream_count> input_ptrs{};
    std::array<size_t, max_stream_count> input_sizes{};
    uint8_t* d_planes = nullptr;
    void* d_scratch = nullptr;
    size_t scratch_capacity = 0;
    size_t scratch_requested = 0;
    std::unique_ptr<nvcomp::ANSManager> manager;
    std::vector<nvcomp::DecompressionConfig> configs;

    GpuAnsDecodeWorkspace(
        const uint8_t* const h_blobs[4], const size_t blob_lens[4],
        size_t element_count, bool include_u1, cudaStream_t user_stream)
        : n(element_count),
          stride((element_count + 255) & ~size_t(255)),
          input_count(include_u1 ? 6 : 4),
          include_u1_pair(include_u1),
          stream(user_stream)
    {
        try {
            bool zigzag_U = false, zigzag_V = false;
            const AnsU16FieldHeader header_U = parse_u16_header(
                h_blobs[0], blob_lens[0], n, zigzag_U);
            const AnsU16FieldHeader header_V = parse_u16_header(
                h_blobs[1], blob_lens[1], n, zigzag_V);
            if (zigzag_U != zigzag_V) {
                throw std::runtime_error("ANS U/V byte-plane transforms do not match");
            }
            zigzag = zigzag_U;

            const size_t low_U = static_cast<size_t>(header_U.low_blob_bytes);
            const size_t low_V = static_cast<size_t>(header_V.low_blob_bytes);
            const std::array<const uint8_t*, max_stream_count> host_inputs = {
                h_blobs[0] + sizeof(AnsU16FieldHeader),
                h_blobs[0] + sizeof(AnsU16FieldHeader) + low_U,
                h_blobs[1] + sizeof(AnsU16FieldHeader),
                h_blobs[1] + sizeof(AnsU16FieldHeader) + low_V,
                include_u1_pair ? h_blobs[2] : nullptr,
                include_u1_pair ? h_blobs[3] : nullptr};
            input_sizes = {
                low_U, static_cast<size_t>(header_U.high_blob_bytes),
                low_V, static_cast<size_t>(header_V.high_blob_bytes),
                include_u1_pair ? blob_lens[2] : 0,
                include_u1_pair ? blob_lens[3] : 0};

            check_cuda(cudaMalloc(&d_planes, 4 * stride),
                       "allocate persistent ANS decode planes");
            for (size_t i = 0; i < input_count; ++i) {
                if (!host_inputs[i] || input_sizes[i] == 0) {
                    throw std::runtime_error("Invalid ANS compressed substream");
                }
                check_cuda(cudaMalloc(&d_inputs[i], input_sizes[i]),
                           "allocate persistent ANS decode input");
                check_cuda(cudaMemcpyAsync(
                               d_inputs[i], host_inputs[i], input_sizes[i],
                               cudaMemcpyHostToDevice, stream),
                           "upload persistent ANS decode input");
                input_ptrs[i] = d_inputs[i];
            }

            auto compress_opts = nvcompBatchedANSCompressDefaultOpts;
            compress_opts.data_type = NVCOMP_TYPE_UCHAR;
            auto decompress_opts = nvcompBatchedANSDecompressDefaultOpts;
            decompress_opts.data_type = NVCOMP_TYPE_UCHAR;
            manager = std::make_unique<nvcomp::ANSManager>(
                kAnsChunkBytes, compress_opts, decompress_opts, stream,
                nvcomp::NoComputeNoVerify,
                nvcomp::BitstreamKind::NVCOMP_NATIVE);

            constexpr size_t scratch_bytes_per_element = 6;
            constexpr size_t scratch_margin = 64ull << 20;
            if (n > (std::numeric_limits<size_t>::max() - scratch_margin) /
                        scratch_bytes_per_element) {
                throw std::overflow_error("ANS decode scratch capacity overflow");
            }
            scratch_capacity = scratch_bytes_per_element * n + scratch_margin;
            check_cuda(cudaMalloc(&d_scratch, scratch_capacity),
                       "allocate persistent ANS decode scratch");
            manager->set_scratch_allocators(
                [this](size_t requested) -> void* {
                    scratch_requested = requested;
                    if (requested > scratch_capacity) {
                        throw std::runtime_error(
                            "Preallocated ANS decode scratch is too small: requested " +
                            std::to_string(requested) + ", capacity " +
                            std::to_string(scratch_capacity));
                    }
                    return d_scratch;
                },
                [](void*, size_t) {});

            // Configuration inspects compressed metadata, so finish all uploads
            // here and keep that synchronization outside the timed decode call.
            check_cuda(cudaStreamSynchronize(stream),
                       "finish persistent ANS decode uploads");
            configs = manager->configure_decompression(
                input_ptrs.data(), input_count);
            for (size_t i = 0; i < input_count; ++i) {
                if (configs[i].decomp_data_size != n) {
                    throw std::runtime_error(
                        "ANS decompressed size does not match expected payload size");
                }
            }
            check_cuda(cudaStreamSynchronize(stream),
                       "finish persistent ANS decode configuration");
        }
        catch (...) {
            release();
            throw;
        }
    }

    ~GpuAnsDecodeWorkspace() { release(); }

    void release()
    {
        if (manager) manager->deallocate_gpu_mem();
        configs.clear();
        manager.reset();
        if (d_scratch) cudaFree(d_scratch);
        d_scratch = nullptr;
        for (auto& input : d_inputs) {
            if (input) cudaFree(input);
            input = nullptr;
        }
        if (d_planes) cudaFree(d_planes);
        d_planes = nullptr;
    }
};

GpuAnsU2Workspace* create_gpu_ans_u2_workspace(size_t n, void* stream)
{
    return new GpuAnsU2Workspace(n, reinterpret_cast<cudaStream_t>(stream));
}

GpuAnsU1Workspace* create_gpu_ans_u1_workspace(size_t n, void* stream)
{
    return new GpuAnsU1Workspace(n, reinterpret_cast<cudaStream_t>(stream));
}

void destroy_gpu_ans_u1_workspace(GpuAnsU1Workspace* workspace)
{
    delete workspace;
}

void destroy_gpu_ans_u2_workspace(GpuAnsU2Workspace* workspace)
{
    delete workspace;
}

GpuAnsDecodeWorkspace* create_gpu_ans_decode_workspace(
    const uint8_t* const h_blobs[4], const size_t blob_lens[4],
    size_t n, bool include_u1_pair, void* stream)
{
    return new GpuAnsDecodeWorkspace(
        h_blobs, blob_lens, n, include_u1_pair,
        reinterpret_cast<cudaStream_t>(stream));
}

void destroy_gpu_ans_decode_workspace(GpuAnsDecodeWorkspace* workspace)
{
    delete workspace;
}

void run_gpu_ans_decode_device(
    GpuAnsDecodeWorkspace* workspace,
    uint16_t* d_eq_U, uint16_t* d_eq_V,
    uint8_t* d_eq_dEb_U, uint8_t* d_eq_dEb_V)
{
    if (!workspace || !d_eq_U || !d_eq_V) {
        throw std::invalid_argument("Invalid persistent ANS decode arguments");
    }
    if (workspace->include_u1_pair && (!d_eq_dEb_U || !d_eq_dEb_V)) {
        throw std::invalid_argument("Missing persistent ANS U1 decode outputs");
    }

    std::array<uint8_t*, GpuAnsDecodeWorkspace::max_stream_count> outputs = {
        workspace->d_planes,
        workspace->d_planes + workspace->stride,
        workspace->d_planes + 2 * workspace->stride,
        workspace->d_planes + 3 * workspace->stride,
        workspace->include_u1_pair ? d_eq_dEb_U : nullptr,
        workspace->include_u1_pair ? d_eq_dEb_V : nullptr};
    workspace->manager->decompress(
        outputs.data(), workspace->input_ptrs.data(), workspace->configs);

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(
        (workspace->n + block_size - 1) / block_size);
    merge_u16_pair_kernel<<<grid_size, block_size, 0, workspace->stream>>>(
        outputs[0], outputs[1], outputs[2], outputs[3],
        d_eq_U, d_eq_V, workspace->zigzag, workspace->n);
    check_cuda(cudaGetLastError(), "launch persistent ANS byte-plane merge");
}

void check_gpu_ans_decode_status(GpuAnsDecodeWorkspace* workspace)
{
    if (!workspace) throw std::invalid_argument("ANS decode workspace is null");
    for (const auto& config : workspace->configs) {
        check_nvcomp(*config.get_status(), "ANS decompression");
    }
}

void run_gpu_ans_u2_arrays_device(
    GpuAnsU2Workspace* workspace,
    const uint16_t* eq_U, const uint16_t* eq_V,
    size_t n, float* out_kernel_ms)
{
    if (!workspace || workspace->n != n) {
        throw std::runtime_error("ANS workspace size mismatch");
    }

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    check_cuda(cudaEventRecord(workspace->split_start, workspace->stream),
               "record persistent ANS split start");
    split_u16_pair_kernel<<<grid_size, block_size, 0, workspace->stream>>>(
        eq_U, eq_V,
        const_cast<uint8_t*>(workspace->inputs[0]),
        const_cast<uint8_t*>(workspace->inputs[1]),
        const_cast<uint8_t*>(workspace->inputs[2]),
        const_cast<uint8_t*>(workspace->inputs[3]), n);
    check_cuda(cudaGetLastError(), "launch persistent ANS byte-plane split");
    check_cuda(cudaEventRecord(workspace->split_stop, workspace->stream),
               "record persistent ANS split stop");

    check_cuda(cudaEventRecord(workspace->compress_start, workspace->stream),
               "record persistent ANS compression start");
    workspace->manager->compress(
        workspace->inputs.data(), workspace->outputs.data(),
        workspace->configs, workspace->d_output_sizes);
    check_cuda(cudaEventRecord(workspace->compress_stop, workspace->stream),
               "record persistent ANS compression stop");
    check_cuda(cudaEventSynchronize(workspace->compress_stop),
               "finish persistent ANS compression");

    float split_ms = 0.0f, compress_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(
                   &split_ms, workspace->split_start, workspace->split_stop),
               "measure persistent ANS split");
    check_cuda(cudaEventElapsedTime(
                   &compress_ms, workspace->compress_start, workspace->compress_stop),
               "measure persistent ANS compression");
    *out_kernel_ms = split_ms + compress_ms;
}

void download_gpu_ans_u2_arrays(
    GpuAnsU2Workspace* workspace,
    uint8_t* out_h_blobs[2], size_t out_lens[2])
{
    if (!workspace) throw std::invalid_argument("ANS workspace is null");

    std::array<size_t, GpuAnsU2Workspace::stream_count> compressed_sizes{};
    check_cuda(cudaMemcpy(
                   compressed_sizes.data(), workspace->d_output_sizes,
                   compressed_sizes.size() * sizeof(size_t),
                   cudaMemcpyDeviceToHost),
               "download persistent ANS output sizes");
    for (size_t i = 0; i < compressed_sizes.size(); ++i) {
        check_nvcomp(*workspace->configs[i].get_status(), "ANS compression");
        if (compressed_sizes[i] > workspace->configs[i].max_compressed_buffer_size) {
            throw std::runtime_error("ANS returned an invalid compressed size");
        }
    }

    for (int field = 0; field < 2; ++field) {
        const size_t low_index = 2 * field;
        const size_t high_index = low_index + 1;
        AnsU16FieldHeader header{};
        std::memcpy(header.magic, kAnsU16ZigzagMagic, sizeof(header.magic));
        header.element_count = workspace->n;
        header.low_blob_bytes = compressed_sizes[low_index];
        header.high_blob_bytes = compressed_sizes[high_index];
        out_lens[field] = sizeof(header) + compressed_sizes[low_index] +
                          compressed_sizes[high_index];
        out_h_blobs[field] = new uint8_t[out_lens[field]];
        std::memcpy(out_h_blobs[field], &header, sizeof(header));
        check_cuda(cudaMemcpy(
                       out_h_blobs[field] + sizeof(header),
                       workspace->outputs[low_index], compressed_sizes[low_index],
                       cudaMemcpyDeviceToHost),
                   "download persistent ANS low-byte stream");
        check_cuda(cudaMemcpy(
                       out_h_blobs[field] + sizeof(header) + compressed_sizes[low_index],
                       workspace->outputs[high_index], compressed_sizes[high_index],
                       cudaMemcpyDeviceToHost),
                   "download persistent ANS high-byte stream");
    }
}

void run_gpu_ans_u1_arrays_device(
    GpuAnsU1Workspace* workspace,
    const uint8_t* eq_U, const uint8_t* eq_V,
    size_t n, float* out_kernel_ms)
{
    if (!workspace || workspace->n != n || !eq_U || !eq_V) {
        throw std::runtime_error("ANS U1 workspace arguments mismatch");
    }
    workspace->inputs[0] = eq_U;
    workspace->inputs[1] = eq_V;
    check_cuda(cudaEventRecord(workspace->compress_start, workspace->stream),
               "record persistent ANS U1 compression start");
    workspace->manager->compress(
        workspace->inputs.data(), workspace->outputs.data(),
        workspace->configs, workspace->d_output_sizes);
    check_cuda(cudaEventRecord(workspace->compress_stop, workspace->stream),
               "record persistent ANS U1 compression stop");
    check_cuda(cudaEventSynchronize(workspace->compress_stop),
               "finish persistent ANS U1 compression");
    check_cuda(cudaEventElapsedTime(
                   out_kernel_ms, workspace->compress_start,
                   workspace->compress_stop),
               "measure persistent ANS U1 compression");
}

void download_gpu_ans_u1_arrays(
    GpuAnsU1Workspace* workspace,
    uint8_t* out_h_blobs[2], size_t out_lens[2])
{
    if (!workspace) throw std::invalid_argument("ANS U1 workspace is null");
    std::array<size_t, GpuAnsU1Workspace::stream_count> compressed_sizes{};
    check_cuda(cudaMemcpy(
                   compressed_sizes.data(), workspace->d_output_sizes,
                   compressed_sizes.size() * sizeof(size_t),
                   cudaMemcpyDeviceToHost),
               "download persistent ANS U1 output sizes");
    for (size_t i = 0; i < compressed_sizes.size(); ++i) {
        check_nvcomp(*workspace->configs[i].get_status(), "ANS U1 compression");
        if (compressed_sizes[i] > workspace->configs[i].max_compressed_buffer_size) {
            throw std::runtime_error("ANS U1 returned an invalid compressed size");
        }
        copy_device_blob(
            workspace->outputs[i], compressed_sizes[i],
            out_h_blobs[i], out_lens[i]);
    }
}

void run_gpu_ans_u2_arrays_timed(
    const uint16_t* eq_U, const uint16_t* eq_V,
    size_t n, void* stream_ptr,
    size_t out_lens[2], uint8_t* out_h_blobs[2],
    float* out_kernel_ms)
{
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const size_t stride = (n + 255) & ~size_t(255);
    uint8_t* d_planes = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaEvent_t split_start = nullptr, split_stop = nullptr;
    std::vector<uint8_t*> device_outputs;

    try {
        check_cuda(cudaMalloc(&d_planes, 4 * stride), "cudaMalloc ANS byte planes");
        check_cuda(cudaEventCreate(&start), "create ANS start event");
        check_cuda(cudaEventCreate(&stop), "create ANS stop event");
        check_cuda(cudaEventCreate(&split_start), "create ANS split start event");
        check_cuda(cudaEventCreate(&split_stop), "create ANS split stop event");

        uint8_t* planes[4] = {
            d_planes, d_planes + stride, d_planes + 2 * stride, d_planes + 3 * stride};
        constexpr int block_size = 256;
        const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
        check_cuda(cudaEventRecord(split_start, stream), "record ANS split start");
        split_u16_pair_kernel<<<grid_size, block_size, 0, stream>>>(
            eq_U, eq_V, planes[0], planes[1], planes[2], planes[3], n);
        check_cuda(cudaGetLastError(), "launch ANS byte-plane split");
        check_cuda(cudaEventRecord(split_stop, stream), "record ANS split stop");

        std::vector<const uint8_t*> inputs = {
            planes[0], planes[1], planes[2], planes[3]};
        std::vector<size_t> input_sizes(4, n);
        std::vector<size_t> compressed_sizes;
        compress_byte_buffers(
            inputs, input_sizes, stream, device_outputs, compressed_sizes,
            start, stop);
        float split_ms = 0.0f, compress_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&split_ms, split_start, split_stop),
                   "measure ANS byte-plane split");
        check_cuda(cudaEventElapsedTime(&compress_ms, start, stop),
                   "measure ANS kernels");
        *out_kernel_ms = split_ms + compress_ms;

        for (int field = 0; field < 2; ++field) {
            const size_t low_index = 2 * field;
            const size_t high_index = low_index + 1;
            AnsU16FieldHeader header{};
            std::memcpy(header.magic, kAnsU16ZigzagMagic, sizeof(header.magic));
            header.element_count = n;
            header.low_blob_bytes = compressed_sizes[low_index];
            header.high_blob_bytes = compressed_sizes[high_index];
            const size_t field_size = sizeof(header) + compressed_sizes[low_index] +
                                      compressed_sizes[high_index];
            out_h_blobs[field] = new uint8_t[field_size];
            out_lens[field] = field_size;
            std::memcpy(out_h_blobs[field], &header, sizeof(header));
            check_cuda(
                cudaMemcpy(out_h_blobs[field] + sizeof(header),
                           device_outputs[low_index], compressed_sizes[low_index],
                           cudaMemcpyDeviceToHost),
                "copy ANS low-byte stream");
            check_cuda(
                cudaMemcpy(out_h_blobs[field] + sizeof(header) + compressed_sizes[low_index],
                           device_outputs[high_index], compressed_sizes[high_index],
                           cudaMemcpyDeviceToHost),
                "copy ANS high-byte stream");
        }
    }
    catch (...) {
        for (uint8_t* ptr : device_outputs) if (ptr) cudaFree(ptr);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (split_start) cudaEventDestroy(split_start);
        if (split_stop) cudaEventDestroy(split_stop);
        if (d_planes) cudaFree(d_planes);
        throw;
    }

    for (uint8_t* ptr : device_outputs) check_cuda(cudaFree(ptr), "cudaFree ANS output");
    check_cuda(cudaEventDestroy(start), "destroy ANS start event");
    check_cuda(cudaEventDestroy(stop), "destroy ANS stop event");
    check_cuda(cudaEventDestroy(split_start), "destroy ANS split start event");
    check_cuda(cudaEventDestroy(split_stop), "destroy ANS split stop event");
    check_cuda(cudaFree(d_planes), "cudaFree ANS byte planes");
}

void run_gpu_ans_u1_arrays_timed(
    const uint8_t* eq_U, const uint8_t* eq_V,
    size_t n, void* stream_ptr,
    size_t out_lens[2], uint8_t* out_h_blobs[2],
    float* out_kernel_ms)
{
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaEvent_t start = nullptr, stop = nullptr;
    std::vector<uint8_t*> device_outputs;
    try {
        check_cuda(cudaEventCreate(&start), "create ANS start event");
        check_cuda(cudaEventCreate(&stop), "create ANS stop event");
        std::vector<const uint8_t*> inputs = {eq_U, eq_V};
        std::vector<size_t> input_sizes(2, n);
        std::vector<size_t> compressed_sizes;
        compress_byte_buffers(
            inputs, input_sizes, stream, device_outputs, compressed_sizes,
            start, stop);
        check_cuda(cudaEventElapsedTime(out_kernel_ms, start, stop),
                   "measure ANS kernels");
        for (int i = 0; i < 2; ++i) {
            copy_device_blob(
                device_outputs[i], compressed_sizes[i],
                out_h_blobs[i], out_lens[i]);
        }
    }
    catch (...) {
        for (uint8_t* ptr : device_outputs) if (ptr) cudaFree(ptr);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        throw;
    }
    for (uint8_t* ptr : device_outputs) check_cuda(cudaFree(ptr), "cudaFree ANS output");
    check_cuda(cudaEventDestroy(start), "destroy ANS start event");
    check_cuda(cudaEventDestroy(stop), "destroy ANS stop event");
}

void gpu_ans_u2_decode_pair_from_blobs(
    const uint8_t* h_blob_U, size_t blob_len_U,
    const uint8_t* h_blob_V, size_t blob_len_V,
    size_t n, uint16_t* d_eq_U, uint16_t* d_eq_V,
    void* stream_ptr)
{
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    bool zigzag_U = false, zigzag_V = false;
    const AnsU16FieldHeader header_U = parse_u16_header(
        h_blob_U, blob_len_U, n, zigzag_U);
    const AnsU16FieldHeader header_V = parse_u16_header(
        h_blob_V, blob_len_V, n, zigzag_V);
    if (zigzag_U != zigzag_V) {
        throw std::runtime_error("ANS U/V byte-plane transforms do not match");
    }
    const size_t low_U_size = static_cast<size_t>(header_U.low_blob_bytes);
    const size_t low_V_size = static_cast<size_t>(header_V.low_blob_bytes);
    const size_t stride = (n + 255) & ~size_t(255);
    uint8_t* d_planes = nullptr;

    check_cuda(cudaMalloc(&d_planes, 4 * stride), "cudaMalloc ANS decoded planes");
    try {
        std::vector<const uint8_t*> inputs = {
            h_blob_U + sizeof(AnsU16FieldHeader),
            h_blob_U + sizeof(AnsU16FieldHeader) + low_U_size,
            h_blob_V + sizeof(AnsU16FieldHeader),
            h_blob_V + sizeof(AnsU16FieldHeader) + low_V_size};
        std::vector<size_t> input_sizes = {
            low_U_size, static_cast<size_t>(header_U.high_blob_bytes),
            low_V_size, static_cast<size_t>(header_V.high_blob_bytes)};
        std::vector<uint8_t*> outputs = {
            d_planes, d_planes + stride, d_planes + 2 * stride, d_planes + 3 * stride};
        std::vector<size_t> output_sizes(4, n);
        decompress_byte_buffers(inputs, input_sizes, outputs, output_sizes, stream);

        constexpr int block_size = 256;
        const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
        merge_u16_pair_kernel<<<grid_size, block_size, 0, stream>>>(
            outputs[0], outputs[1], outputs[2], outputs[3],
            d_eq_U, d_eq_V, zigzag_U, n);
        check_cuda(cudaGetLastError(), "launch ANS byte-plane merge");
        check_cuda(cudaStreamSynchronize(stream), "synchronize ANS byte-plane merge");
    }
    catch (...) {
        cudaFree(d_planes);
        throw;
    }
    check_cuda(cudaFree(d_planes), "cudaFree ANS decoded planes");
}

void gpu_ans_u1_decode_pair_from_blobs(
    const uint8_t* h_blob_U, size_t blob_len_U,
    const uint8_t* h_blob_V, size_t blob_len_V,
    size_t n, uint8_t* d_eq_U, uint8_t* d_eq_V,
    void* stream_ptr)
{
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    std::vector<const uint8_t*> inputs = {h_blob_U, h_blob_V};
    std::vector<size_t> input_sizes = {blob_len_U, blob_len_V};
    std::vector<uint8_t*> outputs = {d_eq_U, d_eq_V};
    std::vector<size_t> output_sizes(2, n);
    decompress_byte_buffers(inputs, input_sizes, outputs, output_sizes, stream);
}
