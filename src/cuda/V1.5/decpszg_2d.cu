#include "cucpsz_2d_format.hpp"
#include "gpu_ans_bridge.hpp"
#include "gpu_huffman_bridge.hpp"
#include "lorenzo_tile_dim.h"
#include "kernel/lrz/lproto.hh"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kRadius = 4096;
constexpr float kThreshold = 1.0f / (1 << 20);

void check_cuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

template <typename T>
void read_items(FILE* file, std::vector<T>& values, size_t count, const char* label)
{
    values.resize(count);
    if (count != 0 && fread(values.data(), sizeof(T), count, file) != count) {
        throw std::runtime_error(std::string("Truncated .cucpsz payload: ") + label);
    }
}

void validate_indices(const std::vector<uint32_t>& indices, size_t n, const char* label)
{
    for (uint32_t index : indices) {
        if (index >= n) {
            throw std::runtime_error(std::string("Out-of-range outlier index in ") + label);
        }
    }
}

struct HostCompressedPayload2D {
    CucpszHeader header{};
    CucpszCodec codec = CucpszCodec::Huffman;
    size_t r1 = 0;
    size_t r2 = 0;
    size_t n = 0;
    bool tile_variant = false;
    uint32_t tile_dim = 0;
    std::array<std::vector<uint8_t>, 4> blobs;
    std::vector<uint8_t> land_bitpack;
    std::vector<uint32_t> ot_idx_U;
    std::vector<uint32_t> ot_idx_V;
    std::vector<float> ot_val_U;
    std::vector<float> ot_val_V;
    std::vector<uint8_t> zeroeb_mask_U;
    std::vector<uint8_t> zeroeb_mask_V;
    std::vector<float> zeroeb_val_U;
    std::vector<float> zeroeb_val_V;
};

HostCompressedPayload2D read_cucpsz_file(
    const char* filename, CucpszCodec requested_codec)
{
    FILE* file = fopen(filename, "rb");
    if (!file) throw std::runtime_error(std::string("Cannot open ") + filename);

    HostCompressedPayload2D payload;
    try {
        if (fread(&payload.header, sizeof(payload.header), 1, file) != 1) {
            throw std::runtime_error("Cannot read .cucpsz header");
        }
        if (!cucpsz_header_has_valid_magic(payload.header)) {
            throw std::runtime_error("Invalid .cucpsz magic");
        }
        if (!cucpsz_header_has_supported_codec(payload.header)) {
            throw std::runtime_error("Unsupported codec in .cucpsz header");
        }
        payload.codec = static_cast<CucpszCodec>(payload.header.codec);
        if (payload.codec != requested_codec) {
            throw std::runtime_error(
                std::string("Codec mismatch: file uses ") +
                cucpsz_codec_name(payload.codec) + ", command requested " +
                cucpsz_codec_name(requested_codec));
        }
        if (payload.header.r1 == 0 || payload.header.r2 == 0 ||
            payload.header.r1 > std::numeric_limits<size_t>::max() / payload.header.r2) {
            throw std::runtime_error("Invalid dimensions in .cucpsz header");
        }
        if (payload.header.lorenzo_variant > 1u) {
            throw std::runtime_error("Unsupported Lorenzo variant in .cucpsz header");
        }

        payload.r1 = static_cast<size_t>(payload.header.r1);
        payload.r2 = static_cast<size_t>(payload.header.r2);
        payload.n = payload.r1 * payload.r2;
        payload.tile_variant = payload.header.lorenzo_variant == 1u;
        payload.tile_dim = payload.header.lorenzo_tile_dim
            ? payload.header.lorenzo_tile_dim
            : static_cast<uint32_t>(LORENZO_TILE_DIM);
        if (payload.header.ot_count_U > payload.n ||
            payload.header.ot_count_V > payload.n ||
            payload.header.zeroeb_count_U > payload.n ||
            payload.header.zeroeb_count_V > payload.n) {
            throw std::runtime_error("Invalid special-value count in .cucpsz header");
        }
        if (payload.tile_variant && payload.tile_dim != 8 &&
            payload.tile_dim != 16 && payload.tile_dim != 32) {
            throw std::runtime_error("Unsupported Lorenzo tile dimension");
        }

        for (int i = 0; i < 4; ++i) {
            if (payload.header.hf_blob_len[i] >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::runtime_error("Payload is too large for this platform");
            }
            read_items(
                file, payload.blobs[i],
                static_cast<size_t>(payload.header.hf_blob_len[i]), "codec blob");
        }

        if (payload.tile_variant) {
            const size_t tile_cols =
                (payload.r2 + payload.tile_dim - 1) / payload.tile_dim;
            const size_t tile_rows =
                (payload.r1 + payload.tile_dim - 1) / payload.tile_dim;
            if (tile_cols != 0 &&
                tile_rows > std::numeric_limits<size_t>::max() / tile_cols) {
                throw std::runtime_error("Tile count overflow");
            }
            const size_t tile_bytes = tile_rows * tile_cols;
            if (payload.blobs[2].size() != tile_bytes ||
                payload.blobs[3].size() != tile_bytes) {
                throw std::runtime_error("Invalid tile error-bound payload size");
            }
        }

        const size_t expected_land_bytes = (payload.n + 7) / 8;
        if (payload.header.land_bitpack_bytes != 0 &&
            payload.header.land_bitpack_bytes != expected_land_bytes) {
            throw std::runtime_error("Invalid land bitpack size");
        }
        read_items(
            file, payload.land_bitpack,
            static_cast<size_t>(payload.header.land_bitpack_bytes), "land bitpack");

        const size_t mask_bytes_U = payload.header.zeroeb_count_U
            ? bitmask_word_bytes(payload.n) : 0;
        const size_t mask_bytes_V = payload.header.zeroeb_count_V
            ? bitmask_word_bytes(payload.n) : 0;
        read_items(file, payload.ot_idx_U, payload.header.ot_count_U,
                   "U outlier indices");
        read_items(file, payload.ot_val_U, payload.header.ot_count_U,
                   "U outlier values");
        read_items(file, payload.zeroeb_mask_U, mask_bytes_U,
                   "U zero-EB mask");
        read_items(file, payload.zeroeb_val_U, payload.header.zeroeb_count_U,
                   "U zero-EB values");
        read_items(file, payload.ot_idx_V, payload.header.ot_count_V,
                   "V outlier indices");
        read_items(file, payload.ot_val_V, payload.header.ot_count_V,
                   "V outlier values");
        read_items(file, payload.zeroeb_mask_V, mask_bytes_V,
                   "V zero-EB mask");
        read_items(file, payload.zeroeb_val_V, payload.header.zeroeb_count_V,
                   "V zero-EB values");

        fclose(file);
        file = nullptr;
        validate_indices(payload.ot_idx_U, payload.n, "U");
        validate_indices(payload.ot_idx_V, payload.n, "V");
        return payload;
    }
    catch (...) {
        if (file) fclose(file);
        throw;
    }
}

__global__ void kernel_scatter_values(
    const uint32_t* __restrict__ indices,
    const float* __restrict__ values,
    uint32_t count,
    float* __restrict__ output)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[indices[i]] = values[i];
}

__global__ void kernel_mask_word_counts_pair(
    const uint32_t* __restrict__ mask_U,
    const uint32_t* __restrict__ mask_V,
    uint32_t* __restrict__ offsets_U,
    uint32_t* __restrict__ offsets_V,
    size_t word_count,
    bool has_U,
    bool has_V)
{
    const size_t word = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (word >= word_count) return;
    offsets_U[word] = has_U ? __popc(mask_U[word]) : 0;
    offsets_V[word] = has_V ? __popc(mask_V[word]) : 0;
}

__global__ void kernel_restore_zeroeb_values_pair(
    const uint32_t* __restrict__ mask_U,
    const uint32_t* __restrict__ mask_V,
    const uint32_t* __restrict__ offsets_U,
    const uint32_t* __restrict__ offsets_V,
    const float* __restrict__ values_U,
    const float* __restrict__ values_V,
    float* __restrict__ output_U,
    float* __restrict__ output_V,
    size_t n,
    bool has_U,
    bool has_V)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t word = i >> 5;
    const unsigned lane = static_cast<unsigned>(i & 31u);
    const uint32_t lower_bits = lane == 0 ? 0u : ((1u << lane) - 1u);
    if (has_U) {
        const uint32_t mask = mask_U[word];
        if ((mask >> lane) & 1u) {
            output_U[i] = values_U[offsets_U[word] + __popc(mask & lower_bits)];
        }
    }
    if (has_V) {
        const uint32_t mask = mask_V[word];
        if ((mask >> lane) & 1u) {
            output_V[i] = values_V[offsets_V[word] + __popc(mask & lower_bits)];
        }
    }
}

__global__ void kernel_eq_ids_to_error_bounds_pair(
    const uint8_t* __restrict__ eq_U,
    const uint8_t* __restrict__ eq_V,
    float* __restrict__ eb_U,
    float* __restrict__ eb_V,
    size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    eb_U[i] = eq_U[i] == 0
        ? 0.0f : static_cast<float>(1ULL << (2 * eq_U[i])) * kThreshold;
    eb_V[i] = eq_V[i] == 0
        ? 0.0f : static_cast<float>(1ULL << (2 * eq_V[i])) * kThreshold;
}

__global__ void kernel_restore_land(
    const uint8_t* __restrict__ land_bitpack,
    float* __restrict__ output_U,
    float* __restrict__ output_V,
    size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && (land_bitpack[i >> 3] & (1u << (i & 7u)))) {
        output_U[i] = 0.0f;
        output_V[i] = 0.0f;
    }
}

struct GpuDecompressionWorkspace2D {
    CucpszCodec codec = CucpszCodec::Huffman;
    size_t r1 = 0;
    size_t r2 = 0;
    size_t n = 0;
    bool tile_variant = false;
    uint32_t tile_dim = 0;
    uint32_t ot_count_U = 0;
    uint32_t ot_count_V = 0;
    uint32_t zeroeb_count_U = 0;
    uint32_t zeroeb_count_V = 0;
    size_t mask_word_count = 0;
    size_t mask_bytes = 0;

    cudaStream_t stream = nullptr;
    cudaEvent_t e2e_start = nullptr;
    cudaEvent_t e2e_stop = nullptr;

    uint16_t* d_eq_U = nullptr;
    uint16_t* d_eq_V = nullptr;
    uint8_t* d_eq_dEb_U = nullptr;
    uint8_t* d_eq_dEb_V = nullptr;
    uint8_t* d_tile_U = nullptr;
    uint8_t* d_tile_V = nullptr;
    float* d_error_bound_U = nullptr;
    float* d_error_bound_V = nullptr;

    uint32_t* d_ot_idx_U = nullptr;
    uint32_t* d_ot_idx_V = nullptr;
    float* d_ot_val_U = nullptr;
    float* d_ot_val_V = nullptr;
    uint32_t* d_zeroeb_mask_U = nullptr;
    uint32_t* d_zeroeb_mask_V = nullptr;
    float* d_zeroeb_val_U = nullptr;
    float* d_zeroeb_val_V = nullptr;
    uint32_t* d_zeroeb_offsets_U = nullptr;
    uint32_t* d_zeroeb_offsets_V = nullptr;
    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    uint8_t* d_land_bitpack = nullptr;

    float* d_output_U = nullptr;
    float* d_output_V = nullptr;
    GpuHuffmanDecodeWorkspace* huffman = nullptr;
    GpuAnsDecodeWorkspace* ans = nullptr;

    explicit GpuDecompressionWorkspace2D(const HostCompressedPayload2D& payload)
        : codec(payload.codec),
          r1(payload.r1),
          r2(payload.r2),
          n(payload.n),
          tile_variant(payload.tile_variant),
          tile_dim(payload.tile_dim),
          ot_count_U(payload.header.ot_count_U),
          ot_count_V(payload.header.ot_count_V),
          zeroeb_count_U(payload.header.zeroeb_count_U),
          zeroeb_count_V(payload.header.zeroeb_count_V),
          mask_word_count((payload.n + 31) / 32),
          mask_bytes(bitmask_word_bytes(payload.n))
    {
        try {
            if (mask_word_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::overflow_error("CUB scan item count exceeds INT_MAX");
            }
            check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "create decompression stream");
            check_cuda(cudaEventCreate(&e2e_start), "create decode E2E start event");
            check_cuda(cudaEventCreate(&e2e_stop), "create decode E2E stop event");

            check_cuda(cudaMalloc(&d_eq_U, n * sizeof(uint16_t)), "allocate U symbols");
            check_cuda(cudaMalloc(&d_eq_V, n * sizeof(uint16_t)), "allocate V symbols");
            check_cuda(cudaMalloc(&d_output_U, n * sizeof(float)), "allocate recovered U");
            check_cuda(cudaMalloc(&d_output_V, n * sizeof(float)), "allocate recovered V");
            check_cuda(cudaMemsetAsync(d_output_U, 0, n * sizeof(float), stream),
                       "initialize recovered U");
            check_cuda(cudaMemsetAsync(d_output_V, 0, n * sizeof(float), stream),
                       "initialize recovered V");

            check_cuda(cudaMalloc(
                           &d_ot_idx_U,
                           std::max<size_t>(1, ot_count_U) * sizeof(uint32_t)),
                       "allocate U outlier indices");
            check_cuda(cudaMalloc(
                           &d_ot_idx_V,
                           std::max<size_t>(1, ot_count_V) * sizeof(uint32_t)),
                       "allocate V outlier indices");
            check_cuda(cudaMalloc(
                           &d_ot_val_U,
                           std::max<size_t>(1, ot_count_U) * sizeof(float)),
                       "allocate U outlier values");
            check_cuda(cudaMalloc(
                           &d_ot_val_V,
                           std::max<size_t>(1, ot_count_V) * sizeof(float)),
                       "allocate V outlier values");
            if (ot_count_U) {
                check_cuda(cudaMemcpyAsync(
                               d_ot_idx_U, payload.ot_idx_U.data(),
                               ot_count_U * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream),
                           "upload U outlier indices");
                check_cuda(cudaMemcpyAsync(
                               d_ot_val_U, payload.ot_val_U.data(),
                               ot_count_U * sizeof(float),
                               cudaMemcpyHostToDevice, stream),
                           "upload U outlier values");
            }
            if (ot_count_V) {
                check_cuda(cudaMemcpyAsync(
                               d_ot_idx_V, payload.ot_idx_V.data(),
                               ot_count_V * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream),
                           "upload V outlier indices");
                check_cuda(cudaMemcpyAsync(
                               d_ot_val_V, payload.ot_val_V.data(),
                               ot_count_V * sizeof(float),
                               cudaMemcpyHostToDevice, stream),
                           "upload V outlier values");
            }

            const bool has_zeroeb = zeroeb_count_U || zeroeb_count_V;
            if (has_zeroeb) {
                check_cuda(cudaMalloc(
                               &d_zeroeb_offsets_U,
                               mask_word_count * sizeof(uint32_t)),
                           "allocate U zeroEB word offsets");
                check_cuda(cudaMalloc(
                               &d_zeroeb_offsets_V,
                               mask_word_count * sizeof(uint32_t)),
                           "allocate V zeroEB word offsets");
                check_cuda(cub::DeviceScan::ExclusiveSum(
                               nullptr, scan_temp_bytes,
                               d_zeroeb_offsets_U, d_zeroeb_offsets_U,
                               static_cast<int>(mask_word_count), stream),
                           "query zeroEB scan workspace");
                check_cuda(cudaMalloc(&d_scan_temp, std::max<size_t>(1, scan_temp_bytes)),
                           "allocate zeroEB scan workspace");
            }
            if (zeroeb_count_U) {
                check_cuda(cudaMalloc(&d_zeroeb_mask_U, mask_bytes),
                           "allocate U zeroEB mask");
                check_cuda(cudaMalloc(
                               &d_zeroeb_val_U,
                               zeroeb_count_U * sizeof(float)),
                           "allocate U zeroEB values");
                check_cuda(cudaMemcpyAsync(
                               d_zeroeb_mask_U, payload.zeroeb_mask_U.data(), mask_bytes,
                               cudaMemcpyHostToDevice, stream),
                           "upload U zeroEB mask");
                check_cuda(cudaMemcpyAsync(
                               d_zeroeb_val_U, payload.zeroeb_val_U.data(),
                               zeroeb_count_U * sizeof(float),
                               cudaMemcpyHostToDevice, stream),
                           "upload U zeroEB values");
            }
            if (zeroeb_count_V) {
                check_cuda(cudaMalloc(&d_zeroeb_mask_V, mask_bytes),
                           "allocate V zeroEB mask");
                check_cuda(cudaMalloc(
                               &d_zeroeb_val_V,
                               zeroeb_count_V * sizeof(float)),
                           "allocate V zeroEB values");
                check_cuda(cudaMemcpyAsync(
                               d_zeroeb_mask_V, payload.zeroeb_mask_V.data(), mask_bytes,
                               cudaMemcpyHostToDevice, stream),
                           "upload V zeroEB mask");
                check_cuda(cudaMemcpyAsync(
                               d_zeroeb_val_V, payload.zeroeb_val_V.data(),
                               zeroeb_count_V * sizeof(float),
                               cudaMemcpyHostToDevice, stream),
                           "upload V zeroEB values");
            }

            if (!payload.land_bitpack.empty()) {
                check_cuda(cudaMalloc(&d_land_bitpack, payload.land_bitpack.size()),
                           "allocate land bitpack");
                check_cuda(cudaMemcpyAsync(
                               d_land_bitpack, payload.land_bitpack.data(),
                               payload.land_bitpack.size(),
                               cudaMemcpyHostToDevice, stream),
                           "upload land bitpack");
            }

            if (tile_variant) {
                check_cuda(cudaMalloc(&d_tile_U, payload.blobs[2].size()),
                           "allocate U tile error bounds");
                check_cuda(cudaMalloc(&d_tile_V, payload.blobs[3].size()),
                           "allocate V tile error bounds");
                check_cuda(cudaMemcpyAsync(
                               d_tile_U, payload.blobs[2].data(), payload.blobs[2].size(),
                               cudaMemcpyHostToDevice, stream),
                           "upload U tile error bounds");
                check_cuda(cudaMemcpyAsync(
                               d_tile_V, payload.blobs[3].data(), payload.blobs[3].size(),
                               cudaMemcpyHostToDevice, stream),
                           "upload V tile error bounds");
            }
            else {
                check_cuda(cudaMalloc(&d_eq_dEb_U, n * sizeof(uint8_t)),
                           "allocate U EB symbols");
                check_cuda(cudaMalloc(&d_eq_dEb_V, n * sizeof(uint8_t)),
                           "allocate V EB symbols");
                check_cuda(cudaMalloc(&d_error_bound_U, n * sizeof(float)),
                           "allocate U error bounds");
                check_cuda(cudaMalloc(&d_error_bound_V, n * sizeof(float)),
                           "allocate V error bounds");
            }

            const uint8_t* h_blobs[4] = {
                payload.blobs[0].data(), payload.blobs[1].data(),
                payload.blobs[2].data(), payload.blobs[3].data()};
            const size_t blob_lens[4] = {
                payload.blobs[0].size(), payload.blobs[1].size(),
                payload.blobs[2].size(), payload.blobs[3].size()};
            if (codec == CucpszCodec::Ans) {
                ans = create_gpu_ans_decode_workspace(
                    h_blobs, blob_lens, n, !tile_variant, stream);
            }
            else {
                huffman = create_gpu_huffman_decode_workspace(
                    h_blobs, blob_lens, n, !tile_variant, stream);
            }

            // This is the end of preparation: allocations, explicit memsets,
            // compressed/special payload H2D, and codec configuration are done.
            check_cuda(cudaDeviceSynchronize(), "finish decompression preparation");
        }
        catch (...) {
            release();
            throw;
        }
    }

    GpuDecompressionWorkspace2D(const GpuDecompressionWorkspace2D&) = delete;
    GpuDecompressionWorkspace2D& operator=(const GpuDecompressionWorkspace2D&) = delete;

    ~GpuDecompressionWorkspace2D() { release(); }

private:
    void release()
    {
        if (ans) destroy_gpu_ans_decode_workspace(ans);
        ans = nullptr;
        if (huffman) destroy_gpu_huffman_decode_workspace(huffman);
        huffman = nullptr;
        if (e2e_start) cudaEventDestroy(e2e_start);
        if (e2e_stop) cudaEventDestroy(e2e_stop);
        e2e_start = e2e_stop = nullptr;
        if (d_output_V) cudaFree(d_output_V);
        if (d_output_U) cudaFree(d_output_U);
        if (d_land_bitpack) cudaFree(d_land_bitpack);
        if (d_scan_temp) cudaFree(d_scan_temp);
        if (d_zeroeb_offsets_V) cudaFree(d_zeroeb_offsets_V);
        if (d_zeroeb_offsets_U) cudaFree(d_zeroeb_offsets_U);
        if (d_zeroeb_val_V) cudaFree(d_zeroeb_val_V);
        if (d_zeroeb_val_U) cudaFree(d_zeroeb_val_U);
        if (d_zeroeb_mask_V) cudaFree(d_zeroeb_mask_V);
        if (d_zeroeb_mask_U) cudaFree(d_zeroeb_mask_U);
        if (d_ot_val_V) cudaFree(d_ot_val_V);
        if (d_ot_val_U) cudaFree(d_ot_val_U);
        if (d_ot_idx_V) cudaFree(d_ot_idx_V);
        if (d_ot_idx_U) cudaFree(d_ot_idx_U);
        if (d_error_bound_V) cudaFree(d_error_bound_V);
        if (d_error_bound_U) cudaFree(d_error_bound_U);
        if (d_tile_V) cudaFree(d_tile_V);
        if (d_tile_U) cudaFree(d_tile_U);
        if (d_eq_dEb_V) cudaFree(d_eq_dEb_V);
        if (d_eq_dEb_U) cudaFree(d_eq_dEb_U);
        if (d_eq_V) cudaFree(d_eq_V);
        if (d_eq_U) cudaFree(d_eq_U);
        d_output_V = d_output_U = nullptr;
        d_land_bitpack = nullptr;
        d_scan_temp = nullptr;
        d_zeroeb_offsets_V = d_zeroeb_offsets_U = nullptr;
        d_zeroeb_val_V = d_zeroeb_val_U = nullptr;
        d_zeroeb_mask_V = d_zeroeb_mask_U = nullptr;
        d_ot_val_V = d_ot_val_U = nullptr;
        d_ot_idx_V = d_ot_idx_U = nullptr;
        d_error_bound_V = d_error_bound_U = nullptr;
        d_tile_V = d_tile_U = nullptr;
        d_eq_dEb_V = d_eq_dEb_U = nullptr;
        d_eq_V = d_eq_U = nullptr;
        if (stream) cudaStreamDestroy(stream);
        stream = nullptr;
    }
};

// GPU-resident boundary: all compressed segments and metadata are already on
// the device, and every output/scratch allocation is owned by the caller.
// On return, recovered U/V are ready in workspace.d_output_U/V.
void decompress_cucpsz(GpuDecompressionWorkspace2D& workspace)
{
    if (workspace.codec == CucpszCodec::Ans) {
        run_gpu_ans_decode_device(
            workspace.ans,
            workspace.d_eq_U, workspace.d_eq_V,
            workspace.d_eq_dEb_U, workspace.d_eq_dEb_V);
    }
    else {
        run_gpu_huffman_decode_device(
            workspace.huffman,
            workspace.d_eq_U, workspace.d_eq_V,
            workspace.d_eq_dEb_U, workspace.d_eq_dEb_V);
    }

    constexpr int block_size = 256;
    const int data_grid = static_cast<int>((workspace.n + block_size - 1) / block_size);
    if (!workspace.tile_variant) {
        kernel_eq_ids_to_error_bounds_pair<<<data_grid, block_size, 0, workspace.stream>>>(
            workspace.d_eq_dEb_U, workspace.d_eq_dEb_V,
            workspace.d_error_bound_U, workspace.d_error_bound_V,
            workspace.n);
        check_cuda(cudaGetLastError(), "launch error-bound reconstruction");
    }

    if (workspace.ot_count_U) {
        const int grid = (workspace.ot_count_U + block_size - 1) / block_size;
        kernel_scatter_values<<<grid, block_size, 0, workspace.stream>>>(
            workspace.d_ot_idx_U, workspace.d_ot_val_U,
            workspace.ot_count_U, workspace.d_output_U);
    }
    float ignored_lorenzo_ms = 0.0f;
    if (workspace.tile_variant) {
        const auto status =
            psz::cuhip::GPU_PROTO_x_lorenzo_nd__tile_eb<float, uint16_t, uint8_t>(
                workspace.d_eq_U, workspace.d_output_U, workspace.d_output_U,
                dim3(workspace.r2, workspace.r1, 1), workspace.d_tile_U,
                static_cast<int>(workspace.tile_dim), kThreshold, kRadius,
                &ignored_lorenzo_ms, workspace.stream);
        if (status != CUSZ_SUCCESS) {
            throw std::runtime_error("U tile Lorenzo decompression failed");
        }
    }
    else {
        const auto status =
            psz::cuhip::GPU_PROTO_x_lorenzo_row1d__eb_list<float, uint16_t>(
                workspace.d_eq_U, workspace.d_output_U,
                dim3(workspace.r2, workspace.r1, 1), workspace.d_error_bound_U,
                kRadius, &ignored_lorenzo_ms, workspace.stream);
        if (status != CUSZ_SUCCESS) {
            throw std::runtime_error("U row Lorenzo decompression failed");
        }
    }

    if (workspace.ot_count_V) {
        const int grid = (workspace.ot_count_V + block_size - 1) / block_size;
        kernel_scatter_values<<<grid, block_size, 0, workspace.stream>>>(
            workspace.d_ot_idx_V, workspace.d_ot_val_V,
            workspace.ot_count_V, workspace.d_output_V);
    }
    if (workspace.tile_variant) {
        const auto status =
            psz::cuhip::GPU_PROTO_x_lorenzo_nd__tile_eb<float, uint16_t, uint8_t>(
                workspace.d_eq_V, workspace.d_output_V, workspace.d_output_V,
                dim3(workspace.r2, workspace.r1, 1), workspace.d_tile_V,
                static_cast<int>(workspace.tile_dim), kThreshold, kRadius,
                &ignored_lorenzo_ms, workspace.stream);
        if (status != CUSZ_SUCCESS) {
            throw std::runtime_error("V tile Lorenzo decompression failed");
        }
    }
    else {
        const auto status =
            psz::cuhip::GPU_PROTO_x_lorenzo_row1d__eb_list<float, uint16_t>(
                workspace.d_eq_V, workspace.d_output_V,
                dim3(workspace.r2, workspace.r1, 1), workspace.d_error_bound_V,
                kRadius, &ignored_lorenzo_ms, workspace.stream);
        if (status != CUSZ_SUCCESS) {
            throw std::runtime_error("V row Lorenzo decompression failed");
        }
    }

    const bool has_zeroeb_U = workspace.zeroeb_count_U != 0;
    const bool has_zeroeb_V = workspace.zeroeb_count_V != 0;
    if (has_zeroeb_U || has_zeroeb_V) {
        const int word_grid = static_cast<int>(
            (workspace.mask_word_count + block_size - 1) / block_size);
        kernel_mask_word_counts_pair<<<word_grid, block_size, 0, workspace.stream>>>(
            workspace.d_zeroeb_mask_U, workspace.d_zeroeb_mask_V,
            workspace.d_zeroeb_offsets_U, workspace.d_zeroeb_offsets_V,
            workspace.mask_word_count, has_zeroeb_U, has_zeroeb_V);
        check_cuda(cub::DeviceScan::ExclusiveSum(
                       workspace.d_scan_temp, workspace.scan_temp_bytes,
                       workspace.d_zeroeb_offsets_U, workspace.d_zeroeb_offsets_U,
                       static_cast<int>(workspace.mask_word_count), workspace.stream),
                   "scan U zeroEB mask words");
        check_cuda(cub::DeviceScan::ExclusiveSum(
                       workspace.d_scan_temp, workspace.scan_temp_bytes,
                       workspace.d_zeroeb_offsets_V, workspace.d_zeroeb_offsets_V,
                       static_cast<int>(workspace.mask_word_count), workspace.stream),
                   "scan V zeroEB mask words");
        kernel_restore_zeroeb_values_pair<<<data_grid, block_size, 0, workspace.stream>>>(
            workspace.d_zeroeb_mask_U, workspace.d_zeroeb_mask_V,
            workspace.d_zeroeb_offsets_U, workspace.d_zeroeb_offsets_V,
            workspace.d_zeroeb_val_U, workspace.d_zeroeb_val_V,
            workspace.d_output_U, workspace.d_output_V,
            workspace.n, has_zeroeb_U, has_zeroeb_V);
    }

    if (workspace.d_land_bitpack) {
        kernel_restore_land<<<data_grid, block_size, 0, workspace.stream>>>(
            workspace.d_land_bitpack,
            workspace.d_output_U, workspace.d_output_V, workspace.n);
    }
    check_cuda(cudaGetLastError(), "launch final decompression restore kernels");
}

struct RecoveredOutputTimes {
    double host_alloc_seconds = 0.0;
    double d2h_seconds = 0.0;
    double write_seconds = 0.0;
};

void write_field(const char* filename, const std::vector<float>& values)
{
    std::ofstream output(filename, std::ios::binary);
    if (!output) {
        throw std::runtime_error(std::string("Cannot open output file: ") + filename);
    }
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error(std::string("Failed to write output file: ") + filename);
    }
}

// This is deliberately outside decompress_cucpsz and its GPU-resident timer.
// It owns the recovered-data GPU -> CPU -> disk path.
RecoveredOutputTimes download_and_write_recovered_fields(
    const GpuDecompressionWorkspace2D& workspace,
    const char* output_U,
    const char* output_V)
{
    using Clock = std::chrono::steady_clock;
    const auto alloc_start = Clock::now();
    std::vector<float> host_U(workspace.n);
    std::vector<float> host_V(workspace.n);
    const auto alloc_stop = Clock::now();

    const auto d2h_start = alloc_stop;
    check_cuda(cudaMemcpy(
                   host_U.data(), workspace.d_output_U,
                   workspace.n * sizeof(float), cudaMemcpyDeviceToHost),
               "download recovered U");
    check_cuda(cudaMemcpy(
                   host_V.data(), workspace.d_output_V,
                   workspace.n * sizeof(float), cudaMemcpyDeviceToHost),
               "download recovered V");
    const auto d2h_stop = Clock::now();

    write_field(output_U, host_U);
    write_field(output_V, host_V);
    const auto write_stop = Clock::now();

    RecoveredOutputTimes times;
    times.host_alloc_seconds =
        std::chrono::duration<double>(alloc_stop - alloc_start).count();
    times.d2h_seconds =
        std::chrono::duration<double>(d2h_stop - d2h_start).count();
    times.write_seconds =
        std::chrono::duration<double>(write_stop - d2h_stop).count();
    return times;
}

// Runs one full untimed decompression on a tiny synthetic payload before the
// real, timed decompression. This absorbs CUDA context init, kernel/module
// JIT, and GPU clock ramp-up so GPU_RESIDENT_DECOMPRESSION_E2E reflects
// steady-state throughput instead of first-launch cold start. The fixture is
// built in-process (encode a tiny quant-code array to get a valid codec
// blob) so no fixture file is needed.
double warmup_offline_decompress(CucpszCodec codec)
{
    using WarmClock = std::chrono::steady_clock;
    const auto warm_start = WarmClock::now();

    constexpr size_t warm_r1 = 256, warm_r2 = 256;
    constexpr size_t warm_n = warm_r1 * warm_r2;
    constexpr uint32_t warm_tile_dim = 32;
    const size_t tile_cols = (warm_r2 + warm_tile_dim - 1) / warm_tile_dim;
    const size_t tile_rows = (warm_r1 + warm_tile_dim - 1) / warm_tile_dim;
    const size_t tile_count = tile_cols * tile_rows;

    std::vector<uint16_t> h_eq(warm_n);
    for (size_t i = 0; i < warm_n; ++i) {
        h_eq[i] = static_cast<uint16_t>(50 + (i * 37 + i / 13) % 400);
    }
    uint16_t* d_eq_U = nullptr;
    uint16_t* d_eq_V = nullptr;
    cudaMalloc(&d_eq_U, warm_n * sizeof(uint16_t));
    cudaMalloc(&d_eq_V, warm_n * sizeof(uint16_t));
    cudaMemcpy(d_eq_U, h_eq.data(), warm_n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eq_V, h_eq.data(), warm_n * sizeof(uint16_t), cudaMemcpyHostToDevice);

    HostCompressedPayload2D payload;
    payload.codec = codec;
    payload.r1 = warm_r1;
    payload.r2 = warm_r2;
    payload.n = warm_n;
    payload.tile_variant = true;
    payload.tile_dim = warm_tile_dim;
    payload.header.r1 = warm_r1;
    payload.header.r2 = warm_r2;
    payload.header.max_pwr_eb = 0.1f;
    payload.header.codec = static_cast<uint32_t>(codec);
    payload.header.lorenzo_variant = 1;
    payload.header.lorenzo_tile_dim = warm_tile_dim;

    if (codec == CucpszCodec::Huffman) {
        GpuHuffmanU2Workspace* enc = create_gpu_huffman_u2_workspace(warm_n);
        size_t out_lens[2] = {0, 0};
        uint8_t* out_d_blobs[2] = {nullptr, nullptr};
        float encode_ms[2] = {0.0f, 0.0f}, wall_ms = 0.0f, kernel_ms = 0.0f;
        run_gpu_huffman_u2_arrays_device(
            enc, d_eq_U, d_eq_V, warm_n, nullptr,
            out_lens, out_d_blobs, encode_ms, &wall_ms, &kernel_ms);
        for (int i = 0; i < 2; ++i) {
            payload.blobs[i].resize(out_lens[i]);
            cudaMemcpy(payload.blobs[i].data(), out_d_blobs[i], out_lens[i],
                       cudaMemcpyDeviceToHost);
        }
        destroy_gpu_huffman_u2_workspace(enc);
    }
    else {
        GpuAnsU2Workspace* enc = create_gpu_ans_u2_workspace(warm_n, nullptr);
        float kernel_ms = 0.0f;
        run_gpu_ans_u2_arrays_device(enc, d_eq_U, d_eq_V, warm_n, &kernel_ms);
        uint8_t* out_h_blobs[2] = {nullptr, nullptr};
        size_t out_lens[2] = {0, 0};
        download_gpu_ans_u2_arrays(enc, out_h_blobs, out_lens);
        for (int i = 0; i < 2; ++i) {
            payload.blobs[i].assign(out_h_blobs[i], out_h_blobs[i] + out_lens[i]);
            delete[] out_h_blobs[i];
        }
        destroy_gpu_ans_u2_workspace(enc);
    }
    cudaFree(d_eq_V);
    cudaFree(d_eq_U);

    payload.blobs[2].assign(tile_count, uint8_t{3});
    payload.blobs[3].assign(tile_count, uint8_t{3});
    for (int i = 0; i < 4; ++i) {
        payload.header.hf_blob_len[i] = payload.blobs[i].size();
    }

    {
        GpuDecompressionWorkspace2D workspace(payload);
        decompress_cucpsz(workspace);
        check_cuda(cudaStreamSynchronize(workspace.stream), "finish warmup decompression");
    }
    return std::chrono::duration<double>(WarmClock::now() - warm_start).count();
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s input.cucpsz U.cucpsz.out V.cucpsz.out [hf|ans]\n", argv[0]);
        return 1;
    }

    CucpszCodec codec = CucpszCodec::Huffman;
    if (argc == 5 && !parse_cucpsz_codec(argv[4], codec)) {
        fprintf(stderr, "Invalid codec '%s'; expected 'hf' or 'ans'\n", argv[4]);
        return 1;
    }

    using Clock = std::chrono::steady_clock;
    try {
        printf("=== GPU warmup (untimed, synthetic 256x256 payload) ===\n");
        const double warmup_seconds = warmup_offline_decompress(codec);
        printf("=== warmup done: %.6f s ===\n\n", warmup_seconds);

        const auto read_start = Clock::now();
        HostCompressedPayload2D payload = read_cucpsz_file(argv[1], codec);
        const auto read_stop = Clock::now();
        printf("Decompress: codec=%s r1=%zu r2=%zu  ot_U=%u zeb_U=%u  ot_V=%u zeb_V=%u\n",
               cucpsz_codec_name(payload.codec), payload.r1, payload.r2,
               payload.header.ot_count_U, payload.header.zeroeb_count_U,
               payload.header.ot_count_V, payload.header.zeroeb_count_V);
        if (payload.tile_variant) {
            printf("  tile EB payload: tile=%u  U=%zu bytes  V=%zu bytes\n",
                   payload.tile_dim, payload.blobs[2].size(), payload.blobs[3].size());
        }

        const auto prepare_start = Clock::now();
        GpuDecompressionWorkspace2D workspace(payload);
        const auto prepare_stop = Clock::now();

        check_cuda(cudaEventRecord(workspace.e2e_start, workspace.stream),
                   "record decompression E2E start");
        const auto resident_start = Clock::now();
        decompress_cucpsz(workspace);
        check_cuda(cudaEventRecord(workspace.e2e_stop, workspace.stream),
                   "record decompression E2E stop");
        check_cuda(cudaEventSynchronize(workspace.e2e_stop),
                   "finish GPU-resident decompression");
        const auto resident_stop = Clock::now();

        float resident_event_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(
                       &resident_event_ms, workspace.e2e_start, workspace.e2e_stop),
                   "measure GPU-resident decompression");
        if (codec == CucpszCodec::Ans) {
            check_gpu_ans_decode_status(workspace.ans);
        }

        const double resident_wall_ms =
            std::chrono::duration<double, std::milli>(
                resident_stop - resident_start).count();
        const double output_gib =
            payload.n * 2.0 * sizeof(float) /
            (1024.0 * 1024.0 * 1024.0);
        printf("\nGPU_RESIDENT_DECOMPRESSION_E2E (warmed up with RANDOM dataset)\n");
        printf("  boundary   : compressed device segments ready -> recovered U/V ready on GPU\n");
        printf("  wall       : %8.3f ms  (%.2f GiB/s)\n",
               resident_wall_ms, output_gib / (resident_wall_ms / 1000.0));
        printf("  CUDA event : %8.3f ms  (same device-ready boundary)\n\n",
               resident_event_ms);

        const RecoveredOutputTimes output_times =
            download_and_write_recovered_fields(workspace, argv[2], argv[3]);
        const double read_seconds =
            std::chrono::duration<double>(read_stop - read_start).count();
        const double prepare_seconds =
            std::chrono::duration<double>(prepare_stop - prepare_start).count();

        printf("DECOMPRESSION_PREPARE_TIME (excluded from GPU-resident E2E)\n");
        printf("  file read                 : %.6f s\n", read_seconds);
        printf("  alloc + H2D + init/config : %.6f s\n\n", prepare_seconds);
        printf("RECOVERED_OUTPUT_TIME (excluded from GPU-resident E2E)\n");
        printf("  host buffer alloc : %.6f s\n", output_times.host_alloc_seconds);
        printf("  recovered D2H : %.6f s\n", output_times.d2h_seconds);
        printf("  file write    : %.6f s\n", output_times.write_seconds);
        printf("  total         : %.6f s\n",
               output_times.host_alloc_seconds + output_times.d2h_seconds +
               output_times.write_seconds);
        printf("Written %s\nWritten %s\n", argv[2], argv[3]);
        return 0;
    }
    catch (const std::exception& error) {
        fprintf(stderr, "Decompression failed: %s\n", error.what());
        return 1;
    }
}
