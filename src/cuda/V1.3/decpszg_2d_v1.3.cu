#include "cucpsz_2d_format.hpp"
#include "lorenzo_tile_dim.h"
#include "kernel/lrz/lproto.hh"

#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kRadius = 4096;

extern "C" void hf_u1_decode_from_blob(
    uint8_t* h_blob, size_t blob_len, size_t n,
    uint8_t* d_decoded, void* stream);

extern "C" void hf_u2_decode_from_blob(
    uint8_t* h_blob, size_t blob_len, size_t n,
    uint16_t* d_decoded, void* stream);

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

__global__ void kernel_mask_to_flags(
    const uint32_t* __restrict__ mask,
    uint32_t* __restrict__ flags,
    size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) flags[i] = (mask[i >> 5] >> (i & 31u)) & 1u;
}

template <typename T>
__global__ void kernel_restore_zeroeb_values(
    const uint32_t* __restrict__ mask,
    const uint32_t* __restrict__ offsets,
    const T* __restrict__ zeroeb_values,
    T* __restrict__ data,
    size_t n)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && ((mask[i >> 5] >> (i & 31u)) & 1u)) {
        data[i] = zeroeb_values[offsets[i]];
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

void decompress_cucpsz(
    const char* filename,
    std::vector<float>& u_decompressed,
    std::vector<float>& v_decompressed,
    CucpszHeader& header)
{
    FILE* file = fopen(filename, "rb");
    if (!file) throw std::runtime_error(std::string("Cannot open ") + filename);

    try {
        if (fread(&header, sizeof(header), 1, file) != 1) {
            throw std::runtime_error("Cannot read .cucpsz header");
        }
        if (!cucpsz_header_has_valid_magic(header)) {
            throw std::runtime_error("Invalid .cucpsz magic");
        }
        if (header.r1 == 0 || header.r2 == 0 ||
            header.r1 > std::numeric_limits<size_t>::max() / header.r2) {
            throw std::runtime_error("Invalid dimensions in .cucpsz header");
        }
        if (header.lorenzo_variant > 1u) {
            throw std::runtime_error("Unsupported Lorenzo variant in .cucpsz header");
        }

        const size_t r1 = static_cast<size_t>(header.r1);
        const size_t r2 = static_cast<size_t>(header.r2);
        const size_t n = r1 * r2;
        if (header.ot_count_U > n || header.ot_count_V > n ||
            header.zeroeb_count_U > n || header.zeroeb_count_V > n) {
            throw std::runtime_error("Invalid special-value count in .cucpsz header");
        }

        std::vector<uint8_t> blobs[4];
        for (int i = 0; i < 4; i++) {
            if (header.hf_blob_len[i] > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("Payload is too large for this platform");
            }
            read_items(file, blobs[i], static_cast<size_t>(header.hf_blob_len[i]), "codec blob");
        }

        const size_t expected_land_bytes = (n + 7) / 8;
        if (header.land_bitpack_bytes != 0 &&
            header.land_bitpack_bytes != expected_land_bytes) {
            throw std::runtime_error("Invalid land bitpack size");
        }
        std::vector<uint8_t> land_bitpack;
        read_items(file, land_bitpack, static_cast<size_t>(header.land_bitpack_bytes), "land bitpack");

        const size_t mask_bytes_U = header.zeroeb_count_U ? bitmask_word_bytes(n) : 0;
        const size_t mask_bytes_V = header.zeroeb_count_V ? bitmask_word_bytes(n) : 0;
        std::vector<uint32_t> ot_idx_U, ot_idx_V;
        std::vector<float> ot_val_U, ot_val_V, zeroeb_val_U, zeroeb_val_V;
        std::vector<uint8_t> zeroeb_mask_U, zeroeb_mask_V;
        read_items(file, ot_idx_U, header.ot_count_U, "U outlier indices");
        read_items(file, ot_val_U, header.ot_count_U, "U outlier values");
        read_items(file, zeroeb_mask_U, mask_bytes_U, "U zero-EB mask");
        read_items(file, zeroeb_val_U, header.zeroeb_count_U, "U zero-EB values");
        read_items(file, ot_idx_V, header.ot_count_V, "V outlier indices");
        read_items(file, ot_val_V, header.ot_count_V, "V outlier values");
        read_items(file, zeroeb_mask_V, mask_bytes_V, "V zero-EB mask");
        read_items(file, zeroeb_val_V, header.zeroeb_count_V, "V zero-EB values");
        fclose(file);
        file = nullptr;

        validate_indices(ot_idx_U, n, "U");
        validate_indices(ot_idx_V, n, "V");

        const bool tile_variant = header.lorenzo_variant == 1u;
        const uint32_t tile_dim = header.lorenzo_tile_dim
            ? header.lorenzo_tile_dim : static_cast<uint32_t>(LORENZO_TILE_DIM);
        if (tile_variant && tile_dim == 0) {
            throw std::runtime_error("Invalid tile dimension in .cucpsz header");
        }

        printf("Decompress: r1=%zu r2=%zu  ot_U=%u zeb_U=%u  ot_V=%u zeb_V=%u\n",
            r1, r2, header.ot_count_U, header.zeroeb_count_U,
            header.ot_count_V, header.zeroeb_count_V);

        using Eq = uint16_t;
        using EqEb = uint8_t;
        constexpr float threshold = 1.0f / (1 << 20);

        cudaStream_t stream = nullptr;
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

        Eq *d_eq_U = nullptr, *d_eq_V = nullptr;
        EqEb *d_eq_dEb_U = nullptr, *d_eq_dEb_V = nullptr;
        check_cuda(cudaMalloc(&d_eq_U, n * sizeof(Eq)), "cudaMalloc d_eq_U");
        check_cuda(cudaMalloc(&d_eq_V, n * sizeof(Eq)), "cudaMalloc d_eq_V");
        hf_u2_decode_from_blob(blobs[0].data(), blobs[0].size(), n, d_eq_U, stream);
        hf_u2_decode_from_blob(blobs[1].data(), blobs[1].size(), n, d_eq_V, stream);
        if (!tile_variant) {
            check_cuda(cudaMalloc(&d_eq_dEb_U, n * sizeof(EqEb)), "cudaMalloc d_eq_dEb_U");
            check_cuda(cudaMalloc(&d_eq_dEb_V, n * sizeof(EqEb)), "cudaMalloc d_eq_dEb_V");
            hf_u1_decode_from_blob(blobs[2].data(), blobs[2].size(), n, d_eq_dEb_U, stream);
            hf_u1_decode_from_blob(blobs[3].data(), blobs[3].size(), n, d_eq_dEb_V, stream);
        }

        uint32_t *d_ot_idx_U = nullptr, *d_ot_idx_V = nullptr;
        uint32_t *d_zeroeb_mask_U = nullptr, *d_zeroeb_mask_V = nullptr;
        uint32_t* d_zeroeb_offsets = nullptr;
        float *d_ot_val_U = nullptr, *d_ot_val_V = nullptr;
        float *d_zeroeb_val_U = nullptr, *d_zeroeb_val_V = nullptr;
        check_cuda(cudaMalloc(&d_ot_idx_U, std::max<size_t>(1, ot_idx_U.size()) * sizeof(uint32_t)), "cudaMalloc d_ot_idx_U");
        check_cuda(cudaMalloc(&d_ot_idx_V, std::max<size_t>(1, ot_idx_V.size()) * sizeof(uint32_t)), "cudaMalloc d_ot_idx_V");
        check_cuda(cudaMalloc(&d_ot_val_U, std::max<size_t>(1, ot_val_U.size()) * sizeof(float)), "cudaMalloc d_ot_val_U");
        check_cuda(cudaMalloc(&d_ot_val_V, std::max<size_t>(1, ot_val_V.size()) * sizeof(float)), "cudaMalloc d_ot_val_V");
        check_cuda(cudaMalloc(&d_zeroeb_val_U, std::max<size_t>(1, zeroeb_val_U.size()) * sizeof(float)), "cudaMalloc d_zeroeb_val_U");
        check_cuda(cudaMalloc(&d_zeroeb_val_V, std::max<size_t>(1, zeroeb_val_V.size()) * sizeof(float)), "cudaMalloc d_zeroeb_val_V");
        if (!ot_idx_U.empty()) {
            check_cuda(cudaMemcpy(d_ot_idx_U, ot_idx_U.data(), ot_idx_U.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy U outlier indices");
            check_cuda(cudaMemcpy(d_ot_val_U, ot_val_U.data(), ot_val_U.size() * sizeof(float), cudaMemcpyHostToDevice), "copy U outlier values");
        }
        if (!ot_idx_V.empty()) {
            check_cuda(cudaMemcpy(d_ot_idx_V, ot_idx_V.data(), ot_idx_V.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy V outlier indices");
            check_cuda(cudaMemcpy(d_ot_val_V, ot_val_V.data(), ot_val_V.size() * sizeof(float), cudaMemcpyHostToDevice), "copy V outlier values");
        }
        if (!zeroeb_val_U.empty()) {
            check_cuda(cudaMalloc(&d_zeroeb_mask_U, mask_bytes_U), "cudaMalloc d_zeroeb_mask_U");
            check_cuda(cudaMemcpy(d_zeroeb_mask_U, zeroeb_mask_U.data(), mask_bytes_U, cudaMemcpyHostToDevice), "copy U zero-EB mask");
            check_cuda(cudaMemcpy(d_zeroeb_val_U, zeroeb_val_U.data(), zeroeb_val_U.size() * sizeof(float), cudaMemcpyHostToDevice), "copy U zero-EB values");
        }
        if (!zeroeb_val_V.empty()) {
            check_cuda(cudaMalloc(&d_zeroeb_mask_V, mask_bytes_V), "cudaMalloc d_zeroeb_mask_V");
            check_cuda(cudaMemcpy(d_zeroeb_mask_V, zeroeb_mask_V.data(), mask_bytes_V, cudaMemcpyHostToDevice), "copy V zero-EB mask");
            check_cuda(cudaMemcpy(d_zeroeb_val_V, zeroeb_val_V.data(), zeroeb_val_V.size() * sizeof(float), cudaMemcpyHostToDevice), "copy V zero-EB values");
        }
        if (!zeroeb_val_U.empty() || !zeroeb_val_V.empty()) {
            check_cuda(cudaMalloc(&d_zeroeb_offsets, n * sizeof(uint32_t)), "cudaMalloc d_zeroeb_offsets");
        }

        float *d_error_bound_U = nullptr, *d_error_bound_V = nullptr;
        EqEb *d_tile_U = nullptr, *d_tile_V = nullptr;
        if (tile_variant) {
            const size_t tile_cols = (r2 + tile_dim - 1) / tile_dim;
            const size_t tile_rows = (r1 + tile_dim - 1) / tile_dim;
            if (tile_rows > std::numeric_limits<size_t>::max() / tile_cols) {
                throw std::runtime_error("Tile count overflow");
            }
            const size_t tile_bytes = tile_rows * tile_cols * sizeof(EqEb);
            if (blobs[2].size() != tile_bytes || blobs[3].size() != tile_bytes) {
                throw std::runtime_error("Invalid tile error-bound payload size");
            }
            check_cuda(cudaMalloc(&d_tile_U, tile_bytes), "cudaMalloc d_tile_U");
            check_cuda(cudaMalloc(&d_tile_V, tile_bytes), "cudaMalloc d_tile_V");
            check_cuda(cudaMemcpy(d_tile_U, blobs[2].data(), tile_bytes, cudaMemcpyHostToDevice), "copy U tile error bounds");
            check_cuda(cudaMemcpy(d_tile_V, blobs[3].data(), tile_bytes, cudaMemcpyHostToDevice), "copy V tile error bounds");
            printf("  tile EB payload: tile=%u  U=%zu bytes  V=%zu bytes\n",
                tile_dim, blobs[2].size(), blobs[3].size());
        }
        else {
            check_cuda(cudaMalloc(&d_error_bound_U, n * sizeof(float)), "cudaMalloc d_error_bound_U");
            check_cuda(cudaMalloc(&d_error_bound_V, n * sizeof(float)), "cudaMalloc d_error_bound_V");
            thrust::counting_iterator<size_t> first(0);
            thrust::for_each(thrust::device, first, first + n, [=] __device__ (size_t i) {
                d_error_bound_U[i] = d_eq_dEb_U[i] == 0
                    ? 0.0f : static_cast<float>(1ULL << (2 * d_eq_dEb_U[i])) * threshold;
                d_error_bound_V[i] = d_eq_dEb_V[i] == 0
                    ? 0.0f : static_cast<float>(1ULL << (2 * d_eq_dEb_V[i])) * threshold;
            });
        }

        float* d_u_decompressed = nullptr;
        float* d_v_decompressed = nullptr;
        check_cuda(cudaMalloc(&d_u_decompressed, n * sizeof(float)), "cudaMalloc d_u_decompressed");
        check_cuda(cudaMalloc(&d_v_decompressed, n * sizeof(float)), "cudaMalloc d_v_decompressed");
        check_cuda(cudaMemset(d_u_decompressed, 0, n * sizeof(float)), "clear U output");
        check_cuda(cudaMemset(d_v_decompressed, 0, n * sizeof(float)), "clear V output");

        float lorenzo_time = 0.0f;
        if (!ot_idx_U.empty()) {
            thrust::scatter(thrust::device, d_ot_val_U, d_ot_val_U + ot_idx_U.size(), d_ot_idx_U, d_u_decompressed);
        }
        if (tile_variant) {
            psz::cuhip::GPU_PROTO_x_lorenzo_nd__tile_eb<float, Eq, EqEb>(
                d_eq_U, d_u_decompressed, d_u_decompressed, dim3(r2, r1, 1),
                d_tile_U, static_cast<int>(tile_dim), threshold, kRadius, &lorenzo_time, 0);
        }
        else {
            psz::cuhip::GPU_PROTO_x_lorenzo_row1d__eb_list<float, Eq>(
                d_eq_U, d_u_decompressed, dim3(r2, r1, 1),
                d_error_bound_U, kRadius, &lorenzo_time, 0);
        }
        check_cuda(cudaDeviceSynchronize(), "decompress U");

        constexpr int restore_block_size = 512;
        const size_t restore_grid_size = (n + restore_block_size - 1) / restore_block_size;
        if (restore_grid_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Dataset is too large for zero-EB restore grid");
        }
        if (!zeroeb_val_U.empty()) {
            kernel_mask_to_flags<<<static_cast<int>(restore_grid_size), restore_block_size>>>(
                d_zeroeb_mask_U, d_zeroeb_offsets, n);
            thrust::exclusive_scan(thrust::device, d_zeroeb_offsets, d_zeroeb_offsets + n, d_zeroeb_offsets);
            kernel_restore_zeroeb_values<<<static_cast<int>(restore_grid_size), restore_block_size>>>(
                d_zeroeb_mask_U, d_zeroeb_offsets, d_zeroeb_val_U, d_u_decompressed, n);
        }

        if (!ot_idx_V.empty()) {
            thrust::scatter(thrust::device, d_ot_val_V, d_ot_val_V + ot_idx_V.size(), d_ot_idx_V, d_v_decompressed);
        }
        if (tile_variant) {
            psz::cuhip::GPU_PROTO_x_lorenzo_nd__tile_eb<float, Eq, EqEb>(
                d_eq_V, d_v_decompressed, d_v_decompressed, dim3(r2, r1, 1),
                d_tile_V, static_cast<int>(tile_dim), threshold, kRadius, &lorenzo_time, 0);
        }
        else {
            psz::cuhip::GPU_PROTO_x_lorenzo_row1d__eb_list<float, Eq>(
                d_eq_V, d_v_decompressed, dim3(r2, r1, 1),
                d_error_bound_V, kRadius, &lorenzo_time, 0);
        }
        check_cuda(cudaDeviceSynchronize(), "decompress V");
        if (!zeroeb_val_V.empty()) {
            kernel_mask_to_flags<<<static_cast<int>(restore_grid_size), restore_block_size>>>(
                d_zeroeb_mask_V, d_zeroeb_offsets, n);
            thrust::exclusive_scan(thrust::device, d_zeroeb_offsets, d_zeroeb_offsets + n, d_zeroeb_offsets);
            kernel_restore_zeroeb_values<<<static_cast<int>(restore_grid_size), restore_block_size>>>(
                d_zeroeb_mask_V, d_zeroeb_offsets, d_zeroeb_val_V, d_v_decompressed, n);
        }

        if (!land_bitpack.empty()) {
            uint8_t* d_land_bitpack = nullptr;
            check_cuda(cudaMalloc(&d_land_bitpack, land_bitpack.size()), "cudaMalloc d_land_bitpack");
            check_cuda(cudaMemcpy(d_land_bitpack, land_bitpack.data(), land_bitpack.size(), cudaMemcpyHostToDevice), "copy land bitpack");
            thrust::counting_iterator<size_t> first(0);
            thrust::for_each(thrust::device, first, first + n, [=] __device__ (size_t i) {
                if (d_land_bitpack[i / 8] & (1u << (i % 8))) {
                    d_u_decompressed[i] = 0.0f;
                    d_v_decompressed[i] = 0.0f;
                }
            });
            check_cuda(cudaFree(d_land_bitpack), "cudaFree d_land_bitpack");
        }

        u_decompressed.resize(n);
        v_decompressed.resize(n);
        check_cuda(cudaMemcpy(u_decompressed.data(), d_u_decompressed, n * sizeof(float), cudaMemcpyDeviceToHost), "copy U output");
        check_cuda(cudaMemcpy(v_decompressed.data(), d_v_decompressed, n * sizeof(float), cudaMemcpyDeviceToHost), "copy V output");

        cudaFree(d_eq_U); cudaFree(d_eq_V);
        if (d_eq_dEb_U) cudaFree(d_eq_dEb_U);
        if (d_eq_dEb_V) cudaFree(d_eq_dEb_V);
        cudaFree(d_ot_idx_U); cudaFree(d_ot_idx_V);
        cudaFree(d_ot_val_U); cudaFree(d_ot_val_V);
        cudaFree(d_zeroeb_val_U); cudaFree(d_zeroeb_val_V);
        if (d_zeroeb_mask_U) cudaFree(d_zeroeb_mask_U);
        if (d_zeroeb_mask_V) cudaFree(d_zeroeb_mask_V);
        if (d_zeroeb_offsets) cudaFree(d_zeroeb_offsets);
        if (d_error_bound_U) cudaFree(d_error_bound_U);
        if (d_error_bound_V) cudaFree(d_error_bound_V);
        if (d_tile_U) cudaFree(d_tile_U);
        if (d_tile_V) cudaFree(d_tile_V);
        cudaFree(d_u_decompressed); cudaFree(d_v_decompressed);
        cudaStreamDestroy(stream);
    }
    catch (...) {
        if (file) fclose(file);
        throw;
    }
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s input.cucpsz U.cucpsz.out V.cucpsz.out\n", argv[0]);
        return 1;
    }

    try {
        std::vector<float> u_decompressed, v_decompressed;
        CucpszHeader header{};
        const auto start = std::chrono::steady_clock::now();
        decompress_cucpsz(argv[1], u_decompressed, v_decompressed, header);
        const auto decompressed = std::chrono::steady_clock::now();
        write_field(argv[2], u_decompressed);
        write_field(argv[3], v_decompressed);
        const auto written = std::chrono::steady_clock::now();

        const double decompress_seconds =
            std::chrono::duration<double>(decompressed - start).count();
        const double write_seconds =
            std::chrono::duration<double>(written - decompressed).count();
        printf("DECOMPRESS_TIME total: %.6f s (file read + GPU decode + D2H)\n", decompress_seconds);
        printf("WRITE_TIME total: %.6f s\n", write_seconds);
        printf("Written %s\nWritten %s\n", argv[2], argv[3]);
        return 0;
    }
    catch (const std::exception& error) {
        fprintf(stderr, "Decompression failed: %s\n", error.what());
        return 1;
    }
}
