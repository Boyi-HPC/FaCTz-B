/*
 * GPU Huffman integration for cpSZ-GPU.
 *
 * This bridge uses the cuSZ/pSZ sources vendored under external/cusz.
 * Upstream: https://github.com/szcompressor/cuSZ
 * Revision: d22d9c44b791da798ee8c4e8d68f3a1fb17613f1
 * License and copyright notices: external/cusz/LICENSE
 */

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <cuda_runtime.h>
#include "hf_hl.hh"
#include "gpu_huffman_bridge.hpp"
#include "kernel/hist.hh"

namespace {

void hf_check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// hf_buf is pre-allocated by the caller and reused across arrays.
// Each call: build_book overwrites the codebook, encode overwrites the
// bitstream, then h_blob is copied out before the next call touches hf_buf.
void hf_encode_decode_u1(
    const char* name, uint8_t* d_symbols, size_t n, cudaStream_t stream,
    phf::Buf<uint8_t>& hf_buf,
    size_t* out_encoded_len, float* out_encode_ms, float* out_decode_ms,
    uint8_t** out_h_blob)
{
    constexpr uint16_t bklen = 256;

    uint32_t* d_hist = nullptr;
    uint32_t* h_hist = nullptr;
    cudaMalloc(&d_hist, bklen * sizeof(uint32_t));
    cudaMallocHost(&h_hist, bklen * sizeof(uint32_t));
    cudaMemsetAsync(d_hist, 0, bklen * sizeof(uint32_t), stream);

    int grid_dim, block_dim, shmem_use, r_per_block;
    psz::module::GPU_histogram_generic<uint8_t>::init(n, bklen, grid_dim, block_dim, shmem_use, r_per_block);

    auto t0 = std::chrono::high_resolution_clock::now();
    cudaEvent_t hist_start, hist_stop;
    cudaEventCreate(&hist_start);
    cudaEventCreate(&hist_stop);
    cudaEventRecord(hist_start, stream);
    psz::module::GPU_histogram_generic<uint8_t>::kernel(
        d_symbols, n, d_hist, bklen, grid_dim, block_dim, shmem_use, r_per_block, stream);
    cudaEventRecord(hist_stop, stream);
    cudaMemcpyAsync(h_hist, d_hist, bklen * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    float hist_kernel_ms = 0.0f;
    cudaEventElapsedTime(&hist_kernel_ms, hist_start, hist_stop);
    cudaEventDestroy(hist_start);
    cudaEventDestroy(hist_stop);
    auto t_hist = std::chrono::high_resolution_clock::now();

    // buf is pre-allocated — no construction cost here
    // PHF library bug: single-symbol tree assigns bitcount=0 → empty bitstream.
    if (h_hist[0] == 0) h_hist[0] = 1;
    phf::high_level<uint8_t>::build_book(&hf_buf, h_hist, bklen, stream);
    auto t_book = std::chrono::high_resolution_clock::now();

    uint8_t* encoded = nullptr;
    size_t encoded_len = 0;
    phf_header hf_header{};
    float encode_kernel_ms = 0.0f;
    phf::high_level<uint8_t>::encode_timed(
        &hf_buf, d_symbols, n, &encoded, &encoded_len, hf_header, stream, &encode_kernel_ms);
    cudaStreamSynchronize(stream);

    auto t1 = std::chrono::high_resolution_clock::now();
    *out_encode_ms = hist_kernel_ms + encode_kernel_ms;
    *out_encoded_len = encoded_len;
    auto ms = [](auto a, auto b){ return std::chrono::duration<float, std::milli>(b-a).count(); };
    printf("%s  n=%zu  hist=%.2f ms  book=%.2f ms  encode=%.2f ms  total=%.2f ms  kernels=%.2f ms\n",
           name, n, ms(t0,t_hist), ms(t_hist,t_book), ms(t_book,t1), ms(t0,t1),
           *out_encode_ms);

    // Copy blob to host before hf_buf is reused for the next array
    if (out_h_blob) {
        uint8_t* h_blob = new uint8_t[encoded_len];
        cudaMemcpy(h_blob, encoded, encoded_len, cudaMemcpyDeviceToHost);
        *out_h_blob = h_blob;  // caller owns, must delete[]
    }

    // Decode (verification) removed for speed — decode only happens in hf_u1_decode_from_blob.
    *out_decode_ms = 0.0f;

    cudaFreeHost(h_hist);
    cudaFree(d_hist);
}


void hf_encode_decode_u2(
    const char* name, uint16_t* d_symbols, size_t n, cudaStream_t stream,
    phf::Buf<uint16_t>& hf_buf,
    size_t* out_encoded_len, float* out_encode_ms, float* out_decode_ms,
    uint8_t** out_h_blob)
{
    constexpr uint16_t bklen = 8192;   // 2 * RADIUS (4096); Phase1 shmem = 4*8192 = 32KB < 48KB V100 default

    uint32_t* d_hist = nullptr;
    uint32_t* h_hist = nullptr;
    cudaMalloc(&d_hist, bklen * sizeof(uint32_t));
    cudaMallocHost(&h_hist, bklen * sizeof(uint32_t));
    cudaMemsetAsync(d_hist, 0, bklen * sizeof(uint32_t), stream);

    int grid_dim, block_dim, shmem_use, r_per_block;
    psz::module::GPU_histogram_generic<uint16_t>::init(n, bklen, grid_dim, block_dim, shmem_use, r_per_block);

    auto t0 = std::chrono::high_resolution_clock::now();
    cudaEvent_t hist_start, hist_stop;
    cudaEventCreate(&hist_start);
    cudaEventCreate(&hist_stop);
    cudaEventRecord(hist_start, stream);
    psz::module::GPU_histogram_generic<uint16_t>::kernel(
        d_symbols, n, d_hist, bklen, grid_dim, block_dim, shmem_use, r_per_block, stream);
    cudaEventRecord(hist_stop, stream);
    cudaMemcpyAsync(h_hist, d_hist, bklen * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    float hist_kernel_ms = 0.0f;
    cudaEventElapsedTime(&hist_kernel_ms, hist_start, hist_stop);
    cudaEventDestroy(hist_start);
    cudaEventDestroy(hist_stop);
    auto t_hist = std::chrono::high_resolution_clock::now();

    // PHF library bug: single-symbol Huffman tree assigns bitcount=0, producing empty bitstream.
    // Ensure at least 2 distinct symbols by adding a sentinel for symbol 0.
    // Symbol 0 (eq=0, delta=-RADIUS) is always an outlier and not present in d_symbols.
    {
        int nonzero_bins = 0;
        uint32_t first_nonzero_sym = 0;
        for (int i = 0; i < bklen; i++) if (h_hist[i] > 0) { nonzero_bins++; first_nonzero_sym = i; }
        printf("  [dbg-u2-hist] %s  nonzero_bins=%d  first_sym=%u  h_hist[0]=%u\n",
               name, nonzero_bins, first_nonzero_sym, h_hist[0]);
    }
    if (h_hist[0] == 0) h_hist[0] = 1;

    phf::high_level<uint16_t>::build_book(&hf_buf, h_hist, bklen, stream);
    auto t_book = std::chrono::high_resolution_clock::now();

    uint8_t* encoded = nullptr;
    size_t encoded_len = 0;
    phf_header hf_header{};
    float encode_kernel_ms = 0.0f;
    phf::high_level<uint16_t>::encode_timed(
        &hf_buf, d_symbols, n, &encoded, &encoded_len, hf_header, stream, &encode_kernel_ms);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("  [cuda-error] %s after u2 encode: %s\n", name, cudaGetErrorString(err));
    }
    cudaStreamSynchronize(stream);

    auto t1 = std::chrono::high_resolution_clock::now();
    *out_encode_ms = hist_kernel_ms + encode_kernel_ms;
    *out_encoded_len = encoded_len;
    auto ms = [](auto a, auto b){ return std::chrono::duration<float, std::milli>(b-a).count(); };
    printf("%s  n=%zu  hist=%.2f ms  book=%.2f ms  encode=%.2f ms  total=%.2f ms  kernels=%.2f ms  total_nbit=%zu  encoded_len=%zu\n",
           name, n, ms(t0,t_hist), ms(t_hist,t_book), ms(t_book,t1), ms(t0,t1),
           *out_encode_ms, (size_t)hf_header.total_nbit, encoded_len);

    if (out_h_blob) {
        uint8_t* h_blob = new uint8_t[encoded_len];
        cudaMemcpy(h_blob, encoded, encoded_len, cudaMemcpyDeviceToHost);
        *out_h_blob = h_blob;
    }

    // Decode (verification) removed for speed — decode only happens in hf_u2_decode_from_blob.
    *out_decode_ms = 0.0f;

    cudaFreeHost(h_hist);
    cudaFree(d_hist);
}

struct U2DeviceResult {
    uint8_t* encoded = nullptr;
    size_t encoded_len = 0;
    phf_header header{};
};

}  // namespace

struct GpuHuffmanDecodeWorkspace {
    size_t n = 0;
    bool include_u1_pair = false;
    cudaStream_t stream = nullptr;
    std::array<phf_header, 4> headers{};
    std::array<uint8_t*, 4> d_blobs{};
    std::unique_ptr<phf::Buf<uint16_t>> u2;
    std::unique_ptr<phf::Buf<uint8_t>> u1;

    GpuHuffmanDecodeWorkspace(
        const uint8_t* const h_blobs[4], const size_t blob_lens[4],
        size_t element_count, bool include_u1, cudaStream_t user_stream)
        : n(element_count), include_u1_pair(include_u1), stream(user_stream)
    {
        try {
            const int blob_count = include_u1_pair ? 4 : 2;
            for (int i = 0; i < blob_count; ++i) {
                if (!h_blobs[i] || blob_lens[i] < sizeof(phf_header)) {
                    throw std::runtime_error("Truncated Huffman payload");
                }
                std::memcpy(&headers[i], h_blobs[i], sizeof(phf_header));
                hf_check_cuda(cudaMalloc(&d_blobs[i], blob_lens[i]),
                              "allocate persistent Huffman input");
                hf_check_cuda(cudaMemcpyAsync(
                                  d_blobs[i], h_blobs[i], blob_lens[i],
                                  cudaMemcpyHostToDevice, stream),
                              "upload persistent Huffman input");
            }

            // Buf owns all decoder-side scratch and is intentionally created
            // during preparation, outside the GPU-resident decode call.
            u2 = std::make_unique<phf::Buf<uint16_t>>(
                n, 8192, -1, false, false, false);
            if (include_u1_pair) {
                u1 = std::make_unique<phf::Buf<uint8_t>>(
                    n, 256, -1, false, false, false);
            }
            hf_check_cuda(cudaStreamSynchronize(stream),
                          "finish persistent Huffman setup");
        }
        catch (...) {
            release();
            throw;
        }
    }

    ~GpuHuffmanDecodeWorkspace() { release(); }

    void release()
    {
        u1.reset();
        u2.reset();
        for (auto& blob : d_blobs) {
            if (blob) cudaFree(blob);
            blob = nullptr;
        }
    }
};

GpuHuffmanDecodeWorkspace* create_gpu_huffman_decode_workspace(
    const uint8_t* const h_blobs[4], const size_t blob_lens[4],
    size_t n, bool include_u1_pair, void* stream)
{
    return new GpuHuffmanDecodeWorkspace(
        h_blobs, blob_lens, n, include_u1_pair,
        reinterpret_cast<cudaStream_t>(stream));
}

void destroy_gpu_huffman_decode_workspace(
    GpuHuffmanDecodeWorkspace* workspace)
{
    delete workspace;
}

void run_gpu_huffman_decode_device(
    GpuHuffmanDecodeWorkspace* workspace,
    uint16_t* d_eq_U, uint16_t* d_eq_V,
    uint8_t* d_eq_dEb_U, uint8_t* d_eq_dEb_V)
{
    if (!workspace || !d_eq_U || !d_eq_V) {
        throw std::invalid_argument("Invalid persistent Huffman decode arguments");
    }
    phf::high_level<uint16_t>::decode(
        workspace->u2.get(), workspace->headers[0],
        workspace->d_blobs[0], d_eq_U, workspace->stream);
    phf::high_level<uint16_t>::decode(
        workspace->u2.get(), workspace->headers[1],
        workspace->d_blobs[1], d_eq_V, workspace->stream);
    if (workspace->include_u1_pair) {
        if (!d_eq_dEb_U || !d_eq_dEb_V) {
            throw std::invalid_argument("Missing Huffman U1 decode outputs");
        }
        phf::high_level<uint8_t>::decode(
            workspace->u1.get(), workspace->headers[2],
            workspace->d_blobs[2], d_eq_dEb_U, workspace->stream);
        phf::high_level<uint8_t>::decode(
            workspace->u1.get(), workspace->headers[3],
            workspace->d_blobs[3], d_eq_dEb_V, workspace->stream);
    }
    hf_check_cuda(cudaGetLastError(), "launch persistent Huffman decode");
}

struct GpuHuffmanU2Workspace {
    static constexpr uint16_t bklen = 8192;

    size_t n;
    phf::Buf<uint16_t> hf_buf_U;
    phf::Buf<uint16_t> hf_buf_V;
    uint32_t* d_hist_U = nullptr;
    uint32_t* d_hist_V = nullptr;
    uint32_t* h_hist_U = nullptr;
    uint32_t* h_hist_V = nullptr;
    cudaStream_t stream_V = nullptr;
    cudaEvent_t hist_start_U = nullptr;
    cudaEvent_t hist_stop_U = nullptr;
    cudaEvent_t hist_start_V = nullptr;
    cudaEvent_t hist_stop_V = nullptr;

    explicit GpuHuffmanU2Workspace(size_t element_count)
        : n(element_count),
          hf_buf_U(element_count, bklen, -1, false, false, false),
          hf_buf_V(element_count, bklen, -1, false, false, false)
    {
        cudaMalloc(&d_hist_U, bklen * sizeof(uint32_t));
        cudaMalloc(&d_hist_V, bklen * sizeof(uint32_t));
        cudaMallocHost(&h_hist_U, bklen * sizeof(uint32_t));
        cudaMallocHost(&h_hist_V, bklen * sizeof(uint32_t));
        cudaStreamCreateWithFlags(&stream_V, cudaStreamNonBlocking);
        cudaEventCreate(&hist_start_U);
        cudaEventCreate(&hist_stop_U);
        cudaEventCreate(&hist_start_V);
        cudaEventCreate(&hist_stop_V);
    }

    ~GpuHuffmanU2Workspace()
    {
        if (hist_start_U) cudaEventDestroy(hist_start_U);
        if (hist_stop_U) cudaEventDestroy(hist_stop_U);
        if (hist_start_V) cudaEventDestroy(hist_start_V);
        if (hist_stop_V) cudaEventDestroy(hist_stop_V);
        if (stream_V) cudaStreamDestroy(stream_V);
        if (h_hist_U) cudaFreeHost(h_hist_U);
        if (h_hist_V) cudaFreeHost(h_hist_V);
        if (d_hist_U) cudaFree(d_hist_U);
        if (d_hist_V) cudaFree(d_hist_V);
    }
};

GpuHuffmanU2Workspace* create_gpu_huffman_u2_workspace(size_t n)
{
    return new GpuHuffmanU2Workspace(n);
}

void destroy_gpu_huffman_u2_workspace(GpuHuffmanU2Workspace* workspace)
{
    delete workspace;
}

void run_gpu_huffman_u2_arrays_device(
    GpuHuffmanU2Workspace* workspace,
    uint16_t* eq_U, uint16_t* eq_V,
    size_t n, void* stream,
    size_t out_lens[2], uint8_t* out_d_blobs[2],
    float out_encode_ms[2], float* out_wall_ms, float* out_kernel_ms)
{
    if (!workspace || workspace->n != n) {
        throw std::runtime_error("Huffman workspace size mismatch");
    }

    auto stream_U = reinterpret_cast<cudaStream_t>(stream);
    auto stream_V = workspace->stream_V;
    int grid_dim, block_dim, shmem_use, r_per_block;
    psz::module::GPU_histogram_generic<uint16_t>::init(
        n, GpuHuffmanU2Workspace::bklen,
        grid_dim, block_dim, shmem_use, r_per_block);

    printf("RUN_HF (device-resident output)\n");
    auto wall_start = std::chrono::high_resolution_clock::now();

    cudaMemsetAsync(
        workspace->d_hist_U, 0,
        GpuHuffmanU2Workspace::bklen * sizeof(uint32_t), stream_U);
    cudaMemsetAsync(
        workspace->d_hist_V, 0,
        GpuHuffmanU2Workspace::bklen * sizeof(uint32_t), stream_V);
    cudaEventRecord(workspace->hist_start_U, stream_U);
    psz::module::GPU_histogram_generic<uint16_t>::kernel(
        eq_U, n, workspace->d_hist_U, GpuHuffmanU2Workspace::bklen,
        grid_dim, block_dim, shmem_use, r_per_block, stream_U);
    cudaEventRecord(workspace->hist_stop_U, stream_U);
    cudaEventRecord(workspace->hist_start_V, stream_V);
    psz::module::GPU_histogram_generic<uint16_t>::kernel(
        eq_V, n, workspace->d_hist_V, GpuHuffmanU2Workspace::bklen,
        grid_dim, block_dim, shmem_use, r_per_block, stream_V);
    cudaEventRecord(workspace->hist_stop_V, stream_V);
    cudaMemcpyAsync(
        workspace->h_hist_U, workspace->d_hist_U,
        GpuHuffmanU2Workspace::bklen * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream_U);
    cudaMemcpyAsync(
        workspace->h_hist_V, workspace->d_hist_V,
        GpuHuffmanU2Workspace::bklen * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream_V);
    cudaStreamSynchronize(stream_U);
    cudaStreamSynchronize(stream_V);

    float hist_kernel_U_ms = 0.0f, hist_kernel_V_ms = 0.0f;
    cudaEventElapsedTime(
        &hist_kernel_U_ms, workspace->hist_start_U, workspace->hist_stop_U);
    cudaEventElapsedTime(
        &hist_kernel_V_ms, workspace->hist_start_V, workspace->hist_stop_V);
    auto hist_end = std::chrono::high_resolution_clock::now();

    if (workspace->h_hist_U[0] == 0) workspace->h_hist_U[0] = 1;
    if (workspace->h_hist_V[0] == 0) workspace->h_hist_V[0] = 1;
    std::thread book_U([&] {
        phf::high_level<uint16_t>::build_book(
            &workspace->hf_buf_U, workspace->h_hist_U,
            GpuHuffmanU2Workspace::bklen, stream_U);
    });
    std::thread book_V([&] {
        phf::high_level<uint16_t>::build_book(
            &workspace->hf_buf_V, workspace->h_hist_V,
            GpuHuffmanU2Workspace::bklen, stream_V);
    });
    book_U.join();
    book_V.join();
    auto book_end = std::chrono::high_resolution_clock::now();

    U2DeviceResult result_U, result_V;
    float encode_kernel_U_ms = 0.0f, encode_kernel_V_ms = 0.0f;
    // encode_timed() blocks internally on its own stream (it reads back
    // per-partition bit counts to build the header), so U and V only
    // overlap on the GPU if issued from two host threads.
    std::thread encode_U([&] {
        phf::high_level<uint16_t>::encode_timed(
            &workspace->hf_buf_U, eq_U, n,
            &result_U.encoded, &result_U.encoded_len,
            result_U.header, stream_U, &encode_kernel_U_ms);
    });
    std::thread encode_V([&] {
        phf::high_level<uint16_t>::encode_timed(
            &workspace->hf_buf_V, eq_V, n,
            &result_V.encoded, &result_V.encoded_len,
            result_V.header, stream_V, &encode_kernel_V_ms);
    });
    encode_U.join();
    encode_V.join();
    auto wall_end = std::chrono::high_resolution_clock::now();

    out_lens[0] = result_U.encoded_len;
    out_lens[1] = result_V.encoded_len;
    out_d_blobs[0] = result_U.encoded;
    out_d_blobs[1] = result_V.encoded;
    out_encode_ms[0] = encode_kernel_U_ms;
    out_encode_ms[1] = encode_kernel_V_ms;
    *out_wall_ms = std::chrono::duration<float, std::milli>(
        wall_end - wall_start).count();
    *out_kernel_ms = std::max(hist_kernel_U_ms, hist_kernel_V_ms) +
                     encode_kernel_U_ms + encode_kernel_V_ms;

    auto elapsed = [](auto a, auto b) {
        return std::chrono::duration<float, std::milli>(b - a).count();
    };
    printf("eq_U  n=%zu  encode-kernels=%.2f ms  total_nbit=%zu  encoded_len=%zu\n",
           n, encode_kernel_U_ms, (size_t)result_U.header.total_nbit, out_lens[0]);
    printf("eq_V  n=%zu  encode-kernels=%.2f ms  total_nbit=%zu  encoded_len=%zu\n",
           n, encode_kernel_V_ms, (size_t)result_V.header.total_nbit, out_lens[1]);
    printf("  prep: hist=%.2f ms book=%.2f ms; HF device-ready wall=%.2f ms\n",
           elapsed(wall_start, hist_end), elapsed(hist_end, book_end), *out_wall_ms);
    printf("  kernel-only: hist=max(%.2f, %.2f) ms encode=%.2f+%.2f ms total=%.2f ms\n\n",
           hist_kernel_U_ms, hist_kernel_V_ms,
           encode_kernel_U_ms, encode_kernel_V_ms, *out_kernel_ms);
}

struct GpuHuffmanU1Workspace {
    static constexpr uint16_t bklen = 256;

    size_t n;
    phf::Buf<uint8_t> hf_buf_U;
    phf::Buf<uint8_t> hf_buf_V;
    uint32_t* d_hist_U = nullptr;
    uint32_t* d_hist_V = nullptr;
    uint32_t* h_hist_U = nullptr;
    uint32_t* h_hist_V = nullptr;
    cudaStream_t stream_V = nullptr;
    cudaEvent_t hist_start_U = nullptr;
    cudaEvent_t hist_stop_U = nullptr;
    cudaEvent_t hist_start_V = nullptr;
    cudaEvent_t hist_stop_V = nullptr;

    explicit GpuHuffmanU1Workspace(size_t element_count)
        : n(element_count),
          hf_buf_U(element_count, bklen, -1, false, false, false),
          hf_buf_V(element_count, bklen, -1, false, false, false)
    {
        hf_check_cuda(cudaMalloc(&d_hist_U, bklen * sizeof(uint32_t)),
                      "allocate U1 U histogram");
        hf_check_cuda(cudaMalloc(&d_hist_V, bklen * sizeof(uint32_t)),
                      "allocate U1 V histogram");
        hf_check_cuda(cudaMallocHost(&h_hist_U, bklen * sizeof(uint32_t)),
                      "allocate U1 host U histogram");
        hf_check_cuda(cudaMallocHost(&h_hist_V, bklen * sizeof(uint32_t)),
                      "allocate U1 host V histogram");
        hf_check_cuda(cudaStreamCreateWithFlags(&stream_V, cudaStreamNonBlocking),
                      "create U1 V stream");
        hf_check_cuda(cudaEventCreate(&hist_start_U), "create U1 U start event");
        hf_check_cuda(cudaEventCreate(&hist_stop_U), "create U1 U stop event");
        hf_check_cuda(cudaEventCreate(&hist_start_V), "create U1 V start event");
        hf_check_cuda(cudaEventCreate(&hist_stop_V), "create U1 V stop event");
    }

    ~GpuHuffmanU1Workspace()
    {
        if (hist_start_U) cudaEventDestroy(hist_start_U);
        if (hist_stop_U) cudaEventDestroy(hist_stop_U);
        if (hist_start_V) cudaEventDestroy(hist_start_V);
        if (hist_stop_V) cudaEventDestroy(hist_stop_V);
        if (stream_V) cudaStreamDestroy(stream_V);
        if (h_hist_U) cudaFreeHost(h_hist_U);
        if (h_hist_V) cudaFreeHost(h_hist_V);
        if (d_hist_U) cudaFree(d_hist_U);
        if (d_hist_V) cudaFree(d_hist_V);
    }
};

GpuHuffmanU1Workspace* create_gpu_huffman_u1_workspace(size_t n)
{
    return new GpuHuffmanU1Workspace(n);
}

void destroy_gpu_huffman_u1_workspace(GpuHuffmanU1Workspace* workspace)
{
    delete workspace;
}

void run_gpu_huffman_u1_arrays_device(
    GpuHuffmanU1Workspace* workspace,
    uint8_t* eq_U, uint8_t* eq_V,
    size_t n, void* stream,
    size_t out_lens[2], uint8_t* out_d_blobs[2],
    float out_encode_ms[2], float* out_wall_ms, float* out_kernel_ms)
{
    if (!workspace || workspace->n != n || !eq_U || !eq_V) {
        throw std::runtime_error("Huffman U1 workspace arguments mismatch");
    }
    auto stream_U = reinterpret_cast<cudaStream_t>(stream);
    auto stream_V = workspace->stream_V;
    int grid_dim, block_dim, shmem_use, r_per_block;
    psz::module::GPU_histogram_generic<uint8_t>::init(
        n, GpuHuffmanU1Workspace::bklen,
        grid_dim, block_dim, shmem_use, r_per_block);

    auto wall_start = std::chrono::high_resolution_clock::now();
    cudaMemsetAsync(workspace->d_hist_U, 0,
                    GpuHuffmanU1Workspace::bklen * sizeof(uint32_t), stream_U);
    cudaMemsetAsync(workspace->d_hist_V, 0,
                    GpuHuffmanU1Workspace::bklen * sizeof(uint32_t), stream_V);
    cudaEventRecord(workspace->hist_start_U, stream_U);
    psz::module::GPU_histogram_generic<uint8_t>::kernel(
        eq_U, n, workspace->d_hist_U, GpuHuffmanU1Workspace::bklen,
        grid_dim, block_dim, shmem_use, r_per_block, stream_U);
    cudaEventRecord(workspace->hist_stop_U, stream_U);
    cudaEventRecord(workspace->hist_start_V, stream_V);
    psz::module::GPU_histogram_generic<uint8_t>::kernel(
        eq_V, n, workspace->d_hist_V, GpuHuffmanU1Workspace::bklen,
        grid_dim, block_dim, shmem_use, r_per_block, stream_V);
    cudaEventRecord(workspace->hist_stop_V, stream_V);
    cudaMemcpyAsync(workspace->h_hist_U, workspace->d_hist_U,
                    GpuHuffmanU1Workspace::bklen * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream_U);
    cudaMemcpyAsync(workspace->h_hist_V, workspace->d_hist_V,
                    GpuHuffmanU1Workspace::bklen * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream_V);
    cudaStreamSynchronize(stream_U);
    cudaStreamSynchronize(stream_V);

    float hist_U_ms = 0.0f, hist_V_ms = 0.0f;
    cudaEventElapsedTime(&hist_U_ms, workspace->hist_start_U, workspace->hist_stop_U);
    cudaEventElapsedTime(&hist_V_ms, workspace->hist_start_V, workspace->hist_stop_V);
    if (workspace->h_hist_U[0] == 0) workspace->h_hist_U[0] = 1;
    if (workspace->h_hist_V[0] == 0) workspace->h_hist_V[0] = 1;

    std::thread book_U([&] {
        phf::high_level<uint8_t>::build_book(
            &workspace->hf_buf_U, workspace->h_hist_U,
            GpuHuffmanU1Workspace::bklen, stream_U);
    });
    std::thread book_V([&] {
        phf::high_level<uint8_t>::build_book(
            &workspace->hf_buf_V, workspace->h_hist_V,
            GpuHuffmanU1Workspace::bklen, stream_V);
    });
    book_U.join();
    book_V.join();

    U2DeviceResult result_U, result_V;
    float encode_U_ms = 0.0f, encode_V_ms = 0.0f;
    // encode_timed() blocks internally on its own stream (it reads back
    // per-partition bit counts to build the header), so U and V only
    // overlap on the GPU if issued from two host threads.
    std::thread encode_U([&] {
        phf::high_level<uint8_t>::encode_timed(
            &workspace->hf_buf_U, eq_U, n,
            &result_U.encoded, &result_U.encoded_len,
            result_U.header, stream_U, &encode_U_ms);
    });
    std::thread encode_V([&] {
        phf::high_level<uint8_t>::encode_timed(
            &workspace->hf_buf_V, eq_V, n,
            &result_V.encoded, &result_V.encoded_len,
            result_V.header, stream_V, &encode_V_ms);
    });
    encode_U.join();
    encode_V.join();

    out_lens[0] = result_U.encoded_len;
    out_lens[1] = result_V.encoded_len;
    out_d_blobs[0] = result_U.encoded;
    out_d_blobs[1] = result_V.encoded;
    out_encode_ms[0] = encode_U_ms;
    out_encode_ms[1] = encode_V_ms;
    *out_kernel_ms = std::max(hist_U_ms, hist_V_ms) + encode_U_ms + encode_V_ms;
    *out_wall_ms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - wall_start).count();
}

extern "C" void run_gpu_huffman_u1_arrays(
    uint8_t* eq_U, uint8_t* eq_V, uint8_t* eq_dEb_U, uint8_t* eq_dEb_V,
    size_t n, void* stream,
    size_t out_lens[4], float out_encode_ms[4], float out_decode_ms[4],
    uint8_t* out_h_blobs[4])
{
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr uint16_t bklen = 256;

    // Single allocation shared across all 4 arrays.
    // build_book overwrites the codebook before each encode, so reuse is safe.
    phf::Buf<uint8_t> hf_buf(n, bklen, -1, false, false, false);

    printf("RUN_HF\n");
    hf_encode_decode_u1("eq_U",     eq_U,     n, cuda_stream, hf_buf, &out_lens[0], &out_encode_ms[0], &out_decode_ms[0], out_h_blobs ? &out_h_blobs[0] : nullptr);
    hf_encode_decode_u1("eq_V",     eq_V,     n, cuda_stream, hf_buf, &out_lens[1], &out_encode_ms[1], &out_decode_ms[1], out_h_blobs ? &out_h_blobs[1] : nullptr);
    hf_encode_decode_u1("eq_dEb_U", eq_dEb_U, n, cuda_stream, hf_buf, &out_lens[2], &out_encode_ms[2], &out_decode_ms[2], out_h_blobs ? &out_h_blobs[2] : nullptr);
    hf_encode_decode_u1("eq_dEb_V", eq_dEb_V, n, cuda_stream, hf_buf, &out_lens[3], &out_encode_ms[3], &out_decode_ms[3], out_h_blobs ? &out_h_blobs[3] : nullptr);
    printf("\n");
}

// eq_U/V are uint16_t (RADIUS=32768, no outliers); eq_dEb_U/V remain uint8_t.
extern "C" void run_gpu_huffman_u2_u1_arrays(
    uint16_t* eq_U, uint16_t* eq_V,
    uint8_t* eq_dEb_U, uint8_t* eq_dEb_V,
    size_t n, void* stream,
    size_t out_lens[4], float out_encode_ms[4], float out_decode_ms[4],
    uint8_t* out_h_blobs[4])
{
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    phf::Buf<uint16_t> hf_buf_u2(n, 8192, -1, false, false, false);
    phf::Buf<uint8_t>  hf_buf_u1(n, 256, -1, false, false, false);

    printf("RUN_HF\n");
    hf_encode_decode_u2("eq_U",     eq_U,     n, cuda_stream, hf_buf_u2, &out_lens[0], &out_encode_ms[0], &out_decode_ms[0], out_h_blobs ? &out_h_blobs[0] : nullptr);
    hf_encode_decode_u2("eq_V",     eq_V,     n, cuda_stream, hf_buf_u2, &out_lens[1], &out_encode_ms[1], &out_decode_ms[1], out_h_blobs ? &out_h_blobs[1] : nullptr);
    hf_encode_decode_u1("eq_dEb_U", eq_dEb_U, n, cuda_stream, hf_buf_u1, &out_lens[2], &out_encode_ms[2], &out_decode_ms[2], out_h_blobs ? &out_h_blobs[2] : nullptr);
    hf_encode_decode_u1("eq_dEb_V", eq_dEb_V, n, cuda_stream, hf_buf_u1, &out_lens[3], &out_encode_ms[3], &out_decode_ms[3], out_h_blobs ? &out_h_blobs[3] : nullptr);
    printf("\n");
}

// Tile-uniform EB path stores one EB symbol per Lorenzo tile, so only eq_U/V
// need full-size Huffman blobs.
extern "C" void run_gpu_huffman_u2_arrays_timed(
    uint16_t* eq_U, uint16_t* eq_V,
    size_t n, void* stream,
    size_t out_lens[2], float out_encode_ms[2], float out_decode_ms[2],
    uint8_t* out_h_blobs[2], float* out_parallel_wall_ms, float* out_kernel_ms)
{
    constexpr uint16_t bklen = 8192;
    auto stream_U = reinterpret_cast<cudaStream_t>(stream);
    cudaStream_t stream_V;
    cudaStreamCreateWithFlags(&stream_V, cudaStreamNonBlocking);

    phf::Buf<uint16_t> hf_buf_U(n, bklen, -1, false, false, false);
    phf::Buf<uint16_t> hf_buf_V(n, bklen, -1, false, false, false);
    uint32_t *d_hist_U = nullptr, *d_hist_V = nullptr;
    uint32_t *h_hist_U = nullptr, *h_hist_V = nullptr;
    cudaMalloc(&d_hist_U, bklen * sizeof(uint32_t));
    cudaMalloc(&d_hist_V, bklen * sizeof(uint32_t));
    cudaMallocHost(&h_hist_U, bklen * sizeof(uint32_t));
    cudaMallocHost(&h_hist_V, bklen * sizeof(uint32_t));
    U2DeviceResult result_U, result_V;
    int grid_dim, block_dim, shmem_use, r_per_block;
    psz::module::GPU_histogram_generic<uint16_t>::init(
        n, bklen, grid_dim, block_dim, shmem_use, r_per_block);

    printf("RUN_HF\n");
    auto wall_start = std::chrono::high_resolution_clock::now();

    // Histograms are light enough to overlap efficiently.
    cudaMemsetAsync(d_hist_U, 0, bklen * sizeof(uint32_t), stream_U);
    cudaMemsetAsync(d_hist_V, 0, bklen * sizeof(uint32_t), stream_V);
    cudaEvent_t hist_start_U, hist_stop_U, hist_start_V, hist_stop_V;
    cudaEventCreate(&hist_start_U);
    cudaEventCreate(&hist_stop_U);
    cudaEventCreate(&hist_start_V);
    cudaEventCreate(&hist_stop_V);
    cudaEventRecord(hist_start_U, stream_U);
    psz::module::GPU_histogram_generic<uint16_t>::kernel(
        eq_U, n, d_hist_U, bklen,
        grid_dim, block_dim, shmem_use, r_per_block, stream_U);
    cudaEventRecord(hist_stop_U, stream_U);
    cudaEventRecord(hist_start_V, stream_V);
    psz::module::GPU_histogram_generic<uint16_t>::kernel(
        eq_V, n, d_hist_V, bklen,
        grid_dim, block_dim, shmem_use, r_per_block, stream_V);
    cudaEventRecord(hist_stop_V, stream_V);
    cudaMemcpyAsync(
        h_hist_U, d_hist_U, bklen * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream_U);
    cudaMemcpyAsync(
        h_hist_V, d_hist_V, bklen * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream_V);
    cudaStreamSynchronize(stream_U);
    cudaStreamSynchronize(stream_V);
    float hist_kernel_U_ms = 0.0f, hist_kernel_V_ms = 0.0f;
    cudaEventElapsedTime(&hist_kernel_U_ms, hist_start_U, hist_stop_U);
    cudaEventElapsedTime(&hist_kernel_V_ms, hist_start_V, hist_stop_V);
    cudaEventDestroy(hist_start_U);
    cudaEventDestroy(hist_stop_U);
    cudaEventDestroy(hist_start_V);
    cudaEventDestroy(hist_stop_V);
    auto hist_end = std::chrono::high_resolution_clock::now();

    if (h_hist_U[0] == 0) h_hist_U[0] = 1;
    if (h_hist_V[0] == 0) h_hist_V[0] = 1;
    std::thread book_U([&] {
        phf::high_level<uint16_t>::build_book(&hf_buf_U, h_hist_U, bklen, stream_U);
    });
    std::thread book_V([&] {
        phf::high_level<uint16_t>::build_book(&hf_buf_V, h_hist_V, bklen, stream_V);
    });
    book_U.join();
    book_V.join();
    auto book_end = std::chrono::high_resolution_clock::now();

    // PHF encode saturates the GPU; overlapping two encodes causes severe
    // contention on large arrays, so execute only this heavy phase serially.
    float encode_kernel_U_ms = 0.0f, encode_kernel_V_ms = 0.0f;
    phf::high_level<uint16_t>::encode_timed(
        &hf_buf_U, eq_U, n, &result_U.encoded, &result_U.encoded_len,
        result_U.header, stream_U, &encode_kernel_U_ms);
    cudaStreamSynchronize(stream_U);
    phf::high_level<uint16_t>::encode_timed(
        &hf_buf_V, eq_V, n, &result_V.encoded, &result_V.encoded_len,
        result_V.header, stream_V, &encode_kernel_V_ms);
    cudaStreamSynchronize(stream_V);
    auto wall_end = std::chrono::high_resolution_clock::now();
    out_encode_ms[0] = encode_kernel_U_ms;
    out_encode_ms[1] = encode_kernel_V_ms;
    *out_parallel_wall_ms =
        std::chrono::duration<float, std::milli>(wall_end - wall_start).count();
    if (out_kernel_ms) {
        *out_kernel_ms = std::max(hist_kernel_U_ms, hist_kernel_V_ms) +
                         encode_kernel_U_ms + encode_kernel_V_ms;
    }

    out_lens[0] = result_U.encoded_len;
    out_lens[1] = result_V.encoded_len;
    out_decode_ms[0] = out_decode_ms[1] = 0.0f;
    if (out_h_blobs) {
        out_h_blobs[0] = new uint8_t[out_lens[0]];
        out_h_blobs[1] = new uint8_t[out_lens[1]];
        cudaMemcpy(out_h_blobs[0], result_U.encoded, out_lens[0], cudaMemcpyDeviceToHost);
        cudaMemcpy(out_h_blobs[1], result_V.encoded, out_lens[1], cudaMemcpyDeviceToHost);
    }
    auto elapsed = [](auto a, auto b) {
        return std::chrono::duration<float, std::milli>(b - a).count();
    };
    printf("eq_U  n=%zu  encode-kernels=%.2f ms  total_nbit=%zu  encoded_len=%zu\n",
           n, out_encode_ms[0], (size_t)result_U.header.total_nbit, out_lens[0]);
    printf("eq_V  n=%zu  encode-kernels=%.2f ms  total_nbit=%zu  encoded_len=%zu\n",
           n, out_encode_ms[1], (size_t)result_V.header.total_nbit, out_lens[1]);
    printf("  parallel prep: hist=%.2f ms book=%.2f ms; HF wall=%.2f ms\n",
           elapsed(wall_start, hist_end), elapsed(hist_end, book_end),
           *out_parallel_wall_ms);
    printf("  kernel-only: hist=max(%.2f, %.2f) ms encode=%.2f+%.2f ms total=%.2f ms\n",
           hist_kernel_U_ms, hist_kernel_V_ms, encode_kernel_U_ms, encode_kernel_V_ms,
           out_kernel_ms ? *out_kernel_ms : 0.0f);
    printf("\n");

    cudaFreeHost(h_hist_U);
    cudaFreeHost(h_hist_V);
    cudaFree(d_hist_U);
    cudaFree(d_hist_V);
    cudaStreamDestroy(stream_V);
}

extern "C" void run_gpu_huffman_u2_arrays(
    uint16_t* eq_U, uint16_t* eq_V,
    size_t n, void* stream,
    size_t out_lens[2], float out_encode_ms[2], float out_decode_ms[2],
    uint8_t* out_h_blobs[2], float* out_parallel_wall_ms)
{
    float kernel_ms = 0.0f;
    run_gpu_huffman_u2_arrays_timed(
        eq_U, eq_V, n, stream, out_lens, out_encode_ms, out_decode_ms,
        out_h_blobs, out_parallel_wall_ms, &kernel_ms);
}

// Pipelined HF: eq_dEb arrays are encoded on bg_stream (overlaps Lorenzo on main_stream).
// eq_U/V are encoded on main_stream after Lorenzo completes.
// bg_stream must already have DEAL_WITH_EBZERO results visible (cudaStreamSynchronize called).
extern "C" void run_gpu_huffman_u2_u1_pipelined(
    uint16_t* eq_U, uint16_t* eq_V,
    uint8_t* eq_dEb_U, uint8_t* eq_dEb_V,
    size_t n, void* main_stream, void* bg_stream,
    size_t out_lens[4], float out_encode_ms[4], float out_decode_ms[4],
    uint8_t* out_h_blobs[4])
{
    auto ms = reinterpret_cast<cudaStream_t>(main_stream);
    auto bs = reinterpret_cast<cudaStream_t>(bg_stream);

    // Each stream needs its OWN Buf (separate GPU allocations, no sharing)
    phf::Buf<uint16_t> hf_u2_A(n, 8192, -1, false, false, false);  // eq_U
    phf::Buf<uint16_t> hf_u2_B(n, 8192, -1, false, false, false);  // eq_V
    phf::Buf<uint8_t>  hf_u1_A(n, 256, -1, false, false, false);   // eq_dEb_U
    phf::Buf<uint8_t>  hf_u1_B(n, 256, -1, false, false, false);   // eq_dEb_V

    // dEb on bg_stream: starts NOW, overlaps Lorenzo on main_stream
    hf_encode_decode_u1("eq_dEb_U", eq_dEb_U, n, bs, hf_u1_A, &out_lens[2], &out_encode_ms[2], &out_decode_ms[2], out_h_blobs ? &out_h_blobs[2] : nullptr);
    hf_encode_decode_u1("eq_dEb_V", eq_dEb_V, n, bs, hf_u1_B, &out_lens[3], &out_encode_ms[3], &out_decode_ms[3], out_h_blobs ? &out_h_blobs[3] : nullptr);

    // eq_U/V on main_stream: caller ensures Lorenzo is complete before calling
    hf_encode_decode_u2("eq_U", eq_U, n, ms, hf_u2_A, &out_lens[0], &out_encode_ms[0], &out_decode_ms[0], out_h_blobs ? &out_h_blobs[0] : nullptr);
    hf_encode_decode_u2("eq_V", eq_V, n, ms, hf_u2_B, &out_lens[1], &out_encode_ms[1], &out_decode_ms[1], out_h_blobs ? &out_h_blobs[1] : nullptr);
    printf("\n");
}

extern "C" void hf_u1_decode_from_blob(
    uint8_t* h_blob, size_t blob_len, size_t n,
    uint8_t* d_decoded, void* stream)
{
    constexpr uint16_t bklen = 256;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    uint8_t* d_blob;
    cudaMalloc(&d_blob, blob_len);
    cudaMemcpy(d_blob, h_blob, blob_len, cudaMemcpyHostToDevice);
    phf_header hdr = *reinterpret_cast<phf_header*>(h_blob);
    phf::Buf<uint8_t> buf(n, bklen, -1, false, false, false);
    phf::high_level<uint8_t>::decode(&buf, hdr, d_blob, d_decoded, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
    cudaFree(d_blob);
}

extern "C" void hf_u2_decode_from_blob(
    uint8_t* h_blob, size_t blob_len, size_t n,
    uint16_t* d_decoded, void* stream)
{
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    uint8_t* d_blob;
    cudaMalloc(&d_blob, blob_len);
    cudaMemcpy(d_blob, h_blob, blob_len, cudaMemcpyHostToDevice);
    phf_header hdr = *reinterpret_cast<phf_header*>(h_blob);
    phf::Buf<uint16_t> buf(n, 8192, -1, false, false, false);
    phf::high_level<uint16_t>::decode(&buf, hdr, d_blob, d_decoded, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
    {
        uint16_t h_check[5];
        cudaMemcpy(h_check, d_decoded, 5*sizeof(uint16_t), cudaMemcpyDeviceToHost);
        printf("  [dbg-u2-blob] decoded[0..4]=%u %u %u %u %u  (hdr.bklen=%d sublen=%d pardeg=%d)\n",
               h_check[0],h_check[1],h_check[2],h_check[3],h_check[4],
               hdr.bklen, (int)hdr.sublen, (int)hdr.pardeg);
    }
    cudaFree(d_blob);
}
