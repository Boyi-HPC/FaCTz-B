#include <chrono>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include "hf_hl.hh"
#include "kernel/hist.hh"

namespace {

void encode_decode_u2(
    const char* name,
    uint16_t* symbols,
    size_t n,
    cudaStream_t stream,
    phf::Buf<uint16_t>& buffer,
    size_t* encoded_len_out,
    float* encode_ms_out,
    float* decode_ms_out,
    uint8_t** host_blob_out)
{
    constexpr uint16_t book_length = 8192;
    uint32_t* device_histogram = nullptr;
    uint32_t* host_histogram = nullptr;
    cudaMalloc(&device_histogram, book_length * sizeof(uint32_t));
    cudaMallocHost(&host_histogram, book_length * sizeof(uint32_t));
    cudaMemsetAsync(device_histogram, 0, book_length * sizeof(uint32_t), stream);

    int grid_dim, block_dim, shared_bytes, symbols_per_block;
    psz::module::GPU_histogram_generic<uint16_t>::init(
        n, book_length, grid_dim, block_dim, shared_bytes, symbols_per_block);

    const auto start = std::chrono::high_resolution_clock::now();
    psz::module::GPU_histogram_generic<uint16_t>::kernel(
        symbols, n, device_histogram, book_length, grid_dim, block_dim,
        shared_bytes, symbols_per_block, stream);
    cudaMemcpyAsync(
        host_histogram, device_histogram, book_length * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const auto histogram_done = std::chrono::high_resolution_clock::now();

    // PHF requires at least two symbols to construct a non-empty bitstream.
    if (host_histogram[0] == 0) host_histogram[0] = 1;
    phf::high_level<uint16_t>::build_book(
        &buffer, host_histogram, book_length, stream);
    const auto book_done = std::chrono::high_resolution_clock::now();

    uint8_t* encoded = nullptr;
    size_t encoded_len = 0;
    phf_header header{};
    phf::high_level<uint16_t>::encode(
        &buffer, symbols, n, &encoded, &encoded_len, header, stream);
    cudaStreamSynchronize(stream);
    const auto encode_done = std::chrono::high_resolution_clock::now();

    const auto elapsed_ms = [](auto a, auto b) {
        return std::chrono::duration<float, std::milli>(b - a).count();
    };
    *encode_ms_out = elapsed_ms(start, encode_done);
    *encoded_len_out = encoded_len;
    std::printf(
        "%s  n=%zu  hist=%.2f ms  book=%.2f ms  encode=%.2f ms  total=%.2f ms\n",
        name, n, elapsed_ms(start, histogram_done),
        elapsed_ms(histogram_done, book_done),
        elapsed_ms(book_done, encode_done), *encode_ms_out);

    if (host_blob_out) {
        auto* host_blob = new uint8_t[encoded_len];
        cudaMemcpy(host_blob, encoded, encoded_len, cudaMemcpyDeviceToHost);
        *host_blob_out = host_blob;
    }

    uint16_t* decoded = nullptr;
    cudaMalloc(&decoded, n * sizeof(uint16_t));
    cudaEvent_t decode_start, decode_stop;
    cudaEventCreate(&decode_start);
    cudaEventCreate(&decode_stop);
    cudaEventRecord(decode_start, stream);
    phf::high_level<uint16_t>::decode(
        &buffer, header, encoded, decoded, stream);
    cudaEventRecord(decode_stop, stream);
    cudaStreamSynchronize(stream);
    cudaEventElapsedTime(decode_ms_out, decode_start, decode_stop);
    cudaEventDestroy(decode_start);
    cudaEventDestroy(decode_stop);
    cudaFree(decoded);
    cudaFreeHost(host_histogram);
    cudaFree(device_histogram);
}

}  // namespace

extern "C" void run_gpu_huffman_u2_arrays(
    uint16_t* eq_U,
    uint16_t* eq_V,
    size_t n,
    void* stream,
    size_t out_lens[2],
    float out_encode_ms[2],
    float out_decode_ms[2],
    uint8_t* out_h_blobs[2])
{
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    phf::Buf<uint16_t> buffer(n, 8192);
    encode_decode_u2(
        "eq_U", eq_U, n, cuda_stream, buffer, &out_lens[0],
        &out_encode_ms[0], &out_decode_ms[0],
        out_h_blobs ? &out_h_blobs[0] : nullptr);
    encode_decode_u2(
        "eq_V", eq_V, n, cuda_stream, buffer, &out_lens[1],
        &out_encode_ms[1], &out_decode_ms[1],
        out_h_blobs ? &out_h_blobs[1] : nullptr);
}
