#include <cstdint>
#include <cstdio>
#include <chrono>

#include <cuda_runtime.h>
#include "hf_hl.hh"
#include "kernel/hist.hh"

namespace {

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

    psz::module::GPU_histogram_generic<uint8_t>::kernel(
        d_symbols, n, d_hist, bklen, grid_dim, block_dim, shmem_use, r_per_block, stream);
    cudaMemcpyAsync(h_hist, d_hist, bklen * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    auto t_hist = std::chrono::high_resolution_clock::now();

    // buf is pre-allocated — no construction cost here
    // PHF library bug: single-symbol tree assigns bitcount=0 → empty bitstream.
    if (h_hist[0] == 0) h_hist[0] = 1;
    phf::high_level<uint8_t>::build_book(&hf_buf, h_hist, bklen, stream);
    auto t_book = std::chrono::high_resolution_clock::now();

    uint8_t* encoded = nullptr;
    size_t encoded_len = 0;
    phf_header hf_header{};
    phf::high_level<uint8_t>::encode(&hf_buf, d_symbols, n, &encoded, &encoded_len, hf_header, stream);
    cudaStreamSynchronize(stream);

    auto t1 = std::chrono::high_resolution_clock::now();
    *out_encode_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    *out_encoded_len = encoded_len;
    auto ms = [](auto a, auto b){ return std::chrono::duration<float, std::milli>(b-a).count(); };
    printf("%s  n=%zu  hist=%.2f ms  book=%.2f ms  encode=%.2f ms  total=%.2f ms\n",
           name, n, ms(t0,t_hist), ms(t_hist,t_book), ms(t_book,t1), ms(t0,t1));

    // Copy blob to host before hf_buf is reused for the next array
    if (out_h_blob) {
        uint8_t* h_blob = new uint8_t[encoded_len];
        cudaMemcpy(h_blob, encoded, encoded_len, cudaMemcpyDeviceToHost);
        *out_h_blob = h_blob;  // caller owns, must delete[]
    }

    // Decode timing (same hf_buf state, called before next build_book)
    uint8_t* d_decoded = nullptr;
    cudaMalloc(&d_decoded, n * sizeof(uint8_t));
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0, stream);
    phf::high_level<uint8_t>::decode(&hf_buf, hf_header, encoded, d_decoded, stream);
    cudaEventRecord(ev1, stream);
    cudaStreamSynchronize(stream);
    cudaEventElapsedTime(out_decode_ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_decoded);

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

    psz::module::GPU_histogram_generic<uint16_t>::kernel(
        d_symbols, n, d_hist, bklen, grid_dim, block_dim, shmem_use, r_per_block, stream);
    cudaMemcpyAsync(h_hist, d_hist, bklen * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
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
    phf::high_level<uint16_t>::encode(&hf_buf, d_symbols, n, &encoded, &encoded_len, hf_header, stream);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("  [cuda-error] %s after u2 encode: %s\n", name, cudaGetErrorString(err));
    }
    cudaStreamSynchronize(stream);

    auto t1 = std::chrono::high_resolution_clock::now();
    *out_encode_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    *out_encoded_len = encoded_len;
    auto ms = [](auto a, auto b){ return std::chrono::duration<float, std::milli>(b-a).count(); };
    printf("%s  n=%zu  hist=%.2f ms  book=%.2f ms  encode=%.2f ms  total=%.2f ms  total_nbit=%zu  encoded_len=%zu\n",
           name, n, ms(t0,t_hist), ms(t_hist,t_book), ms(t_book,t1), ms(t0,t1),
           (size_t)hf_header.total_nbit, encoded_len);

    if (out_h_blob) {
        uint8_t* h_blob = new uint8_t[encoded_len];
        cudaMemcpy(h_blob, encoded, encoded_len, cudaMemcpyDeviceToHost);
        *out_h_blob = h_blob;
    }

    uint16_t* d_decoded = nullptr;
    cudaMalloc(&d_decoded, n * sizeof(uint16_t));
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0, stream);
    phf::high_level<uint16_t>::decode(&hf_buf, hf_header, encoded, d_decoded, stream);
    cudaEventRecord(ev1, stream);
    cudaStreamSynchronize(stream);
    cudaEventElapsedTime(out_decode_ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    {
        uint16_t h_check_in[5], h_check_out[5];
        cudaMemcpy(h_check_in,  d_symbols, 5*sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_check_out, d_decoded, 5*sizeof(uint16_t), cudaMemcpyDeviceToHost);
        printf("  [dbg-u2-internal] %s  in[0..4]=%u %u %u %u %u  decoded[0..4]=%u %u %u %u %u\n",
               name, h_check_in[0],h_check_in[1],h_check_in[2],h_check_in[3],h_check_in[4],
               h_check_out[0],h_check_out[1],h_check_out[2],h_check_out[3],h_check_out[4]);
    }
    cudaFree(d_decoded);

    cudaFreeHost(h_hist);
    cudaFree(d_hist);
}

}  // namespace

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
    phf::Buf<uint8_t> hf_buf(n, bklen);

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

    phf::Buf<uint16_t> hf_buf_u2(n, 8192);
    phf::Buf<uint8_t>  hf_buf_u1(n, 256);

    printf("RUN_HF\n");
    hf_encode_decode_u2("eq_U",     eq_U,     n, cuda_stream, hf_buf_u2, &out_lens[0], &out_encode_ms[0], &out_decode_ms[0], out_h_blobs ? &out_h_blobs[0] : nullptr);
    hf_encode_decode_u2("eq_V",     eq_V,     n, cuda_stream, hf_buf_u2, &out_lens[1], &out_encode_ms[1], &out_decode_ms[1], out_h_blobs ? &out_h_blobs[1] : nullptr);
    hf_encode_decode_u1("eq_dEb_U", eq_dEb_U, n, cuda_stream, hf_buf_u1, &out_lens[2], &out_encode_ms[2], &out_decode_ms[2], out_h_blobs ? &out_h_blobs[2] : nullptr);
    hf_encode_decode_u1("eq_dEb_V", eq_dEb_V, n, cuda_stream, hf_buf_u1, &out_lens[3], &out_encode_ms[3], &out_decode_ms[3], out_h_blobs ? &out_h_blobs[3] : nullptr);
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
    phf::Buf<uint8_t> buf(n, bklen);
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
    phf::Buf<uint16_t> buf(n, 8192);
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
