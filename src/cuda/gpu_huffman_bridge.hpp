#ifndef GPU_HUFFMAN_BRIDGE_HPP
#define GPU_HUFFMAN_BRIDGE_HPP

#include <cstddef>
#include <cstdint>

struct GpuHuffmanU2Workspace;
struct GpuHuffmanU1Workspace;
struct GpuHuffmanDecodeWorkspace;

GpuHuffmanU2Workspace* create_gpu_huffman_u2_workspace(std::size_t n);
void destroy_gpu_huffman_u2_workspace(GpuHuffmanU2Workspace* workspace);

void run_gpu_huffman_u2_arrays_device(
    GpuHuffmanU2Workspace* workspace,
    std::uint16_t* eq_U, std::uint16_t* eq_V,
    std::size_t n, void* stream,
    std::size_t out_lens[2], std::uint8_t* out_d_blobs[2],
    float out_encode_ms[2], float* out_wall_ms, float* out_kernel_ms);

GpuHuffmanU1Workspace* create_gpu_huffman_u1_workspace(std::size_t n);
void destroy_gpu_huffman_u1_workspace(GpuHuffmanU1Workspace* workspace);

void run_gpu_huffman_u1_arrays_device(
    GpuHuffmanU1Workspace* workspace,
    std::uint8_t* eq_U, std::uint8_t* eq_V,
    std::size_t n, void* stream,
    std::size_t out_lens[2], std::uint8_t* out_d_blobs[2],
    float out_encode_ms[2], float* out_wall_ms, float* out_kernel_ms);

GpuHuffmanDecodeWorkspace* create_gpu_huffman_decode_workspace(
    const std::uint8_t* const h_blobs[4], const std::size_t blob_lens[4],
    std::size_t n, bool include_u1_pair, void* stream);
void destroy_gpu_huffman_decode_workspace(
    GpuHuffmanDecodeWorkspace* workspace);

void run_gpu_huffman_decode_device(
    GpuHuffmanDecodeWorkspace* workspace,
    std::uint16_t* d_eq_U, std::uint16_t* d_eq_V,
    std::uint8_t* d_eq_dEb_U, std::uint8_t* d_eq_dEb_V);

#endif
