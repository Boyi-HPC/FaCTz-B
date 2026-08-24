#ifndef GPU_ANS_BRIDGE_HPP
#define GPU_ANS_BRIDGE_HPP

#include <cstddef>
#include <cstdint>

struct GpuAnsU2Workspace;
struct GpuAnsU1Workspace;
struct GpuAnsDecodeWorkspace;

GpuAnsU2Workspace* create_gpu_ans_u2_workspace(size_t n, void* stream);
void destroy_gpu_ans_u2_workspace(GpuAnsU2Workspace* workspace);

void run_gpu_ans_u2_arrays_device(
    GpuAnsU2Workspace* workspace,
    const uint16_t* eq_U, const uint16_t* eq_V,
    size_t n, float* out_kernel_ms);

void download_gpu_ans_u2_arrays(
    GpuAnsU2Workspace* workspace,
    uint8_t* out_h_blobs[2], size_t out_lens[2]);

GpuAnsU1Workspace* create_gpu_ans_u1_workspace(size_t n, void* stream);
void destroy_gpu_ans_u1_workspace(GpuAnsU1Workspace* workspace);

void run_gpu_ans_u1_arrays_device(
    GpuAnsU1Workspace* workspace,
    const uint8_t* eq_U, const uint8_t* eq_V,
    size_t n, float* out_kernel_ms);

void download_gpu_ans_u1_arrays(
    GpuAnsU1Workspace* workspace,
    uint8_t* out_h_blobs[2], size_t out_lens[2]);

GpuAnsDecodeWorkspace* create_gpu_ans_decode_workspace(
    const std::uint8_t* const h_blobs[4], const std::size_t blob_lens[4],
    std::size_t n, bool include_u1_pair, void* stream);
void destroy_gpu_ans_decode_workspace(GpuAnsDecodeWorkspace* workspace);

void run_gpu_ans_decode_device(
    GpuAnsDecodeWorkspace* workspace,
    std::uint16_t* d_eq_U, std::uint16_t* d_eq_V,
    std::uint8_t* d_eq_dEb_U, std::uint8_t* d_eq_dEb_V);
void check_gpu_ans_decode_status(GpuAnsDecodeWorkspace* workspace);

void run_gpu_ans_u2_arrays_timed(
    const uint16_t* eq_U, const uint16_t* eq_V,
    size_t n, void* stream,
    size_t out_lens[2], uint8_t* out_h_blobs[2],
    float* out_kernel_ms);

void run_gpu_ans_u1_arrays_timed(
    const uint8_t* eq_U, const uint8_t* eq_V,
    size_t n, void* stream,
    size_t out_lens[2], uint8_t* out_h_blobs[2],
    float* out_kernel_ms);

void gpu_ans_u2_decode_pair_from_blobs(
    const uint8_t* h_blob_U, size_t blob_len_U,
    const uint8_t* h_blob_V, size_t blob_len_V,
    size_t n, uint16_t* d_eq_U, uint16_t* d_eq_V,
    void* stream);

void gpu_ans_u1_decode_pair_from_blobs(
    const uint8_t* h_blob_U, size_t blob_len_U,
    const uint8_t* h_blob_V, size_t blob_len_V,
    size_t n, uint8_t* d_eq_U, uint8_t* d_eq_V,
    void* stream);

#endif
