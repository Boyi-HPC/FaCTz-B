#ifndef CUCPSZ_2D_FORMAT_HPP
#define CUCPSZ_2D_FORMAT_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>

enum class CucpszCodec : uint32_t {
    Huffman = 0,
    Ans = 1,
};

// Binary layout shared by the standalone compressor and decompressor.
struct CucpszHeader {
    char magic[8];
    uint64_t r1, r2;
    float max_pwr_eb;
    uint32_t ot_count_U, ot_count_V;
    uint32_t zeroeb_count_U, zeroeb_count_V;
    // This occupies four bytes that were ABI padding in legacy headers.
    // Old files zero-initialized the padding, so they remain Huffman files.
    uint32_t codec;
    uint64_t hf_blob_len[4];
    uint64_t land_bitpack_bytes;
    uint32_t lorenzo_variant;
    uint32_t lorenzo_tile_dim;
};

static_assert(sizeof(CucpszHeader) == 96, "Unexpected .cucpsz header layout");

inline bool cucpsz_header_has_valid_magic(const CucpszHeader& header)
{
    return std::memcmp(header.magic, "CUCPSZ\0\0", 8) == 0;
}

inline const char* cucpsz_codec_name(CucpszCodec codec)
{
    return codec == CucpszCodec::Ans ? "ans" : "hf";
}

inline bool parse_cucpsz_codec(const char* text, CucpszCodec& codec)
{
    if (std::strcmp(text, "hf") == 0) {
        codec = CucpszCodec::Huffman;
        return true;
    }
    if (std::strcmp(text, "ans") == 0) {
        codec = CucpszCodec::Ans;
        return true;
    }
    return false;
}

inline bool cucpsz_header_has_supported_codec(const CucpszHeader& header)
{
    return header.codec == static_cast<uint32_t>(CucpszCodec::Huffman) ||
           header.codec == static_cast<uint32_t>(CucpszCodec::Ans);
}

inline size_t bitmask_word_bytes(size_t n)
{
    return ((n + 31) / 32) * sizeof(uint32_t);
}

#endif
