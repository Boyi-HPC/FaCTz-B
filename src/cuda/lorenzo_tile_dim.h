#pragma once
// Single source of truth for the 2D-Lorenzo tile size used by the
// tile-uniform-eb path. MUST be identical across:
//   - tile_uniform_eb.cuh           (eb uniformization tiling)
//   - lproto_c.cuhip.inl            (compress kernel TileDim + launch Tile2D/Block2D)
//   - lproto_x.cuhip.inl            (decompress kernel TileDim + launch Tile2D/Block2D)
// Changing this one value re-tiles all three consistently.
// Valid values: 8, 16, 32 (power of 2; block has TileDim*TileDim threads,
// so 32 → 1024 threads/block which is the hardware max).
#ifndef LORENZO_TILE_DIM
#define LORENZO_TILE_DIM 32
#endif
