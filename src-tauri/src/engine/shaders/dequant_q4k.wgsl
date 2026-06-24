// Dequantize Q4_K blocks to f32.
//
// Q4_K layout (per 256-element block, 144 bytes total):
//   bytes 0-11  : scales_and_mins (12 bytes) — 8 sub-block scales + 8 sub-block mins
//                 each pair packed as 6-bit values (see llama.cpp ggml-quants.c)
//   bytes 12-143: 128 bytes of 4-bit nibbles (256 values, 2 per byte)
//
// Each block is divided into 8 sub-blocks of 32 elements.
// Sub-block k:  scale = d * scales[k],  min = dmin * mins[k]
// where d and dmin are two f16 values stored in the first 4 bytes of the 12-byte scale region.
//
// Dispatch: one thread per output element (256 per block).
//   workgroup_size = (256, 1, 1)

struct Params {
    n_elements: u32,
}

@group(0) @binding(0) var<storage, read>       quant_data : array<u32>;
@group(0) @binding(1) var<storage, read_write> out_f32    : array<f32>;
@group(0) @binding(2) var<uniform>             params     : Params;

fn read_byte_at(byte_idx: u32) -> u32 {
    return (quant_data[byte_idx / 4u] >> ((byte_idx % 4u) * 8u)) & 0xFFu;
}

fn decode_f16_bytes(lo: u32, hi: u32) -> f32 {
    let bits = lo | (hi << 8u);
    let sign  = f32((bits >> 15u) & 1u);
    let exp   = i32((bits >> 10u) & 0x1Fu);
    let mant  = f32(bits & 0x3FFu);
    if exp == 0  { return (1.0 - 2.0 * sign) * (mant / 1024.0) * (1.0 / 16384.0); }
    if exp == 31 { return select(1e38, -1e38, sign != 0.0); }
    return (1.0 - 2.0 * sign) * pow(2.0, f32(exp - 15)) * (1.0 + mant / 1024.0);
}

// Decode one 6-bit scale from the packed 12-byte region.
// llama.cpp layout: bytes 0-5 hold the low 4 bits of all 8 scales+mins interleaved,
// bytes 6-11 hold the high 2 bits.  Index 0..7 = scales, 8..15 = mins.
fn decode_6bit_scale(base: u32, idx: u32) -> f32 {
    // Low 4 bits are in bytes 0..5; two values per byte.
    let lo_byte = base + idx / 2u;
    let lo_shift = (idx % 2u) * 4u;
    let lo4 = (read_byte_at(lo_byte) >> lo_shift) & 0x0Fu;

    // High 2 bits come from bytes 6..11; each byte holds two 2-bit pairs.
    let hi_byte = base + 6u + idx / 4u;
    let hi_shift = (idx % 4u) * 2u;
    let hi2 = (read_byte_at(hi_byte) >> hi_shift) & 0x03u;

    return f32(lo4 | (hi2 << 4u));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_idx = gid.x;
    if elem_idx >= params.n_elements { return; }

    // Q4_K block = 144 bytes, 256 elements.
    let block_idx = elem_idx / 256u;
    let lane      = elem_idx % 256u;
    let block_base = block_idx * 144u;

    // d and dmin are the two f16 super-scalars (bytes 0-3 of block).
    let d    = decode_f16_bytes(read_byte_at(block_base),      read_byte_at(block_base + 1u));
    let dmin = decode_f16_bytes(read_byte_at(block_base + 2u), read_byte_at(block_base + 3u));

    // Scale bytes start at offset 4 (bytes 4-15 of block, 12 bytes).
    let scale_base = block_base + 4u;

    // Sub-block index (8 sub-blocks of 32 elements each).
    let sub = lane / 32u;

    let scale = d    * decode_6bit_scale(scale_base, sub);
    let min   = dmin * decode_6bit_scale(scale_base, sub + 8u);

    // Nibble data starts at offset 16 (bytes 16-143 of block, 128 bytes).
    // Two 4-bit values per byte; lane 0 and lane 128 share byte 0, etc.
    let nibble_base = block_base + 16u;
    let nibble_byte = nibble_base + lane / 2u;
    let raw_byte    = read_byte_at(nibble_byte);
    let nibble      = select((raw_byte >> 4u) & 0x0Fu, raw_byte & 0x0Fu, (lane % 2u) == 0u);

    out_f32[elem_idx] = scale * f32(nibble) - min;
}
