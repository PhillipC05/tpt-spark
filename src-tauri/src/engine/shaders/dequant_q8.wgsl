// Dequantize Q8_0 blocks to f32.
//
// Q8_0 layout (per 32-element block):
//   bytes 0-1  : f16 scale
//   bytes 2-33 : 32 x i8 quantized values
// Block size on disk: 34 bytes  (QK8_0 = 32)
//
// Dispatch: one thread per output element.
//   total_elements = n_blocks * 32
//   workgroup_size = (256, 1, 1)  →  dispatch((total_elements + 255) / 256, 1, 1)

struct Params {
    n_elements: u32,
}

@group(0) @binding(0) var<storage, read>       quant_data : array<u32>;   // raw bytes packed as u32
@group(0) @binding(1) var<storage, read_write> out_f32    : array<f32>;
@group(0) @binding(2) var<uniform>             params     : Params;

// Read a single byte from the u32-packed buffer.
fn read_byte(base_u32: u32, byte_idx: u32) -> i32 {
    let word = quant_data[base_u32 + byte_idx / 4u];
    let shift = (byte_idx % 4u) * 8u;
    return i32((word >> shift) & 0xFFu);
}

// Sign-extend an 8-bit value stored as u32 to i32.
fn sign_extend_8(v: u32) -> i32 {
    if (v & 0x80u) != 0u {
        return i32(v) - 256;
    }
    return i32(v);
}

// Decode f16 from two bytes (little-endian) to f32.
fn decode_f16(lo: u32, hi: u32) -> f32 {
    let bits = lo | (hi << 8u);
    let sign  = f32((bits >> 15u) & 1u);
    let exp   = i32((bits >> 10u) & 0x1Fu);
    let mant  = f32(bits & 0x3FFu);
    if exp == 0 {
        return (1.0 - 2.0 * sign) * (mant / 1024.0) * (1.0 / 16384.0);
    }
    if exp == 31 {
        if mant == 0.0 { return select(f32(1e38), f32(-1e38), sign != 0.0); }
        return 0.0; // NaN → 0
    }
    return (1.0 - 2.0 * sign) * pow(2.0, f32(exp - 15)) * (1.0 + mant / 1024.0);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_idx = gid.x;
    if elem_idx >= params.n_elements { return; }

    let block_idx = elem_idx / 32u;
    let lane      = elem_idx % 32u;

    // Each block is 34 bytes: 2 (f16 scale) + 32 (i8 values).
    let block_byte_base = block_idx * 34u;

    // Decode f16 scale (bytes 0-1 of block).
    let lo = (quant_data[(block_byte_base)     / 4u] >> (((block_byte_base)     % 4u) * 8u)) & 0xFFu;
    let hi = (quant_data[(block_byte_base + 1u) / 4u] >> (((block_byte_base + 1u) % 4u) * 8u)) & 0xFFu;
    let scale = decode_f16(lo, hi);

    // Decode i8 quantized value (bytes 2..33 of block).
    let val_byte = block_byte_base + 2u + lane;
    let raw = (quant_data[val_byte / 4u] >> ((val_byte % 4u) * 8u)) & 0xFFu;
    let ival = sign_extend_8(raw);

    out_f32[elem_idx] = scale * f32(ival);
}
