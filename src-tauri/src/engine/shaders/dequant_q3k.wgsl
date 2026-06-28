// Dequantize Q3_K blocks to f32.
//
// Q3_K layout (per 256-element block, 110 bytes):
//   bytes  0-31:  hmask[32]  — high bits: bit i selects bit 2 of q3[i]
//   bytes 32-95:  qs[64]     — low 2-bit values, 4 per byte (LSB first)
//   bytes 96-107: scales[12] — 3-bit signed scales packed 8-per-3-bytes
//   bytes 108-109: d (f16)
//
// Extracting scale for sub-block k (i / 16, range 0..15):
//   The 12-byte scales region holds 16 × 3-bit signed values (48 bits used out of 96).
//   Layout mirrors llama.cpp: first 8 sub-blocks occupy low 3 bits of bytes 0-7,
//   next 8 sub-blocks occupy bits 4-6 of bytes 0-7 (shifted right by 4).
//     if k < 8:  raw3 = read_byte(scales_base + k)        & 0x07
//     else:      raw3 = (read_byte(scales_base + k - 8) >> 4) & 0x07
//   Signed: scale_signed = raw3 - 4   (range -4..3, note: llama.cpp center is -4)
//   (llama.cpp: d * (scale - 4) * q3signed where q3signed = q3 - 4)
//
// 3-bit value: q2 = (qs[i/4] >> (2*(i%4))) & 0x3
//              hb  = (hmask[i/8] >> (i%8)) & 0x1
//              q3  = q2 | (hb << 2)        → 0..7
//   q3_signed = q3 - 4                    → -4..3
//
// Dequant: x[i] = d * (scale_k - 4) * (q3 - 4)
//   where scale_k is the raw 3-bit value (unsigned 0..7).
//
// Dispatch: one thread per output element.
//   workgroup_size = (256, 1, 1)

struct Params { n_elements: u32 }

@group(0) @binding(0) var<storage, read>       quant_data : array<u32>;
@group(0) @binding(1) var<storage, read_write> out_f32    : array<f32>;
@group(0) @binding(2) var<uniform>             params     : Params;

fn read_byte(byte_idx: u32) -> u32 {
    return (quant_data[byte_idx / 4u] >> ((byte_idx % 4u) * 8u)) & 0xFFu;
}

fn decode_f16(lo: u32, hi: u32) -> f32 {
    let bits = lo | (hi << 8u);
    let sign = f32((bits >> 15u) & 1u);
    let exp  = i32((bits >> 10u) & 0x1Fu);
    let mant = f32(bits & 0x3FFu);
    if exp == 0  { return (1.0 - 2.0 * sign) * (mant / 1024.0) * (1.0 / 16384.0); }
    if exp == 31 {
        if mant == 0.0 { return select(1e38, -1e38, sign != 0.0); }
        return 0.0;
    }
    return (1.0 - 2.0 * sign) * pow(2.0, f32(exp - 15)) * (1.0 + mant / 1024.0);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_idx = gid.x;
    if elem_idx >= params.n_elements { return; }

    let block_idx = elem_idx / 256u;
    let lane      = elem_idx % 256u;
    let bb        = block_idx * 110u;

    let d = decode_f16(read_byte(bb + 108u), read_byte(bb + 109u));

    // 3-bit signed scale for sub-block k.
    let k           = lane / 16u;
    let scales_base = bb + 96u;
    var raw3: u32;
    if k < 8u {
        raw3 = read_byte(scales_base + k) & 0x07u;
    } else {
        raw3 = (read_byte(scales_base + k - 8u) >> 4u) & 0x07u;
    }
    let scale_signed = f32(i32(raw3) - 4);

    // Low 2 bits from qs.
    let qs_byte = read_byte(bb + 32u + lane / 4u);
    let q2      = (qs_byte >> ((lane % 4u) * 2u)) & 0x3u;

    // High bit from hmask.
    let hb = (read_byte(bb + lane / 8u) >> (lane % 8u)) & 0x1u;

    let q3        = q2 | (hb << 2u);        // 0..7
    let q3_signed = f32(i32(q3) - 4);       // -4..3

    out_f32[elem_idx] = d * scale_signed * q3_signed;
}
