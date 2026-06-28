// Dequantize Q2_K blocks to f32.
//
// Q2_K layout (per 256-element block, 84 bytes):
//   bytes  0-15: scales[16] — packed 4-bit pairs per sub-block (low nibble = scale, high = min)
//   bytes 16-79: qs[64]     — 2-bit values, 4 per byte (LSB first)
//   bytes 80-81: d    (f16 super-scale)
//   bytes 82-83: dmin (f16 super-min)
//
// Sub-block: 16 elements per sub-block (256 / 16 = 16 sub-blocks)
//   sub   = i / 16
//   sc    = scales[sub] & 0x0F   (unsigned 4-bit scale multiplier)
//   mn    = scales[sub] >> 4     (unsigned 4-bit min multiplier)
//   q2    = (qs[i/4] >> (2*(i%4))) & 0x3   (0..3)
//   x[i]  = d * sc * q2 - dmin * mn
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
    let bb        = block_idx * 84u;

    let d    = decode_f16(read_byte(bb + 80u), read_byte(bb + 81u));
    let dmin = decode_f16(read_byte(bb + 82u), read_byte(bb + 83u));

    let sub = lane / 16u;
    let sc_byte = read_byte(bb + sub);
    let sc  = f32(sc_byte & 0x0Fu);
    let mn  = f32(sc_byte >> 4u);

    // 2-bit quantized value: 4 per byte, LSB first.
    let qs_byte = read_byte(bb + 16u + lane / 4u);
    let q2      = f32((qs_byte >> ((lane % 4u) * 2u)) & 0x3u);

    out_f32[elem_idx] = d * sc * q2 - dmin * mn;
}
