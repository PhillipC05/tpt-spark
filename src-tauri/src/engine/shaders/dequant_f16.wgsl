// Dequantize F16 tensors to f32.
//
// Layout: 2 bytes per element, little-endian IEEE 754 half precision.
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

    let byte_base = elem_idx * 2u;
    out_f32[elem_idx] = decode_f16(read_byte(byte_base), read_byte(byte_base + 1u));
}
