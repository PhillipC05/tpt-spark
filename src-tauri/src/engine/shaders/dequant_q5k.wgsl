// Dequantize Q5_K blocks to f32.
//
// Q5_K layout (per 256-element block, 176 bytes):
//   bytes   0-1 : d    (f16 super-scale)
//   bytes   2-3 : dmin (f16 super-min)
//   bytes  4-15 : scales[12] — same get_scale_min_k4 packing as Q4_K
//   bytes 16-47 : qh[32]     — 256 high bits (bit i of qh[i/8])
//   bytes 48-175: qs[128]    — 256 × 4-bit low nibbles, same split as Q4_K
//
// 5-bit value: q5 = lo4 | (high_bit << 4)  → range 0..31
// Sub-block:   same 8-group / 32-element split as Q4_K (scale_idx formula identical).
// Dequant:     x[i] = d * scale_6bit * q5 - dmin * min_6bit
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

fn get_scale_k4(base: u32, j: u32) -> f32 {
    var d: u32;
    if j < 4u {
        d = read_byte(base + j) & 63u;
    } else {
        let k = j - 4u;
        d = (read_byte(base + 8u + k) & 0x0Fu) | ((read_byte(base + k) >> 6u) << 4u);
    }
    return f32(d);
}

fn get_min_k4(base: u32, j: u32) -> f32 {
    var m: u32;
    if j < 4u {
        m = read_byte(base + j + 4u) & 63u;
    } else {
        let k = j - 4u;
        m = (read_byte(base + 8u + k) >> 4u) | ((read_byte(base + 4u + k) >> 6u) << 4u);
    }
    return f32(m);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_idx = gid.x;
    if elem_idx >= params.n_elements { return; }

    let block_idx = elem_idx / 256u;
    let lane      = elem_idx % 256u;
    let bb        = block_idx * 176u;

    let d    = decode_f16(read_byte(bb),      read_byte(bb + 1u));
    let dmin = decode_f16(read_byte(bb + 2u), read_byte(bb + 3u));

    let scale_base  = bb + 4u;   // 12 bytes
    let qh_base     = bb + 16u;  // 32 bytes
    let nibble_base = bb + 48u;  // 128 bytes

    // Same sub-block / scale index as Q4_K.
    var j: u32;
    if lane < 128u {
        j = lane / 32u;
    } else {
        j = (lane - 128u) / 32u;
    }
    let scale_idx = select(2u * j, 2u * j + 1u, lane >= 128u);

    let scale = d    * get_scale_k4(scale_base, scale_idx);
    let min   = dmin * get_min_k4  (scale_base, scale_idx);

    // High bit from qh: bit `lane` of the 256-bit qh array.
    let high_bit = (read_byte(qh_base + lane / 8u) >> (lane % 8u)) & 1u;

    // Low nibble: same Q4_K split layout (elements 0..127 → low, 128..255 → high).
    var qs_idx: u32;
    var use_hi: bool;
    if lane < 128u {
        qs_idx = lane;
        use_hi = false;
    } else {
        qs_idx = lane - 128u;
        use_hi = true;
    }
    let raw = read_byte(nibble_base + qs_idx);
    let lo4 = select(raw & 0x0Fu, (raw >> 4u) & 0x0Fu, use_hi);

    let q5 = lo4 | (high_bit << 4u);   // 0..31
    out_f32[elem_idx] = scale * f32(q5) - min;
}
