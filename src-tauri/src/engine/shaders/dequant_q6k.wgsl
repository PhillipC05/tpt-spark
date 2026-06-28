// Dequantize Q6_K blocks to f32.
//
// Q6_K layout (per 256-element block, 210 bytes):
//   bytes   0-127: ql[128] — low 4 bits of each element (2 nibbles per byte)
//   bytes 128-191: qh[64]  — high 2 bits of each element (4 × 2-bit pairs per byte)
//   bytes 192-207: scales[16] — signed int8 scale per sub-block (16 sub-blocks × 16 elements)
//   bytes 208-209: d (f16 super-scale)
//
// Element mapping for element i (0..255):
//   pass  = i / 128              (0 or 1; each pass covers 128 elements)
//   pos   = i % 128
//   l     = pos % 32             (0..31)
//   quad  = pos / 32             (0..3)
//
//   ql byte index : pass*64 + (quad % 2)*32 + l
//   use high nibble when quad >= 2
//
//   qh byte index : pass*32 + l
//   qh bit shift  : quad * 2
//
//   scale index   : pass*8 + (l/16) + quad*2   (0..15)
//
// 6-bit signed value: q6 = lo4 | (hi2 << 4),  q6_signed = q6 - 32   (range -32..31)
// Dequant: x[i] = d * scales[scale_idx] * q6_signed
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

// Interpret a byte as signed int8 (-128..127).
fn sign_extend_8(v: u32) -> i32 {
    if (v & 0x80u) != 0u { return i32(v) - 256; }
    return i32(v);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem_idx = gid.x;
    if elem_idx >= params.n_elements { return; }

    let block_idx = elem_idx / 256u;
    let lane      = elem_idx % 256u;
    let bb        = block_idx * 210u;

    let d = decode_f16(read_byte(bb + 208u), read_byte(bb + 209u));

    let pass = lane / 128u;
    let pos  = lane % 128u;
    let l    = pos % 32u;
    let quad = pos / 32u;

    // Low 4 bits from ql.
    let ql_idx = bb + pass * 64u + (quad % 2u) * 32u + l;
    let ql_raw = read_byte(ql_idx);
    let lo4    = select(ql_raw & 0x0Fu, (ql_raw >> 4u) & 0x0Fu, quad >= 2u);

    // High 2 bits from qh.
    let qh_idx = bb + 128u + pass * 32u + l;
    let hi2    = (read_byte(qh_idx) >> (quad * 2u)) & 0x3u;

    // 6-bit value, centered.
    let q6_signed = i32(lo4 | (hi2 << 4u)) - 32;

    // Signed int8 scale.
    let scale_idx   = bb + 192u + pass * 8u + l / 16u + quad * 2u;
    let scale_raw   = read_byte(scale_idx);
    let scale_i8    = sign_extend_8(scale_raw);

    out_f32[elem_idx] = d * f32(scale_i8) * f32(q6_signed);
}
