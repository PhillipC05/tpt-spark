// Rotary Position Embedding (RoPE) applied in-place.
//
// For each head, pairs (x[2i], x[2i+1]) are rotated by angle theta_i = pos / freq_base^(2i/head_dim).
//   x'[2i]   = x[2i]   * cos - x[2i+1] * sin
//   x'[2i+1] = x[2i]   * sin + x[2i+1] * cos
//
// Applied to both Q and K after their projection.
//
// Dispatch: (n_heads * head_dim/2, seq_len, 1) with workgroup_size (64, 1, 1).

struct Params {
    n_heads:    u32,
    head_dim:   u32,   // full head dim (pairs = head_dim / 2)
    seq_offset: u32,   // position of the first token in this call (= existing seq_len)
    freq_base:  f32,
}

@group(0) @binding(0) var<storage, read_write> qk_data : array<f32>;
@group(0) @binding(1) var<uniform>             params  : Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx  = gid.x;   // which (head, pair_within_head)
    let token_idx = gid.y;   // which token in this batch

    let half_dim  = params.head_dim / 2u;
    let n_pairs   = params.n_heads * half_dim;

    if pair_idx >= n_pairs { return; }

    let head  = pair_idx / half_dim;
    let pair  = pair_idx % half_dim;
    let pos   = params.seq_offset + token_idx;

    let theta = f32(pos) / pow(params.freq_base, f32(pair * 2u) / f32(params.head_dim));
    let c = cos(theta);
    let s = sin(theta);

    // Index into the flat [n_tokens, n_heads, head_dim] layout.
    let base = (token_idx * params.n_heads + head) * params.head_dim;
    let i0 = base + pair * 2u;
    let i1 = i0 + 1u;

    let x0 = qk_data[i0];
    let x1 = qk_data[i1];

    qk_data[i0] = x0 * c - x1 * s;
    qk_data[i1] = x0 * s + x1 * c;
}
