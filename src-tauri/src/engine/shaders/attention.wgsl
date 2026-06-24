// Causal multi-head (or grouped-query) attention.
//
// Three-pass approach (no subgroup ops needed for portability):
//   Pass 1 (attention_scores):  QK^T / sqrt(head_dim), causal mask applied.
//   Pass 2 (attention_softmax): two-pass numerically stable softmax (max then exp/sum/normalise).
//   Pass 3 (attention_output):  weighted sum of V vectors.
//
// GQA: if n_kv_heads < n_heads, each KV head is shared by (n_heads / n_kv_heads) Q heads.
//
// Layouts:
//   Q:   [seq_len, n_heads,    head_dim]  f32
//   K/V: [ctx_len, n_kv_heads, head_dim]  f32  (KV cache, written up to seq_len)
//   out: [seq_len, n_heads,    head_dim]  f32

// ── Pass 1: compute raw attention scores ─────────────────────────────────────

struct ScoreParams {
    seq_len:    u32,   // number of new tokens (usually 1 during decode)
    kv_len:     u32,   // tokens in KV cache = seq_offset + seq_len
    n_heads:    u32,
    n_kv_heads: u32,
    head_dim:   u32,
    seq_offset: u32,   // position of first new token
}

@group(0) @binding(0) var<storage, read>       q_buf     : array<f32>;
@group(0) @binding(1) var<storage, read>       k_buf     : array<f32>;
@group(0) @binding(2) var<storage, read_write> scores    : array<f32>; // [seq_len, n_heads, kv_len]
@group(0) @binding(3) var<uniform>             sp        : ScoreParams;

@compute @workgroup_size(64, 1, 1)
fn attention_scores(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_pos  = gid.y;   // query position [0, seq_len)
    let head   = gid.z;   // query head     [0, n_heads)
    let kv_pos = gid.x;   // key position   [0, kv_len)

    if q_pos >= sp.seq_len || head >= sp.n_heads || kv_pos >= sp.kv_len { return; }

    // Causal mask: query at absolute position (seq_offset + q_pos) cannot attend to
    // keys at positions > (seq_offset + q_pos).
    let abs_q_pos = sp.seq_offset + q_pos;
    if kv_pos > abs_q_pos {
        scores[(q_pos * sp.n_heads + head) * sp.kv_len + kv_pos] = -1e9;
        return;
    }

    let kv_head = head / (sp.n_heads / sp.n_kv_heads);

    var dot: f32 = 0.0;
    for (var d: u32 = 0u; d < sp.head_dim; d++) {
        let q_val = q_buf[(q_pos * sp.n_heads + head)    * sp.head_dim + d];
        let k_val = k_buf[(kv_pos * sp.n_kv_heads + kv_head) * sp.head_dim + d];
        dot += q_val * k_val;
    }

    let scale = 1.0 / sqrt(f32(sp.head_dim));
    scores[(q_pos * sp.n_heads + head) * sp.kv_len + kv_pos] = dot * scale;
}

// ── Pass 2: softmax over kv_len for each (q_pos, head) ───────────────────────

struct SoftmaxParams {
    seq_len: u32,
    kv_len:  u32,
    n_heads: u32,
}

@group(0) @binding(0) var<storage, read_write> sm_scores : array<f32>;
@group(0) @binding(1) var<uniform>             smp       : SoftmaxParams;

var<workgroup> wg_vals: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn attention_softmax(
    @builtin(workgroup_id)        wgid : vec3<u32>,
    @builtin(local_invocation_id) lid  : vec3<u32>,
) {
    // One workgroup per (q_pos, head) row.
    let row  = wgid.x;  // flattened (q_pos * n_heads + head)
    let tid  = lid.x;

    let row_base = row * smp.kv_len;

    // Phase 1: find row max for numerical stability.
    var row_max: f32 = -1e38;
    var i: u32 = tid;
    loop {
        if i >= smp.kv_len { break; }
        row_max = max(row_max, sm_scores[row_base + i]);
        i += 256u;
    }
    wg_vals[tid] = row_max;
    workgroupBarrier();

    var s: u32 = 128u;
    loop {
        if s == 0u { break; }
        if tid < s { wg_vals[tid] = max(wg_vals[tid], wg_vals[tid + s]); }
        workgroupBarrier();
        s >>= 1u;
    }
    let global_max = wg_vals[0];
    workgroupBarrier();

    // Phase 2: exp(x - max) and sum.
    var exp_sum: f32 = 0.0;
    i = tid;
    loop {
        if i >= smp.kv_len { break; }
        let e = exp(sm_scores[row_base + i] - global_max);
        sm_scores[row_base + i] = e;
        exp_sum += e;
        i += 256u;
    }
    wg_vals[tid] = exp_sum;
    workgroupBarrier();

    s = 128u;
    loop {
        if s == 0u { break; }
        if tid < s { wg_vals[tid] += wg_vals[tid + s]; }
        workgroupBarrier();
        s >>= 1u;
    }
    let total = wg_vals[0];
    workgroupBarrier();

    // Phase 3: normalise in-place.
    i = tid;
    loop {
        if i >= smp.kv_len { break; }
        sm_scores[row_base + i] /= total;
        i += 256u;
    }
}

// ── Pass 3: weighted sum of V ─────────────────────────────────────────────────

struct OutParams {
    seq_len:    u32,
    kv_len:     u32,
    n_heads:    u32,
    n_kv_heads: u32,
    head_dim:   u32,
}

@group(0) @binding(0) var<storage, read>       attn_weights : array<f32>; // [seq_len, n_heads, kv_len]
@group(0) @binding(1) var<storage, read>       v_buf        : array<f32>; // [kv_len, n_kv_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> attn_out     : array<f32>; // [seq_len, n_heads, head_dim]
@group(0) @binding(3) var<uniform>             op           : OutParams;

@compute @workgroup_size(64, 1, 1)
fn attention_output(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_pos = gid.y;
    let head  = gid.z;
    let d     = gid.x;

    if q_pos >= op.seq_len || head >= op.n_heads || d >= op.head_dim { return; }

    let kv_head = head / (op.n_heads / op.n_kv_heads);
    var acc: f32 = 0.0;

    for (var kv_pos: u32 = 0u; kv_pos < op.kv_len; kv_pos++) {
        let w  = attn_weights[(q_pos * op.n_heads + head) * op.kv_len + kv_pos];
        let v  = v_buf[(kv_pos * op.n_kv_heads + kv_head) * op.head_dim + d];
        acc += w * v;
    }

    attn_out[(q_pos * op.n_heads + head) * op.head_dim + d] = acc;
}
