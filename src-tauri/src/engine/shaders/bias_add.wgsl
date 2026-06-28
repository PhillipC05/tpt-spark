// In-place bias addition: buf[i] += bias[i]
//
// Used to apply attention projection biases (attn_q/k/v/output.bias) after each GEMM.
// Dispatch: (ceil(n / 256), 1, 1) with workgroup_size (256, 1, 1).

struct Params { n: u32, _pad: array<u32, 3> }

@group(0) @binding(0) var<storage, read_write> buf  : array<f32>;
@group(0) @binding(1) var<storage, read>       bias : array<f32>;
@group(0) @binding(2) var<uniform>             p    : Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= p.n { return; }
    buf[gid.x] += bias[gid.x];
}
