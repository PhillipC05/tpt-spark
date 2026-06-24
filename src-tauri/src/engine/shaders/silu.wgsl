// SwiGLU activation for the feed-forward block.
//
// LLaMA feed-forward:  FFN(x) = (silu(gate) * up) * down
//   gate = x @ w_gate
//   up   = x @ w_up
//   silu(t) = t * sigmoid(t) = t / (1 + exp(-t))
//
// This shader computes: gate[i] = silu(gate[i]) * up[i]  (in-place on gate buffer)
// Then the outer GEMM multiplies by w_down.
//
// Dispatch: (ceil(n_elements/256), 1, 1) with workgroup_size (256, 1, 1).

struct Params {
    n_elements: u32,
}

@group(0) @binding(0) var<storage, read_write> gate : array<f32>;
@group(0) @binding(1) var<storage, read>       up   : array<f32>;
@group(0) @binding(2) var<uniform>             params : Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_elements { return; }

    let g = gate[i];
    let silu_g = g / (1.0 + exp(-g));
    gate[i] = silu_g * up[i];
}
