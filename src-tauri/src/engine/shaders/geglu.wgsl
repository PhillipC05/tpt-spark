// GEGLU activation for Gemma feed-forward block.
//
// Gemma FFN:  FFN(x) = (gelu(gate) * up) * down
//   gelu(t) = t * 0.5 * (1 + tanh(sqrt(2/pi) * (t + 0.044715 * t^3)))
//
// This shader computes: gate[i] = gelu(gate[i]) * up[i]  (in-place on gate buffer)
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
    let gelu_g = g * 0.5 * (1.0 + tanh(0.7978845608 * (g + 0.044715 * g * g * g)));
    gate[i] = gelu_g * up[i];
}
