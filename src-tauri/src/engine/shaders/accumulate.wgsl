// Weighted accumulate: acc[i] += scale * src[i]
// Used by the MoE FFN to combine expert outputs.

@group(0) @binding(0) var<storage, read_write> acc : array<f32>;
@group(0) @binding(1) var<storage, read>       src : array<f32>;

struct Params {
    n     : u32,
    scale : f32,
    _pad0 : u32,
    _pad1 : u32,
}
@group(0) @binding(2) var<uniform> p : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if i >= p.n { return; }
    acc[i] += p.scale * src[i];
}
