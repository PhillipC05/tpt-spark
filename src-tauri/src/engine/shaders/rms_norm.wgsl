// RMS LayerNorm: out[i] = (x[i] / rms(x)) * weight[i]
//
// One workgroup per row (sequence position or single vector).
// Two-phase parallel reduction: sum-of-squares → broadcast normalisation factor.
//
// Dispatch: (n_rows, 1, 1) with workgroup_size (256, 1, 1).
// Rows with dim > 256 are handled by each thread accumulating multiple elements.

struct Params {
    n_rows: u32,
    dim:    u32,
    eps:    f32,
}

@group(0) @binding(0) var<storage, read>       input   : array<f32>;
@group(0) @binding(1) var<storage, read>       weight  : array<f32>;
@group(0) @binding(2) var<storage, read_write> output  : array<f32>;
@group(0) @binding(3) var<uniform>             params  : Params;

var<workgroup> partial_ss: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id)         wgid : vec3<u32>,
    @builtin(local_invocation_id)  lid  : vec3<u32>,
) {
    let row   = wgid.x;
    let tid   = lid.x;
    let dim   = params.dim;
    let base  = row * dim;

    // Phase 1: each thread accumulates partial sum-of-squares.
    var ss: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if i >= dim { break; }
        let v = input[base + i];
        ss += v * v;
        i += 256u;
    }
    partial_ss[tid] = ss;
    workgroupBarrier();

    // Parallel reduction (power-of-two halving).
    var stride: u32 = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            partial_ss[tid] += partial_ss[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Phase 2: broadcast normalisation factor and apply weight.
    let mean_ss = partial_ss[0] / f32(dim);
    let norm_factor = 1.0 / sqrt(mean_ss + params.eps);

    i = tid;
    loop {
        if i >= dim { break; }
        output[base + i] = input[base + i] * norm_factor * weight[i];
        i += 256u;
    }
}
