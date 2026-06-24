// Tiled GEMM:  C = A * B^T
//
// A: [M, K]  (row-major f32)
// B: [N, K]  (row-major f32 — transposed before multiply, stored as N rows of K cols)
// C: [M, N]  (row-major f32 output)
//
// Workgroup: 16×16 threads.
// Dispatch: (ceil(N/16), ceil(M/16), 1)
//
// Push constants: M, N, K as u32.

struct Dims {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read>       mat_a  : array<f32>;  // [M * K]
@group(0) @binding(1) var<storage, read>       mat_b  : array<f32>;  // [N * K]  (B already transposed)
@group(0) @binding(2) var<storage, read_write> mat_c  : array<f32>;  // [M * N]
@group(0) @binding(3) var<uniform>             dims   : Dims;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id)   gid  : vec3<u32>,
    @builtin(local_invocation_id)    lid  : vec3<u32>,
    @builtin(workgroup_id)           wgid : vec3<u32>,
) {
    let row = gid.y;  // output row index (M dimension)
    let col = gid.x;  // output col index (N dimension)

    var acc: f32 = 0.0;

    let n_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < n_tiles; t++) {
        // Load tile from A: row=row, col=(t*TILE + lid.x)
        let a_col = t * TILE + lid.x;
        if row < dims.M && a_col < dims.K {
            tile_a[lid.y][lid.x] = mat_a[row * dims.K + a_col];
        } else {
            tile_a[lid.y][lid.x] = 0.0;
        }

        // Load tile from B (transposed): row=col, col=(t*TILE + lid.y)
        let b_col = t * TILE + lid.y;
        if col < dims.N && b_col < dims.K {
            tile_b[lid.y][lid.x] = mat_b[col * dims.K + b_col];
        } else {
            tile_b[lid.y][lid.x] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE; k++) {
            acc += tile_a[lid.y][k] * tile_b[k][lid.x];
        }

        workgroupBarrier();
    }

    if row < dims.M && col < dims.N {
        mat_c[row * dims.N + col] = acc;
    }
}
