//! Per-layer GPU KV cache buffers for the wgpu inference engine.
//!
//! Each layer gets its own K and V storage buffer so the attention shader
//! receives a correctly-sized slice (`[capacity, n_kv_heads, head_dim]` f32)
//! without having to index into a large multi-layer flat buffer.

use crate::engine::wgpu_context::GpuContext;

pub struct KvCache {
    /// One buffer per transformer layer, layout: [capacity, n_kv_heads, head_dim] f32.
    pub k_layers: Vec<wgpu::Buffer>,
    pub v_layers: Vec<wgpu::Buffer>,
    /// Number of tokens written into the cache for the current sequence.
    pub seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl KvCache {
    /// Allocate per-layer K/V buffers sized for `capacity` tokens.
    pub fn new(
        ctx: &GpuContext,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Self {
        let bytes = (capacity * n_kv_heads * head_dim * 4) as u64;
        let k_layers = (0..n_layers)
            .map(|l| ctx.create_storage_buffer(&format!("kv_k_{l}"), bytes))
            .collect();
        let v_layers = (0..n_layers)
            .map(|l| ctx.create_storage_buffer(&format!("kv_v_{l}"), bytes))
            .collect();
        Self { k_layers, v_layers, seq_len: 0, n_kv_heads, head_dim }
    }

    /// Byte offset of position `pos` within a single-layer K or V buffer.
    pub fn offset_bytes(&self, pos: usize) -> u64 {
        (pos * self.n_kv_heads * self.head_dim * 4) as u64
    }

    /// Reset the sequence length counter (does not zero GPU memory).
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Explicitly free GPU allocations. Call before dropping when immediate
    /// VRAM reclamation is needed (e.g. on model unload).
    pub fn destroy(&self) {
        for buf in &self.k_layers { buf.destroy(); }
        for buf in &self.v_layers { buf.destroy(); }
    }
}
