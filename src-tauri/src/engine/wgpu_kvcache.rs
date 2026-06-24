//! Persistent GPU-side KV cache buffers, grown on demand as the sequence lengthens.

use crate::engine::wgpu_context::GpuContext;

pub struct KvCache {
    /// Flattened [layers, seq_len, n_kv_heads, head_dim] f32 buffer for keys.
    pub k: wgpu::Buffer,
    /// Same layout for values.
    pub v: wgpu::Buffer,
    /// Number of tokens currently in the cache.
    pub seq_len: usize,
    /// Allocated capacity in tokens.
    capacity: usize,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl KvCache {
    /// Allocate a KV cache for `capacity` tokens.
    /// Each entry is `n_layers * n_kv_heads * head_dim` f32 values (4 bytes each).
    pub fn new(ctx: &GpuContext, n_layers: usize, n_kv_heads: usize, head_dim: usize, capacity: usize) -> Self {
        let bytes = (n_layers * capacity * n_kv_heads * head_dim * 4) as u64;
        let k = ctx.create_storage_buffer("kv_cache_k", bytes);
        let v = ctx.create_storage_buffer("kv_cache_v", bytes);
        Self { k, v, seq_len: 0, capacity, n_layers, n_kv_heads, head_dim }
    }

    /// Byte offset of layer `l`, position `pos` within the flattened buffer.
    pub fn offset_bytes(&self, layer: usize, pos: usize) -> u64 {
        ((layer * self.capacity * self.n_kv_heads * self.head_dim + pos * self.n_kv_heads * self.head_dim) * 4) as u64
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}
