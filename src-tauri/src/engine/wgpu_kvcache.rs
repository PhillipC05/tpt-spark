//! Per-layer GPU KV cache buffers for the wgpu inference engine.
//!
//! Each layer gets its own K and V storage buffer so the attention shader
//! receives a correctly-sized slice (`[capacity, n_kv_heads, head_dim]` f32)
//! without having to index into a large multi-layer flat buffer.
//!
//! Multi-turn prefix caching: `cached_tokens` records the token IDs whose K/V
//! values are currently resident in the GPU buffers.  On each new inference
//! call, the longest common prefix between `cached_tokens` and the new prompt
//! is found; prefill is skipped for those positions because the GPU data is
//! still valid.

use crate::engine::wgpu_context::GpuContext;

pub struct KvCache {
    /// One buffer per transformer layer, layout: [capacity, n_kv_heads, head_dim] f32.
    pub k_layers: Vec<wgpu::Buffer>,
    pub v_layers: Vec<wgpu::Buffer>,
    /// Number of tokens written into the cache for the current sequence.
    pub seq_len: usize,
    /// Token IDs whose K/V projections are resident in the GPU buffers at
    /// positions `[0..seq_len)`.  Empty when the cache is cold or has been reset.
    pub cached_tokens: Vec<u32>,
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
        Self { k_layers, v_layers, seq_len: 0, cached_tokens: Vec::new(), n_kv_heads, head_dim }
    }

    /// Byte offset of position `pos` within a single-layer K or V buffer.
    pub fn offset_bytes(&self, pos: usize) -> u64 {
        (pos * self.n_kv_heads * self.head_dim * 4) as u64
    }

    /// Invalidate the cache: reset the sequence length counter and clear the
    /// cached token record.  Does not zero GPU memory (stale entries are never
    /// read because `seq_len` gates attention).
    pub fn reset(&mut self) {
        self.seq_len = 0;
        self.cached_tokens.clear();
    }

    /// Returns the number of leading tokens shared between `cached_tokens` and
    /// `new_tokens`.  A result of 0 means no prefix can be reused and the cache
    /// must be fully reset before the new sequence begins.
    pub fn common_prefix_len(&self, new_tokens: &[u32]) -> usize {
        self.cached_tokens
            .iter()
            .zip(new_tokens)
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Explicitly free GPU allocations. Call before dropping when immediate
    /// VRAM reclamation is needed (e.g. on model unload).
    pub fn destroy(&self) {
        for buf in &self.k_layers { buf.destroy(); }
        for buf in &self.v_layers { buf.destroy(); }
    }
}

// ── Unit tests (no GPU required) ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_with_tokens(tokens: Vec<u32>) -> KvCache {
        KvCache {
            k_layers: vec![],
            v_layers: vec![],
            seq_len: tokens.len(),
            cached_tokens: tokens,
            n_kv_heads: 8,
            head_dim: 64,
        }
    }

    #[test]
    fn empty_cache_returns_zero() {
        let c = cache_with_tokens(vec![]);
        assert_eq!(c.common_prefix_len(&[1, 2, 3]), 0);
    }

    #[test]
    fn full_prefix_match() {
        let c = cache_with_tokens(vec![1, 2, 3]);
        assert_eq!(c.common_prefix_len(&[1, 2, 3, 4, 5]), 3);
    }

    #[test]
    fn diverges_at_second_token() {
        let c = cache_with_tokens(vec![1, 99, 3]);
        assert_eq!(c.common_prefix_len(&[1, 2, 3]), 1);
    }

    #[test]
    fn no_match_at_all() {
        let c = cache_with_tokens(vec![5, 6, 7]);
        assert_eq!(c.common_prefix_len(&[1, 2, 3]), 0);
    }

    #[test]
    fn new_tokens_shorter_than_cached() {
        let c = cache_with_tokens(vec![1, 2, 3, 4, 5]);
        assert_eq!(c.common_prefix_len(&[1, 2, 3]), 3);
    }

    #[test]
    fn reset_clears_tokens() {
        let mut c = cache_with_tokens(vec![1, 2, 3]);
        c.reset();
        assert_eq!(c.seq_len, 0);
        assert!(c.cached_tokens.is_empty());
    }
}
