//! GGUF mmap reader and staged VRAM upload for the wgpu engine.
//!
//! Design:
//! - The GGUF file is memory-mapped (no RAM allocation beyond kernel page tables).
//! - `gguf_file::Content::read()` parses headers from a Cursor over the mmap slice.
//! - After parsing, cursor.position() gives the data region base offset.
//! - Each tensor slice is uploaded to a wgpu STORAGE buffer via queue.write_buffer().
//! - The mmap is dropped after submission — VRAM holds the only copy.
//!
//! Note: wgpu's staging belt allocates transient RAM during upload.  For a 7B model
//! expect a ~200-400 MB peak that is released once the GPU copy completes.

use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use memmap2::Mmap;
use tracing::info;
use wgpu::util::DeviceExt;

use crate::engine::wgpu_context::GpuContext;

/// Metadata extracted from the GGUF header, needed to configure the inference pipeline.
pub struct GgufMeta {
    pub architecture: String,
    pub context_length: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    /// FFN intermediate (hidden) dimension, e.g. 11008 for LLaMA-1 7B, 14336 for LLaMA-3 8B.
    pub ffn_hidden_size: usize,
}

/// Result of loading a model from disk into VRAM.
pub struct LoadedWeights {
    pub meta: GgufMeta,
    /// Tensor name → GPU storage buffer containing the raw quantized bytes.
    pub buffers: HashMap<String, wgpu::Buffer>,
    pub total_bytes: u64,
}

impl LoadedWeights {
    /// Explicitly free all GPU buffer allocations. Call before drop when immediate
    /// VRAM reclamation is required (e.g. on model unload).
    pub fn destroy(&self) {
        for buf in self.buffers.values() {
            buf.destroy();
        }
    }
}

pub fn load_gguf_to_vram(path: &Path, ctx: &GpuContext) -> Result<LoadedWeights> {
    let file = File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;

    // Safety: the file is not modified while the mmap lives.
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("mmap {}", path.display()))?;

    info!(
        "mmap GGUF: {} ({:.1} GB)",
        path.display(),
        mmap.len() as f64 / 1_073_741_824.0
    );

    let mut cursor = Cursor::new(&mmap[..]);
    let content = gguf_file::Content::read(&mut cursor)
        .map_err(|e| anyhow::anyhow!("GGUF parse error: {}", e))?;

    let data_base = cursor.position() as usize;

    let meta = extract_meta(&content.metadata)
        .context("extracting GGUF metadata")?;

    info!(
        "GGUF meta: arch={} layers={} heads={} kv_heads={} ctx_len={}",
        meta.architecture, meta.n_layers, meta.n_heads, meta.n_kv_heads, meta.context_length
    );

    let mut buffers = HashMap::new();
    let mut total_bytes: u64 = 0;

    for (name, tensor_info) in &content.tensor_infos {
        let offset = data_base + tensor_info.offset as usize;
        let byte_len = tensor_info.shape.elem_count() * tensor_info.ggml_dtype.type_size()
            / tensor_info.ggml_dtype.block_size();

        if offset + byte_len > mmap.len() {
            anyhow::bail!("tensor '{}' extends past end of file", name);
        }

        let slice = &mmap[offset..offset + byte_len];

        let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: slice,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        total_bytes += byte_len as u64;
        buffers.insert(name.clone(), buf);
    }

    // Flush all staging-belt copies to the GPU before mmap is dropped.
    ctx.queue.submit([]);

    info!(
        "Uploaded {} tensors ({:.1} GB) to VRAM",
        buffers.len(),
        total_bytes as f64 / 1_073_741_824.0
    );

    // mmap drops here — VRAM is now the sole owner of the weight data.
    Ok(LoadedWeights { meta, buffers, total_bytes })
}

// ── Metadata helpers ──────────────────────────────────────────────────────────

fn meta_str<'a>(
    meta: &'a HashMap<String, gguf_file::Value>,
    key: &str,
) -> Option<&'a str> {
    meta.get(key).and_then(|v| {
        if let gguf_file::Value::String(s) = v {
            Some(s.as_str())
        } else {
            None
        }
    })
}

fn meta_u64(meta: &HashMap<String, gguf_file::Value>, key: &str) -> Option<u64> {
    meta.get(key).and_then(|v| match v {
        gguf_file::Value::U32(n) => Some(*n as u64),
        gguf_file::Value::U64(n) => Some(*n),
        _ => None,
    })
}

fn meta_f32(meta: &HashMap<String, gguf_file::Value>, key: &str) -> Option<f32> {
    meta.get(key).and_then(|v| match v {
        gguf_file::Value::F32(f) => Some(*f),
        gguf_file::Value::F64(f) => Some(*f as f32),
        _ => None,
    })
}

fn extract_meta(meta: &HashMap<String, gguf_file::Value>) -> Result<GgufMeta> {
    let architecture = meta_str(meta, "general.architecture")
        .unwrap_or("llama")
        .to_string();

    if !matches!(architecture.as_str(), "llama" | "mistral") {
        anyhow::bail!(
            "Architecture '{}' is not supported by the wgpu engine. Supported: llama, mistral.",
            architecture
        );
    }

    let a = &architecture;

    let context_length = meta_u64(meta, &format!("{a}.context_length"))
        .unwrap_or(4096) as usize;
    let n_layers = meta_u64(meta, &format!("{a}.block_count"))
        .unwrap_or(32) as usize;
    let n_heads = meta_u64(meta, &format!("{a}.attention.head_count"))
        .unwrap_or(32) as usize;
    let n_kv_heads = meta_u64(meta, &format!("{a}.attention.head_count_kv"))
        .unwrap_or(n_heads as u64) as usize;
    let embed_dim = meta_u64(meta, &format!("{a}.embedding_length"))
        .unwrap_or(4096) as usize;
    let vocab_size = meta_u64(meta, "tokenizer.ggml.tokens")
        .or_else(|| meta_u64(meta, &format!("{a}.vocab_size")))
        .unwrap_or(32000) as usize;
    let rope_freq_base = meta_f32(meta, &format!("{a}.rope.freq_base"))
        .unwrap_or(10000.0);

    let head_dim = embed_dim / n_heads;

    // Read the true FFN intermediate size from metadata.
    // Falls back to a LLaMA-style SwiGLU estimate (rounds up to 256-multiple) when absent.
    let ffn_hidden_size = meta_u64(meta, &format!("{a}.feed_forward_length"))
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            let est = (embed_dim * 8).div_ceil(3);
            est.div_ceil(256) * 256
        });

    Ok(GgufMeta {
        architecture,
        context_length,
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size,
        rope_freq_base,
        ffn_hidden_size,
    })
}
