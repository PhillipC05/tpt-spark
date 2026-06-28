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
use candle_core::quantized::{gguf_file, GgmlDType};
use memmap2::Mmap;
use tracing::info;
use wgpu::util::DeviceExt;

use crate::engine::wgpu_context::GpuContext;

/// FFN gating activation used by this architecture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FfnActivation {
    SwiGLU,  // LLaMA, Mistral: silu(gate) * up
    GEGLU,   // Gemma: gelu(gate) * up
}

/// A GPU-resident tensor: raw quantized bytes in a storage buffer, plus the dtype needed
/// to select the correct dequantization shader at inference time.
pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    pub dtype: GgmlDType,
}

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
    pub ffn_activation: FfnActivation,
    /// True when the LM head weight is tied to the token embedding table.
    pub tied_embeddings: bool,
    /// Total number of FFN experts (0 = dense model, no MoE).
    pub n_experts: usize,
    /// Number of experts selected per token (top-K routing).  0 when n_experts == 0.
    pub n_experts_used: usize,
    /// FFN hidden size per expert.  Usually equals ffn_hidden_size but some models differ.
    pub expert_ffn_hidden_size: usize,
    /// Epsilon for RMS normalisation layers (model-specific; default 1e-5).
    pub rms_norm_eps: f32,
    /// Linear RoPE position scale factor (1.0 = disabled).
    /// For "linear" rope.scaling: positions are divided by this factor.
    /// For "yarn"/"ntk" scaling: `rope_freq_base` is already adjusted; this stays 1.0.
    pub rope_scale_factor: f32,
}

/// Result of loading a model from disk into VRAM.
pub struct LoadedWeights {
    pub meta: GgufMeta,
    /// Tensor name → GPU tensor (raw quantized bytes + dtype for shader dispatch).
    pub buffers: HashMap<String, GpuTensor>,
    pub total_bytes: u64,
}

impl LoadedWeights {
    /// Explicitly free all GPU buffer allocations. Call before drop when immediate
    /// VRAM reclamation is required (e.g. on model unload).
    pub fn destroy(&self) {
        for t in self.buffers.values() {
            t.buffer.destroy();
        }
    }
}

pub fn load_gguf_to_vram(
    path: &Path,
    ctx: &GpuContext,
    on_progress: Option<&(dyn Fn(u32, u32) + Send + Sync)>,
) -> Result<LoadedWeights> {
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
    let n_experts = meta.n_experts;
    let tensor_total = content.tensor_infos.len() as u32;
    let mut tensors_done: u32 = 0;

    for (name, tensor_info) in &content.tensor_infos {
        let offset = data_base + tensor_info.offset as usize;
        let byte_len = tensor_info.shape.elem_count() * tensor_info.ggml_dtype.type_size()
            / tensor_info.ggml_dtype.block_size();

        if offset + byte_len > mmap.len() {
            anyhow::bail!("tensor '{}' extends past end of file", name);
        }
        if byte_len == 0 {
            continue; // skip zero-size tensors
        }

        let max_buf = ctx.device.limits().max_buffer_size as usize;
        if byte_len > max_buf {
            anyhow::bail!(
                "tensor '{}' is {:.1} MB but the GPU max buffer size is {:.1} MB",
                name,
                byte_len as f64 / 1_048_576.0,
                max_buf as f64 / 1_048_576.0,
            );
        }

        let slice = &mmap[offset..offset + byte_len];

        let dtype = tensor_info.ggml_dtype;

        // Stacked expert tensors (e.g. ffn_gate_exps.weight, shape [n_experts, ffn_h, emb]).
        // Split into per-expert buffers named "blk.N.ffn_gate_exps.weight[E]" so the
        // inference loop can bind individual experts without offset arithmetic.
        if n_experts > 1 && is_expert_weight(name) {
            if byte_len % n_experts != 0 {
                anyhow::bail!(
                    "expert tensor '{}' byte size {} is not divisible by n_experts {}",
                    name, byte_len, n_experts
                );
            }
            let per_expert = byte_len / n_experts;
            for e in 0..n_experts {
                let expert_slice = &mmap[offset + e * per_expert..offset + (e + 1) * per_expert];
                let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{name}[{e}]")),
                    contents: expert_slice,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                buffers.insert(format!("{name}[{e}]"), GpuTensor { buffer: buf, dtype });
            }
            total_bytes += byte_len as u64;
            continue;
        }

        let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: slice,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        total_bytes += byte_len as u64;
        buffers.insert(name.clone(), GpuTensor { buffer: buf, dtype });

        tensors_done += 1;
        if let Some(cb) = on_progress {
            cb(tensors_done, tensor_total);
        }
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

/// Returns true for stacked expert weight tensors that must be split per-expert.
/// Matches `ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`.
fn is_expert_weight(name: &str) -> bool {
    name.contains("_exps.weight")
}

// ── Architecture config table ─────────────────────────────────────────────────

struct ArchCfg {
    ffn_activation: FfnActivation,
    tied_embeddings: bool,
}

impl ArchCfg {
    const fn swiglu() -> Self { Self { ffn_activation: FfnActivation::SwiGLU, tied_embeddings: false } }
    const fn geglu_tied() -> Self { Self { ffn_activation: FfnActivation::GEGLU, tied_embeddings: true } }
}

/// Architectures known to use standard GGUF tensor naming (`blk.N.*`).
/// Models outside this list are rejected at load time with a clear message.
static SUPPORTED_ARCHS: &[&str] = &[
    // LLaMA family (Meta) — includes Llama 4 Scout/Maverick
    "llama", "llama4",
    // Mistral / Mixtral (Mistral AI)
    "mistral",
    // Qwen family (Alibaba) — Qwen2, Qwen2.5, Qwen3, Qwen3.5, Qwen3.6
    "qwen2", "qwen2_5", "qwen3",
    // Gemma family (Google) — GEGLU activation, tied LM head
    "gemma", "gemma2", "gemma3", "gemma4",
    // Phi family (Microsoft) — Phi-3, Phi-3.5, Phi-4
    "phi3", "phi4",
    // GLM family (Zhipu AI) — GLM-4, GLM-4.7
    "glm4",
    // Command-R (Cohere)
    "command-r",
    // StableLM (Stability AI)
    "stablelm",
    // InternLM-2/3 (Shanghai AI Lab)
    "internlm2",
    // DeepSeek-V2/V3 (DeepSeek AI)
    "deepseek2",
    // OLMo / OLMo-2 (Allen AI)
    "olmo", "olmo2",
    // EXAONE (LG AI)
    "exaone",
    // Granite (IBM)
    "granite",
    // MiMo (Xiaomi AI) — MiMo-V2.5
    "mimo2",
    // StarCoder2 (BigCode) — code generation, 3B/7B/15B
    "starcoder2",
    // Phi-2 (Microsoft) — predecessor to phi3
    "phi2",
    // SOLAR (Upstage) — LLaMA-based
    "solar",
    // Baichuan / Baichuan2 (Baichuan AI) — Chinese LLMs
    "baichuan", "baichuan2",
    // Grok (xAI) — very large MoE
    "grok",
    // Falcon (TII UAE) — Falcon 1/2; Falcon 3 uses "llama"
    "falcon",
];

fn arch_config(arch: &str) -> Option<ArchCfg> {
    match arch {
        "gemma" | "gemma2" | "gemma3" | "gemma4" => Some(ArchCfg::geglu_tied()),
        a if SUPPORTED_ARCHS.contains(&a) => Some(ArchCfg::swiglu()),
        _ => None,
    }
}

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

    let arch_cfg = arch_config(&architecture).ok_or_else(|| {
        anyhow::anyhow!(
            "Architecture '{}' is not supported. Supported families: {}.",
            architecture,
            SUPPORTED_ARCHS.join(", ")
        )
    })?;

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
    let vocab_size = meta_u64(meta, &format!("{a}.vocab_size"))
        .unwrap_or(32000) as usize;
    let rope_freq_base_raw = meta_f32(meta, &format!("{a}.rope.freq_base"))
        .unwrap_or(10000.0);

    let head_dim = embed_dim / n_heads;

    // RoPE scaling: linear divides positions by factor; NTK/YaRN scale freq_base instead.
    let rope_scaling_type   = meta_str(meta, &format!("{a}.rope.scaling.type")).unwrap_or("");
    let rope_scaling_factor = meta_f32(meta, &format!("{a}.rope.scaling.factor")).unwrap_or(1.0);
    let (rope_freq_base, rope_scale_factor) = match rope_scaling_type {
        "linear" | "longrope" => (rope_freq_base_raw, rope_scaling_factor.max(1.0)),
        "yarn" | "ntk" | "dynamic_ntk" if head_dim > 2 => {
            let exponent = head_dim as f32 / (head_dim as f32 - 2.0);
            (rope_freq_base_raw * rope_scaling_factor.powf(exponent), 1.0)
        }
        _ => (rope_freq_base_raw, 1.0),
    };
    if rope_scale_factor != 1.0 || rope_scaling_type == "yarn" || rope_scaling_type == "ntk" {
        tracing::info!(
            "RoPE scaling: type='{}' factor={:.2} → freq_base={:.0} scale={}",
            rope_scaling_type, rope_scaling_factor, rope_freq_base, rope_scale_factor
        );
    }

    // Read the true FFN intermediate size from metadata.
    // Falls back to a LLaMA-style SwiGLU estimate (rounds up to 256-multiple) when absent.
    let ffn_hidden_size = meta_u64(meta, &format!("{a}.feed_forward_length"))
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            let est = (embed_dim * 8).div_ceil(3);
            est.div_ceil(256) * 256
        });

    let ffn_activation = arch_cfg.ffn_activation;
    let tied_embeddings = arch_cfg.tied_embeddings;

    // MoE fields — 0 for dense models.
    let n_experts = meta_u64(meta, &format!("{a}.expert_count")).unwrap_or(0) as usize;
    let n_experts_used = meta_u64(meta, &format!("{a}.expert_used_count"))
        .map(|v| v as usize)
        // Default to top-2 when the count is absent but experts exist (e.g. older Mixtral GGUFs).
        .unwrap_or(if n_experts > 0 { 2 } else { 0 });
    // Some models (DeepSeek V3) use a smaller per-expert FFN than the shared FFN hidden size.
    let expert_ffn_hidden_size = meta_u64(meta, &format!("{a}.expert_feed_forward_length"))
        .map(|v| v as usize)
        .unwrap_or(ffn_hidden_size);

    if n_experts > 0 {
        tracing::info!(
            "MoE model: {} total experts, top-{} per token, expert_ffn={}",
            n_experts, n_experts_used, expert_ffn_hidden_size
        );
    }

    let rms_norm_eps = meta_f32(meta, &format!("{a}.attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5);

    Ok(GgufMeta {
        architecture,
        context_length,
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size,
        rope_freq_base,
        rope_scale_factor,
        ffn_hidden_size,
        ffn_activation,
        tied_embeddings,
        n_experts,
        n_experts_used,
        expert_ffn_hidden_size,
        rms_norm_eps,
    })
}
