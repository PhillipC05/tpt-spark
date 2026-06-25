//! GPU inference engine backed by wgpu (Vulkan / Metal / DirectX 12) + WGSL shaders.
//!
//! High-level flow:
//!   load()  → mmap GGUF, upload quantized weights to VRAM, compile shaders, build KV cache.
//!   infer() → tokenize prompt, run prefill + decode loop on GPU, stream tokens back.
//!
//! CPU fallback: when no suitable adapter is found at load() time, a CpuFallback wrapping
//! CandleEngine is used transparently.  The backend field reports "wgpu-cpu-fallback".
//!
//! KV cache is reset on each infer() call so there is no stale state between conversations.

use std::path::Path;

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use tracing::{info, warn};

use crate::engine::cpu_fallback::CpuFallback;
use crate::engine::wgpu_context::GpuContext;
use crate::engine::wgpu_kvcache::KvCache;
use crate::engine::wgpu_loader::{load_gguf_to_vram, LoadedWeights};
use crate::engine::wgpu_ops::{dispatch, readback_f32, BindingEntry, WgpuPipelines};
use crate::engine::{InferenceParams, LlmEngine, ModelInfo, TokenEvent};

// ── Top-level engine ───────────────────────────────────────────────────────────

pub struct WgpuEngine {
    state: EngineState,
}

enum EngineState {
    Empty,
    Gpu(GpuModel),
    Cpu(CpuFallback),
}

struct GpuModel {
    info: ModelInfo,
    ctx: GpuContext,
    weights: LoadedWeights,
    pipelines: WgpuPipelines,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    kv_cache: KvCache,
}

impl WgpuEngine {
    pub fn new() -> Self {
        Self { state: EngineState::Empty }
    }
}

// ── LlmEngine impl ─────────────────────────────────────────────────────────────

impl LlmEngine for WgpuEngine {
    fn load(&mut self, model_path: &str) -> Result<ModelInfo> {
        let path = Path::new(model_path);

        // Try to acquire a GPU context; fall back to CPU if none available.
        let gpu_ctx = pollster::block_on(GpuContext::try_init());

        if let Some(ctx) = gpu_ctx {
            info!("GPU context acquired; loading weights into VRAM");
            let weights = load_gguf_to_vram(path, &ctx)
                .context("uploading GGUF weights to VRAM")?;

            let tokenizer = find_tokenizer(path)?;
            let eos_token_id = find_eos_token(&tokenizer);

            let pipelines = WgpuPipelines::compile(&ctx)
                .context("compiling WGSL shaders")?;

            let m = &weights.meta;
            let kv_cache = KvCache::new(
                &ctx,
                m.n_layers,
                m.n_kv_heads,
                m.head_dim,
                m.context_length,
            );

            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            let info = ModelInfo {
                name: name.clone(),
                path: model_path.to_string(),
                size_bytes: weights.total_bytes,
                backend: format!("wgpu-{:?}", ctx.adapter_info.backend).to_lowercase(),
            };

            self.state = EngineState::Gpu(GpuModel {
                info: info.clone(),
                ctx,
                weights,
                pipelines,
                tokenizer,
                eos_token_id,
                kv_cache,
            });

            info!("Model '{}' ready on GPU", name);
            Ok(info)
        } else {
            info!("No GPU available; falling back to candle CPU engine");
            let mut fb = CpuFallback::new();
            let info = fb.load(model_path)?;
            self.state = EngineState::Cpu(fb);
            Ok(info)
        }
    }

    fn unload(&mut self) -> Result<()> {
        // Take ownership of the old state so we can call destroy() before drop.
        let old = std::mem::replace(&mut self.state, EngineState::Empty);
        if let EngineState::Gpu(m) = old {
            // Explicit destroy releases VRAM immediately rather than waiting for wgpu GC.
            m.kv_cache.destroy();
            m.weights.destroy();
            // GpuModel (and its wgpu::Device/Queue) drops here.
        }
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        !matches!(self.state, EngineState::Empty)
    }

    fn model_info(&self) -> Option<&ModelInfo> {
        match &self.state {
            EngineState::Empty => None,
            EngineState::Gpu(m) => Some(&m.info),
            EngineState::Cpu(fb) => fb.model_info(),
        }
    }

    fn infer(
        &self,
        params: &InferenceParams,
        on_token: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<()> {
        match &self.state {
            EngineState::Empty => bail!("no model loaded"),
            EngineState::Cpu(fb) => fb.infer(params, on_token),
            EngineState::Gpu(m) => gpu_infer(m, params, on_token),
        }
    }
}

// ── GPU inference loop ─────────────────────────────────────────────────────────

fn gpu_infer(
    m: &GpuModel,
    params: &InferenceParams,
    on_token: &mut dyn FnMut(TokenEvent) -> Result<()>,
) -> Result<()> {
    let meta = &m.weights.meta;

    // Tokenize.
    let encoding = m.tokenizer
        .encode(params.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenize: {}", e))?;
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

    // Truncate prompt if it leaves no room for new tokens.
    let max_prompt = meta.context_length.saturating_sub(params.max_tokens as usize);
    if tokens.len() > max_prompt {
        warn!("Prompt truncated {} → {} tokens", tokens.len(), max_prompt);
        let start = tokens.len() - max_prompt;
        tokens = tokens[start..].to_vec();
    }
    if tokens.is_empty() {
        bail!("prompt is empty after tokenization");
    }

    info!(
        "GPU inference: prompt_tokens={} max_new={} temp={}",
        tokens.len(), params.max_tokens, params.temperature
    );

    // Reset KV cache for this conversation turn.
    // SAFETY: `infer` is called with `m: &GpuModel` rather than `&mut GpuModel` because the
    // LlmEngine trait requires `&self`.  The exclusive access invariant is upheld by the
    // `EngineHandle` Mutex, which the caller holds for the entire duration of this call.
    // No other thread can reach this code concurrently, making the aliased &mut sound.
    let kv_cache = unsafe { &mut *(std::ptr::addr_of!(m.kv_cache) as *mut KvCache) };
    kv_cache.reset();

    let embed_dim = meta.n_heads * meta.head_dim;
    let ffn_dim = meta.ffn_hidden_size;

    // Allocate activation scratch buffers once — reused for every token.
    let act_x     = m.ctx.create_storage_buffer("act_x",    (embed_dim * 4) as u64);
    let act_q     = m.ctx.create_storage_buffer("act_q",    (meta.n_heads * meta.head_dim * 4) as u64);
    let act_k     = m.ctx.create_storage_buffer("act_k",    (meta.n_kv_heads * meta.head_dim * 4) as u64);
    let act_v     = m.ctx.create_storage_buffer("act_v",    (meta.n_kv_heads * meta.head_dim * 4) as u64);
    let act_attn  = m.ctx.create_storage_buffer("act_attn", (meta.n_heads * meta.context_length * 4) as u64);
    let act_gate  = m.ctx.create_storage_buffer("act_gate", (ffn_dim * 4) as u64);
    let act_up    = m.ctx.create_storage_buffer("act_up",   (ffn_dim * 4) as u64);
    let act_out   = m.ctx.create_storage_buffer("act_out",  (embed_dim * 4) as u64);
    let logit_buf = m.ctx.create_storage_buffer("logits",   (meta.vocab_size * 4) as u64);

    // Pre-allocate a dequantised embedding table buffer (vocab_size × embed_dim f32).
    // Reused every forward pass to avoid a ~500 MB allocation per token.
    let full_embed = m.ctx.create_storage_buffer(
        "embed_full",
        (meta.vocab_size * embed_dim * 4) as u64,
    );
    // Dequantise once here; the embedding table is static across all tokens.
    if let Some(embed_buf) = m.weights.buffers.get("token_embd.weight") {
        dequantize_tensor(&m.ctx, &m.pipelines, embed_buf, &full_embed, meta.vocab_size * embed_dim);
    }

    let temperature = if params.temperature > 0.0 { Some(params.temperature as f64) } else { None };
    let top_p = if params.top_p < 1.0 { Some(params.top_p as f64) } else { None };
    let mut logits_proc = LogitsProcessor::new(42, temperature, top_p);

    // Prefill: run all prompt tokens through the model.
    let prompt_len = tokens.len();
    for (pos, &tok) in tokens.iter().enumerate() {
        run_forward_pass(
            m, &full_embed,
            &act_x, &act_q, &act_k, &act_v, &act_attn,
            &act_gate, &act_up, &act_out, &logit_buf,
            kv_cache, tok, pos,
        )?;
    }

    // Sample first token from prefill logits.
    let logit_vec = readback_f32(&m.ctx, &logit_buf, meta.vocab_size);
    let mut next_token = sample_token(&mut logits_proc, &logit_vec, &tokens, params.repeat_penalty)?;

    // Autoregressive decode.
    let mut pos = prompt_len;
    let mut generated: usize = 0;

    loop {
        if next_token == m.eos_token_id { break; }
        if generated >= params.max_tokens as usize { break; }
        if pos >= meta.context_length {
            warn!("Context length limit reached");
            break;
        }

        let token_str = m.tokenizer
            .decode(&[next_token], true)
            .map_err(|e| anyhow::anyhow!("decode: {}", e))?;

        on_token(TokenEvent { token: token_str, done: false })?;
        tokens.push(next_token);
        generated += 1;

        run_forward_pass(
            m, &full_embed,
            &act_x, &act_q, &act_k, &act_v, &act_attn,
            &act_gate, &act_up, &act_out, &logit_buf,
            kv_cache, next_token, pos,
        )?;

        let logit_vec = readback_f32(&m.ctx, &logit_buf, meta.vocab_size);
        next_token = sample_token(&mut logits_proc, &logit_vec, &tokens, params.repeat_penalty)?;

        pos += 1;
    }

    info!("GPU inference complete: {} tokens generated", generated);
    on_token(TokenEvent { token: String::new(), done: true })?;
    Ok(())
}

// ── Forward pass (one token) ───────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_forward_pass(
    m: &GpuModel,
    full_embed: &wgpu::Buffer,   // pre-dequantised [vocab_size, embed_dim] f32
    act_x: &wgpu::Buffer,
    act_q: &wgpu::Buffer,
    act_k: &wgpu::Buffer,
    act_v: &wgpu::Buffer,
    act_attn: &wgpu::Buffer,
    act_gate: &wgpu::Buffer,
    act_up: &wgpu::Buffer,
    act_out: &wgpu::Buffer,
    logit_buf: &wgpu::Buffer,
    kv_cache: &mut KvCache,
    token: u32,
    pos: usize,
) -> Result<()> {
    let ctx = &m.ctx;
    let pipes = &m.pipelines;
    let meta = &m.weights.meta;
    let w = &m.weights.buffers;

    let embed_dim = meta.n_heads * meta.head_dim;

    // 1. Token embedding lookup: copy row `token` from the pre-dequantised table.
    let row_bytes = (embed_dim * 4) as u64;
    let src_offset = token as u64 * row_bytes;
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(full_embed, src_offset, act_x, 0, row_bytes);
    ctx.queue.submit([enc.finish()]);

    // 2. Process each transformer layer.
    for layer in 0..meta.n_layers {
        let l = layer.to_string();

        // RMS norm before attention.
        if let Some(attn_norm) = w.get(&format!("blk.{l}.attn_norm.weight")) {
            dispatch_rms_norm(ctx, pipes, act_x, attn_norm, act_out, 1, embed_dim);
        }

        // Q, K, V projections.
        let q_dim  = meta.n_heads * meta.head_dim;
        let kv_dim = meta.n_kv_heads * meta.head_dim;

        if let Some(wq) = w.get(&format!("blk.{l}.attn_q.weight")) {
            dequant_and_gemm(ctx, pipes, act_out, wq, act_q, 1, q_dim, embed_dim);
        }
        if let Some(wk) = w.get(&format!("blk.{l}.attn_k.weight")) {
            dequant_and_gemm(ctx, pipes, act_out, wk, act_k, 1, kv_dim, embed_dim);
        }
        if let Some(wv) = w.get(&format!("blk.{l}.attn_v.weight")) {
            dequant_and_gemm(ctx, pipes, act_out, wv, act_v, 1, kv_dim, embed_dim);
        }

        // RoPE on Q and K.
        dispatch_rope(ctx, pipes, act_q, meta.n_heads, meta.head_dim, pos, meta.rope_freq_base);
        dispatch_rope(ctx, pipes, act_k, meta.n_kv_heads, meta.head_dim, pos, meta.rope_freq_base);

        // Write K, V into this layer's cache slot at position `pos`.
        let kv_row_bytes = (kv_dim * 4) as u64;
        let kv_offset = kv_cache.offset_bytes(pos);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(act_k, 0, &kv_cache.k_layers[layer], kv_offset, kv_row_bytes);
        enc.copy_buffer_to_buffer(act_v, 0, &kv_cache.v_layers[layer], kv_offset, kv_row_bytes);
        ctx.queue.submit([enc.finish()]);

        let kv_len = pos + 1;

        // Attention scores, softmax, weighted sum of V — operating on this layer's KV only.
        dispatch_attention(
            ctx, pipes,
            act_q, &kv_cache.k_layers[layer], &kv_cache.v_layers[layer],
            act_attn, act_out,
            1, kv_len, meta.n_heads, meta.n_kv_heads, meta.head_dim, pos,
        );

        // Attention output projection.
        if let Some(wo) = w.get(&format!("blk.{l}.attn_output.weight")) {
            dequant_and_gemm(ctx, pipes, act_out, wo, act_x, 1, embed_dim, embed_dim);
        }

        // RMS norm before FFN.
        if let Some(ffn_norm) = w.get(&format!("blk.{l}.ffn_norm.weight")) {
            dispatch_rms_norm(ctx, pipes, act_x, ffn_norm, act_out, 1, embed_dim);
        }

        // Feed-forward (SwiGLU).
        let ffn_dim = meta.ffn_hidden_size;
        if let (Some(w_gate), Some(w_up), Some(w_down)) = (
            w.get(&format!("blk.{l}.ffn_gate.weight")),
            w.get(&format!("blk.{l}.ffn_up.weight")),
            w.get(&format!("blk.{l}.ffn_down.weight")),
        ) {
            dequant_and_gemm(ctx, pipes, act_out, w_gate, act_gate, 1, ffn_dim, embed_dim);
            dequant_and_gemm(ctx, pipes, act_out, w_up,   act_up,   1, ffn_dim, embed_dim);
            dispatch_silu(ctx, pipes, act_gate, act_up, ffn_dim);
            dequant_and_gemm(ctx, pipes, act_gate, w_down, act_x,   1, embed_dim, ffn_dim);
        }
    }

    // 3. Final RMS norm + LM head → logits.
    if let (Some(output_norm), Some(lm_head)) = (
        w.get("output_norm.weight"),
        w.get("output.weight"),
    ) {
        dispatch_rms_norm(ctx, pipes, act_x, output_norm, act_out, 1, embed_dim);
        dequant_and_gemm(ctx, pipes, act_out, lm_head, logit_buf, 1, meta.vocab_size, embed_dim);
    }

    kv_cache.seq_len = pos + 1;
    Ok(())
}

// ── Dispatch wrappers ─────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DequantParams { n_elements: u32, _pad: [u32; 3] }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
#[allow(non_snake_case)]
struct GemmDims { M: u32, N: u32, K: u32, _pad: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RmsNormParams { n_rows: u32, dim: u32, eps: f32, _pad: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RopeParams { n_heads: u32, head_dim: u32, seq_offset: u32, freq_base: f32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttnScoreParams {
    seq_len: u32, kv_len: u32, n_heads: u32, n_kv_heads: u32,
    head_dim: u32, seq_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttnSoftmaxParams { seq_len: u32, kv_len: u32, n_heads: u32, _pad: u32 }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttnOutParams {
    seq_len: u32, kv_len: u32, n_heads: u32, n_kv_heads: u32, head_dim: u32, _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SiluParams { n_elements: u32, _pad: [u32; 3] }

fn dequantize_tensor(
    ctx: &GpuContext, pipes: &WgpuPipelines,
    quant: &wgpu::Buffer, out: &wgpu::Buffer, n_elements: usize,
) {
    let p = DequantParams { n_elements: n_elements as u32, _pad: [0; 3] };
    dispatch(
        ctx, &pipes.dequant_q8,
        &[BindingEntry { binding: 0, buffer: quant }, BindingEntry { binding: 1, buffer: out }],
        Some(&p),
        (n_elements as u32 + 255) / 256, 1, 1,
    );
}

fn dequant_and_gemm(
    ctx: &GpuContext, pipes: &WgpuPipelines,
    input: &wgpu::Buffer, weight_q: &wgpu::Buffer, output: &wgpu::Buffer,
    m: usize, n: usize, k: usize,
) {
    let weight_f32 = ctx.create_storage_buffer("weight_f32", (n * k * 4) as u64);
    dequantize_tensor(ctx, pipes, weight_q, &weight_f32, n * k);

    let p = GemmDims { M: m as u32, N: n as u32, K: k as u32, _pad: 0 };
    dispatch(
        ctx, &pipes.gemm,
        &[
            BindingEntry { binding: 0, buffer: input },
            BindingEntry { binding: 1, buffer: &weight_f32 },
            BindingEntry { binding: 2, buffer: output },
        ],
        Some(&p),
        (n as u32 + 15) / 16,
        (m as u32 + 15) / 16,
        1,
    );
}

fn dispatch_rms_norm(
    ctx: &GpuContext, pipes: &WgpuPipelines,
    input: &wgpu::Buffer, weight: &wgpu::Buffer, output: &wgpu::Buffer,
    n_rows: usize, dim: usize,
) {
    let p = RmsNormParams { n_rows: n_rows as u32, dim: dim as u32, eps: 1e-5, _pad: 0 };
    dispatch(
        ctx, &pipes.rms_norm,
        &[
            BindingEntry { binding: 0, buffer: input },
            BindingEntry { binding: 1, buffer: weight },
            BindingEntry { binding: 2, buffer: output },
        ],
        Some(&p),
        n_rows as u32, 1, 1,
    );
}

fn dispatch_rope(
    ctx: &GpuContext, pipes: &WgpuPipelines,
    qk: &wgpu::Buffer, n_heads: usize, head_dim: usize, pos: usize, freq_base: f32,
) {
    let p = RopeParams {
        n_heads: n_heads as u32, head_dim: head_dim as u32,
        seq_offset: pos as u32, freq_base,
    };
    let n_pairs = n_heads * head_dim / 2;
    dispatch(
        ctx, &pipes.rope,
        &[BindingEntry { binding: 0, buffer: qk }],
        Some(&p),
        (n_pairs as u32 + 63) / 64, 1, 1,
    );
}

fn dispatch_attention(
    ctx: &GpuContext, pipes: &WgpuPipelines,
    q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer,
    scores: &wgpu::Buffer, output: &wgpu::Buffer,
    seq_len: usize, kv_len: usize,
    n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_offset: usize,
) {
    let score_p = AttnScoreParams {
        seq_len: seq_len as u32, kv_len: kv_len as u32,
        n_heads: n_heads as u32, n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32, seq_offset: seq_offset as u32,
    };
    dispatch(
        ctx, &pipes.attn_scores,
        &[
            BindingEntry { binding: 0, buffer: q },
            BindingEntry { binding: 1, buffer: k },
            BindingEntry { binding: 2, buffer: scores },
        ],
        Some(&score_p),
        (kv_len as u32 + 63) / 64,
        seq_len as u32,
        n_heads as u32,
    );

    let sm_p = AttnSoftmaxParams {
        seq_len: seq_len as u32, kv_len: kv_len as u32,
        n_heads: n_heads as u32, _pad: 0,
    };
    dispatch(
        ctx, &pipes.attn_softmax,
        &[BindingEntry { binding: 0, buffer: scores }],
        Some(&sm_p),
        (seq_len * n_heads) as u32, 1, 1,
    );

    let out_p = AttnOutParams {
        seq_len: seq_len as u32, kv_len: kv_len as u32,
        n_heads: n_heads as u32, n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32, _pad: 0,
    };
    dispatch(
        ctx, &pipes.attn_output,
        &[
            BindingEntry { binding: 0, buffer: scores },
            BindingEntry { binding: 1, buffer: v },
            BindingEntry { binding: 2, buffer: output },
        ],
        Some(&out_p),
        (head_dim as u32 + 63) / 64,
        seq_len as u32,
        n_heads as u32,
    );
}

fn dispatch_silu(
    ctx: &GpuContext, pipes: &WgpuPipelines,
    gate: &wgpu::Buffer, up: &wgpu::Buffer, n: usize,
) {
    let p = SiluParams { n_elements: n as u32, _pad: [0; 3] };
    dispatch(
        ctx, &pipes.silu,
        &[
            BindingEntry { binding: 0, buffer: gate },
            BindingEntry { binding: 1, buffer: up },
        ],
        Some(&p),
        (n as u32 + 255) / 256, 1, 1,
    );
}

// ── Token sampling (CPU-side) ─────────────────────────────────────────────────

fn sample_token(
    proc: &mut LogitsProcessor,
    logit_vec: &[f32],
    seen_tokens: &[u32],
    repeat_penalty: f32,
) -> Result<u32> {
    let mut logits = logit_vec.to_vec();
    apply_repeat_penalty(&mut logits, seen_tokens, repeat_penalty);
    let t = candle_core::Tensor::new(logits.as_slice(), &candle_core::Device::Cpu)?;
    Ok(proc.sample(&t)?)
}

fn apply_repeat_penalty(logits: &mut [f32], seen: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < f32::EPSILON { return; }
    for &id in seen {
        let idx = id as usize;
        if idx < logits.len() {
            if logits[idx] >= 0.0 { logits[idx] /= penalty; } else { logits[idx] *= penalty; }
        }
    }
}

// ── Tokenizer helpers ─────────────────────────────────────────────────────────

fn find_tokenizer(gguf_path: &Path) -> Result<Tokenizer> {
    let dir = gguf_path.parent().unwrap_or(Path::new("."));
    let stem = gguf_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    for path in &[dir.join("tokenizer.json"), dir.join(stem).join("tokenizer.json")] {
        if path.exists() {
            return Tokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("tokenizer load failed: {}", e));
        }
    }
    bail!(
        "tokenizer.json not found next to the GGUF file.\n\
        Download it from HuggingFace and place it at:\n  {}",
        dir.join("tokenizer.json").display()
    )
}

fn find_eos_token(tok: &Tokenizer) -> u32 {
    for candidate in &["<|eot_id|>", "<|end_of_text|>", "</s>", "<eos>", "[EOS]"] {
        if let Some(id) = tok.token_to_id(candidate) { return id; }
    }
    2
}
