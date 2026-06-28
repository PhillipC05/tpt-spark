//! CPU inference engine backed by HuggingFace candle + quantized GGUF models.
//!
//! Design notes:
//! - The entire GGUF file is read into RAM on `load()` so the OS page cache
//!   is warm.  Each `infer()` call re-parses the GGUF header and rebuilds
//!   `ModelWeights` from the in-memory bytes, giving a fresh KV cache without
//!   any disk I/O.  Phase 3 will replace this with true mmap + wgpu.
//! - Tokenization requires a `tokenizer.json` next to the GGUF file.  Any
//!   HuggingFace-format tokenizer (LLaMA-3 BPE, Mistral SP-BPE, …) works.
//! - `infer` takes `&self` because mutable KV cache lives inside the locally
//!   constructed `ModelWeights`, not in `CandleEngine` fields.

use crate::engine::{InferenceParams, LlmEngine, ModelInfo, TokenEvent};
use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use std::io::Cursor;
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::{info, warn};

// ── Public engine struct ──────────────────────────────────────────────────────

pub struct CandleEngine {
    loaded: Option<LoadedModel>,
}

struct LoadedModel {
    info: ModelInfo,
    /// Entire GGUF file held in RAM so infer() never hits the disk.
    raw_gguf: Vec<u8>,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    context_length: usize,
}

impl CandleEngine {
    pub fn new() -> Self {
        Self { loaded: None }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Look for tokenizer.json next to the GGUF file (required for BPE encoding).
fn find_tokenizer(gguf_path: &Path) -> Result<Tokenizer> {
    let dir = gguf_path.parent().unwrap_or(Path::new("."));
    let stem = gguf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    let candidates = [
        dir.join("tokenizer.json"),
        dir.join(stem).join("tokenizer.json"),
    ];

    for path in &candidates {
        if path.exists() {
            info!("Loading tokenizer from {}", path.display());
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

/// Return the EOS token ID, checking common special-token names in order.
fn find_eos_token(tok: &Tokenizer) -> u32 {
    for candidate in &["<|eot_id|>", "<|end_of_text|>", "</s>", "<eos>", "[EOS]"] {
        if let Some(id) = tok.token_to_id(candidate) {
            return id;
        }
    }
    2 // LLaMA-2 / Mistral default
}

/// Divide logits of previously-seen tokens by `penalty` (standard repetition
/// penalty: values > 1.0 reduce the probability of repeating tokens).
fn apply_repeat_penalty(logits: &mut [f32], seen: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < f32::EPSILON {
        return;
    }
    for &id in seen {
        let idx = id as usize;
        if idx < logits.len() {
            if logits[idx] >= 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Extract a string value from GGUF metadata, returning `None` if absent or
/// the wrong type.
fn meta_str<'a>(meta: &'a std::collections::HashMap<String, gguf_file::Value>, key: &str) -> Option<&'a str> {
    meta.get(key).and_then(|v| {
        if let gguf_file::Value::String(s) = v {
            Some(s.as_str())
        } else {
            None
        }
    })
}

/// Extract a u64 from GGUF metadata, accepting both U32 and U64 variants.
fn meta_u64(meta: &std::collections::HashMap<String, gguf_file::Value>, key: &str) -> Option<u64> {
    meta.get(key).and_then(|v| match v {
        gguf_file::Value::U32(n) => Some(*n as u64),
        gguf_file::Value::U64(n) => Some(*n),
        _ => None,
    })
}

// ── LlmEngine implementation ──────────────────────────────────────────────────

impl LlmEngine for CandleEngine {
    fn load(&mut self, model_path: &str, _on_progress: Option<&(dyn Fn(u32, u32) + Send + Sync)>) -> Result<ModelInfo> {
        let path = Path::new(model_path);

        info!("Reading GGUF into memory: {}", model_path);
        let raw_gguf = std::fs::read(path)
            .with_context(|| format!("reading {}", model_path))?;
        let size_bytes = raw_gguf.len() as u64;

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Parse GGUF metadata to validate architecture + context length.
        let mut cursor = Cursor::new(&raw_gguf);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| anyhow::anyhow!("GGUF parse error: {}", e))?;

        let arch = meta_str(&content.metadata, "general.architecture")
            .unwrap_or("llama")
            .to_string();

        if !matches!(arch.as_str(), "llama" | "mistral") {
            bail!(
                "Architecture '{}' is not yet supported. Supported: llama, mistral.\n\
                Open a GitHub issue to request support for your model family.",
                arch
            );
        }

        let ctx_key = format!("{}.context_length", arch);
        let context_length = meta_u64(&content.metadata, &ctx_key)
            .unwrap_or(4096) as usize;

        info!(
            "GGUF: arch={} context_length={} size={:.1}GB",
            arch,
            context_length,
            size_bytes as f64 / 1_073_741_824.0
        );

        let tokenizer = find_tokenizer(path)?;
        let eos_token_id = find_eos_token(&tokenizer);
        info!("EOS token id: {}", eos_token_id);

        let info = ModelInfo {
            name: name.clone(),
            path: model_path.to_string(),
            size_bytes,
            backend: "candle-cpu".to_string(),
        };

        self.loaded = Some(LoadedModel {
            info: info.clone(),
            raw_gguf,
            tokenizer,
            eos_token_id,
            context_length,
        });

        info!("Model '{}' ready", name);
        Ok(info)
    }

    fn unload(&mut self) -> Result<()> {
        self.loaded = None;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    fn model_info(&self) -> Option<&ModelInfo> {
        self.loaded.as_ref().map(|m| &m.info)
    }

    fn infer(
        &self,
        params: &InferenceParams,
        on_token: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<()> {
        let model = self
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no model loaded"))?;

        // Reconstruct ModelWeights from in-memory GGUF bytes so every call
        // starts with a clean KV cache (no stale state from prior runs).
        let device = Device::Cpu;
        let mut cursor = Cursor::new(&model.raw_gguf);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| anyhow::anyhow!("GGUF re-parse: {}", e))?;
        let mut weights = ModelWeights::from_gguf(content, &mut cursor, &device)
            .context("building model weights for inference")?;

        // Tokenize the prompt.
        let encoding = model
            .tokenizer
            .encode(params.prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenize error: {}", e))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

        // Truncate the prompt if it leaves no room for generated tokens.
        let max_prompt = model
            .context_length
            .saturating_sub(params.max_tokens as usize);
        if tokens.len() > max_prompt {
            warn!(
                "Prompt truncated from {} → {} tokens (context_length={}, max_tokens={})",
                tokens.len(),
                max_prompt,
                model.context_length,
                params.max_tokens
            );
            let start = tokens.len() - max_prompt;
            tokens = tokens[start..].to_vec();
        }

        let prompt_len = tokens.len();
        if prompt_len == 0 {
            bail!("Prompt is empty after tokenization");
        }

        info!(
            "Inference: prompt_tokens={} max_new_tokens={} temp={} top_p={}",
            prompt_len, params.max_tokens, params.temperature, params.top_p
        );

        let temperature = if params.temperature > 0.0 {
            Some(params.temperature as f64)
        } else {
            None
        };
        let top_p = if params.top_p < 1.0 {
            Some(params.top_p as f64)
        } else {
            None
        };
        let mut logits_processor = LogitsProcessor::new(42, temperature, top_p);

        // ── Prefill phase ─────────────────────────────────────────────────────
        // Feed all prompt tokens in one forward pass; the KV cache is populated
        // for positions 0..prompt_len.
        let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
        let logits = weights.forward(&input, 0)?;
        // Shape: [1, prompt_len, vocab_size] → [vocab_size]
        let logits = logits.squeeze(0)?;
        let last_logits = logits.get(prompt_len - 1)?;

        let mut logit_vec: Vec<f32> = last_logits.to_vec1()?;
        apply_repeat_penalty(&mut logit_vec, &tokens, params.repeat_penalty);
        let logit_t = Tensor::new(logit_vec.as_slice(), &device)?;
        let mut next_token = logits_processor.sample(&logit_t)?;

        // ── Autoregressive decode phase ────────────────────────────────────────
        let mut pos = prompt_len;
        let mut generated: usize = 0;

        loop {
            if next_token == model.eos_token_id {
                break;
            }
            if generated >= params.max_tokens as usize {
                break;
            }
            if pos >= model.context_length {
                warn!("Context length limit reached ({} tokens)", model.context_length);
                break;
            }

            let token_str = model
                .tokenizer
                .decode(&[next_token], true)
                .map_err(|e| anyhow::anyhow!("decode error: {}", e))?;

            on_token(TokenEvent {
                token: token_str,
                done: false,
            })?;

            tokens.push(next_token);
            generated += 1;

            // Single-token forward pass at the current sequence position.
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let logits = weights.forward(&input, pos)?;
            // Shape: [1, 1, vocab_size] → [vocab_size]
            let logits = logits.squeeze(0)?.squeeze(0)?;

            let mut logit_vec: Vec<f32> = logits.to_vec1()?;
            apply_repeat_penalty(&mut logit_vec, &tokens, params.repeat_penalty);
            let logit_t = Tensor::new(logit_vec.as_slice(), &device)?;
            next_token = logits_processor.sample(&logit_t)?;

            pos += 1;
        }

        info!("Inference complete: {} tokens generated", generated);

        on_token(TokenEvent {
            token: String::new(),
            done: true,
        })?;

        Ok(())
    }
}
