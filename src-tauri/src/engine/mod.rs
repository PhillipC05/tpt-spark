pub mod stub;

#[cfg(any(feature = "engine-candle", feature = "engine-wgpu"))]
pub mod candle_engine;

#[cfg(feature = "engine-wgpu")]
pub mod wgpu_context;
#[cfg(feature = "engine-wgpu")]
pub mod wgpu_loader;
#[cfg(feature = "engine-wgpu")]
pub mod wgpu_kvcache;
#[cfg(feature = "engine-wgpu")]
pub mod wgpu_ops;
#[cfg(feature = "engine-wgpu")]
pub mod wgpu_engine;
#[cfg(feature = "engine-wgpu")]
pub mod cpu_fallback;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A single streamed token event sent back to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenEvent {
    pub token: String,
    pub done: bool,
}

/// Parameters controlling a single inference run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceParams {
    /// The full formatted prompt (may include conversation history).
    pub prompt: String,
    /// Optional system/persona instruction prepended before the prompt.
    pub system_prompt: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            system_prompt: None,
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
        }
    }
}

/// Core trait that every engine backend must implement.
pub trait LlmEngine: Send + Sync {
    fn load(&mut self, model_path: &str) -> Result<ModelInfo>;
    fn unload(&mut self) -> Result<()>;
    fn is_loaded(&self) -> bool;
    fn model_info(&self) -> Option<&ModelInfo>;

    /// Run inference, calling `on_token` for every generated token.
    /// Implementations should handle the KV cache internally.
    fn infer(
        &self,
        params: &InferenceParams,
        on_token: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<()>;
}

/// Metadata about a loaded model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub backend: String,
}

/// Shared, thread-safe engine handle used from Tauri state.
pub type EngineHandle = Arc<tokio::sync::Mutex<Box<dyn LlmEngine>>>;

pub fn default_engine() -> EngineHandle {
    #[cfg(feature = "engine-wgpu")]
    {
        Arc::new(tokio::sync::Mutex::new(
            Box::new(wgpu_engine::WgpuEngine::new()) as Box<dyn LlmEngine>,
        ))
    }
    #[cfg(all(feature = "engine-candle", not(feature = "engine-wgpu")))]
    {
        Arc::new(tokio::sync::Mutex::new(
            Box::new(candle_engine::CandleEngine::new()) as Box<dyn LlmEngine>,
        ))
    }
    #[cfg(not(any(feature = "engine-candle", feature = "engine-wgpu")))]
    {
        Arc::new(tokio::sync::Mutex::new(
            Box::new(stub::StubEngine::new()) as Box<dyn LlmEngine>,
        ))
    }
}
