//! Thin newtype that wraps CandleEngine as a CPU fallback for the wgpu engine path.
//! Used when no suitable GPU adapter is found at startup.

use crate::engine::candle_engine::CandleEngine;
use crate::engine::{InferenceParams, LlmEngine, ModelInfo, TokenEvent};
use anyhow::Result;

pub struct CpuFallback(pub CandleEngine);

impl CpuFallback {
    pub fn new() -> Self {
        Self(CandleEngine::new())
    }
}

impl LlmEngine for CpuFallback {
    fn load(&mut self, model_path: &str) -> Result<ModelInfo> {
        let mut info = self.0.load(model_path)?;
        info.backend = "wgpu-cpu-fallback".to_string();
        Ok(info)
    }

    fn unload(&mut self) -> Result<()> {
        self.0.unload()
    }

    fn is_loaded(&self) -> bool {
        self.0.is_loaded()
    }

    fn model_info(&self) -> Option<&ModelInfo> {
        self.0.model_info()
    }

    fn infer(&self, params: &InferenceParams, on_token: &mut dyn FnMut(TokenEvent) -> Result<()>) -> Result<()> {
        self.0.infer(params, on_token)
    }
}
