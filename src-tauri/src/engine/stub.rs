//! Stub engine – compiles everywhere, streams mock tokens.
//! Replace with engine-llama feature for real inference.

use super::{InferenceParams, LlmEngine, ModelInfo, TokenEvent};
use anyhow::Result;
use std::path::Path;
use std::thread;
use std::time::Duration;

pub struct StubEngine {
    loaded: Option<ModelInfo>,
}

impl StubEngine {
    pub fn new() -> Self {
        Self { loaded: None }
    }
}

impl LlmEngine for StubEngine {
    fn load(&mut self, model_path: &str) -> Result<ModelInfo> {
        let path = Path::new(model_path);
        let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let info = ModelInfo {
            name,
            path: model_path.to_string(),
            size_bytes,
            backend: "stub".to_string(),
        };
        self.loaded = Some(info.clone());
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
        self.loaded.as_ref()
    }

    fn infer(
        &self,
        params: &InferenceParams,
        on_token: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<()> {
        let response = format!(
            "[Stub engine] Echo: \"{}\"\n\nThis is a placeholder response. \
            Load a real GGUF model and enable the `engine-llama` feature to \
            get actual LLM inference. The architecture uses wgpu for GPU \
            dispatch and llama-cpp-rs for optimized GGUF execution.",
            params.prompt.trim()
        );

        for word in response.split_inclusive(' ') {
            on_token(TokenEvent {
                token: word.to_string(),
                done: false,
            })?;
            thread::sleep(Duration::from_millis(30));
        }

        on_token(TokenEvent {
            token: String::new(),
            done: true,
        })?;
        Ok(())
    }
}
