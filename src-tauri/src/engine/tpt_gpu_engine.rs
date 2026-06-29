// TptGpuEngine — placeholder that will wrap tpt-gpu-runtime once published.
//
// To activate the real implementation:
//   1. Wait for tpt-gpu to publish `tpt-gpu-runtime` (see tpt-gpu/todo1.md item 3)
//   2. Uncomment the [dependencies.tpt-gpu-runtime] entry in Cargo.toml
//   3. Add `tpt-gpu-runtime` to the `engine-tpt-gpu` feature list
//   4. Replace the anyhow::bail! stubs below with calls to tpt_gpu_runtime::LlmInference
//
// GPU detection / fallback order when engine-tpt-gpu is active:
//   TptGpuEngine (if adapter found) → WgpuEngine → CandleEngine

use crate::engine::{InferenceParams, LlmEngine, ModelInfo, TokenEvent};
use anyhow::Result;

pub struct TptGpuEngine {
    model_info: Option<ModelInfo>,
}

impl TptGpuEngine {
    pub fn new() -> Self {
        Self { model_info: None }
    }

    /// Returns true when a tpt-gpu-runtime compatible GPU adapter is available.
    /// Replace this stub with an actual adapter probe once the crate is integrated.
    pub fn gpu_available() -> bool {
        false
    }
}

impl LlmEngine for TptGpuEngine {
    fn load(
        &mut self,
        _model_path: &str,
        _on_progress: Option<&(dyn Fn(u32, u32) + Send + Sync)>,
    ) -> Result<ModelInfo> {
        anyhow::bail!(
            "tpt-gpu-runtime is not yet integrated — build with \
             --features engine-wgpu or engine-candle instead"
        )
    }

    fn unload(&mut self) -> Result<()> {
        self.model_info = None;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model_info.is_some()
    }

    fn model_info(&self) -> Option<&ModelInfo> {
        self.model_info.as_ref()
    }

    fn infer(
        &self,
        _params: &InferenceParams,
        _on_token: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<()> {
        anyhow::bail!("tpt-gpu-runtime not loaded")
    }
}
