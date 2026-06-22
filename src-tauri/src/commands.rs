use crate::engine::{EngineHandle, InferenceParams, ModelInfo};
use crate::models::{scan_models_dir, ModelEntry};
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{ipc::Channel, AppHandle, State};
use tracing::{error, info};

pub struct ModelsDir(pub PathBuf);

// ── Model management ────────────────────────────────────────────────────────

#[tauri::command]
pub async fn list_models(
    models_dir: State<'_, ModelsDir>,
) -> Result<Vec<ModelEntry>, String> {
    scan_models_dir(&models_dir.0).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_models_dir(models_dir: State<'_, ModelsDir>) -> Result<String, String> {
    Ok(models_dir.0.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn load_model(
    path: String,
    engine: State<'_, EngineHandle>,
) -> Result<ModelInfo, String> {
    info!("Loading model: {}", path);
    let mut eng = engine.lock().await;
    eng.load(&path).map_err(|e| {
        error!("Failed to load model: {}", e);
        e.to_string()
    })
}

#[tauri::command]
pub async fn unload_model(engine: State<'_, EngineHandle>) -> Result<(), String> {
    info!("Unloading model");
    let mut eng = engine.lock().await;
    eng.unload().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_loaded_model(
    engine: State<'_, EngineHandle>,
) -> Result<Option<ModelInfo>, String> {
    let eng = engine.lock().await;
    Ok(eng.model_info().cloned())
}

// ── Inference ────────────────────────────────────────────────────────────────

/// Streamed token payload sent through the Tauri Channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamEvent {
    pub token: String,
    pub done: bool,
}

#[tauri::command]
pub async fn run_inference(
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    repeat_penalty: Option<f32>,
    channel: Channel<StreamEvent>,
    engine: State<'_, EngineHandle>,
) -> Result<(), String> {
    info!("Starting inference, prompt length={}", prompt.len());

    let params = InferenceParams {
        prompt,
        max_tokens: max_tokens.unwrap_or(512),
        temperature: temperature.unwrap_or(0.7),
        top_p: top_p.unwrap_or(0.9),
        repeat_penalty: repeat_penalty.unwrap_or(1.1),
    };

    let eng = engine.lock().await;

    if !eng.is_loaded() {
        return Err("No model loaded. Select and load a model first.".to_string());
    }

    eng.infer(&params, &mut |tok| {
        channel
            .send(StreamEvent {
                token: tok.token,
                done: tok.done,
            })
            .map_err(|e| anyhow!("channel send error: {}", e))
    })
    .map_err(|e| e.to_string())
}

// ── System info ──────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemInfo {
    pub backend: String,
    pub engine_loaded: bool,
    pub model_name: Option<String>,
}

#[tauri::command]
pub async fn get_system_info(engine: State<'_, EngineHandle>) -> Result<SystemInfo, String> {
    let eng = engine.lock().await;
    let (engine_loaded, model_name) = if let Some(info) = eng.model_info() {
        (true, Some(info.name.clone()))
    } else {
        (false, None)
    };

    Ok(SystemInfo {
        backend: if cfg!(feature = "engine-llama") {
            "llama-cpp".to_string()
        } else {
            "stub".to_string()
        },
        engine_loaded,
        model_name,
    })
}
