use crate::conversation::{
    delete_conversation, list_conversations, load_conversation, save_conversation, Conversation,
};
use crate::engine::{EngineHandle, InferenceParams, ModelInfo};
use crate::models::{scan_models_dir, ModelEntry};
use anyhow::anyhow;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{ipc::Channel, State};
use tracing::{error, info};

pub struct CancelFlag(pub Arc<AtomicBool>);

pub struct ModelsDir(pub Mutex<PathBuf>);
pub struct HistoryDir(pub PathBuf);
pub struct ConfigPath(pub PathBuf);

// ── Model management ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadProgress {
    pub tensors_done: u32,
    pub tensors_total: u32,
    pub done: bool,
}

#[tauri::command]
pub async fn list_models(
    models_dir: State<'_, ModelsDir>,
) -> Result<Vec<ModelEntry>, String> {
    let dir = models_dir.0.lock().unwrap().clone();
    scan_models_dir(&dir).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_models_dir(models_dir: State<'_, ModelsDir>) -> Result<String, String> {
    Ok(models_dir.0.lock().unwrap().to_string_lossy().to_string())
}

#[tauri::command]
pub async fn load_model(
    path: String,
    channel: Channel<LoadProgress>,
    engine: State<'_, EngineHandle>,
) -> Result<ModelInfo, String> {
    info!("Loading model: {}", path);
    let mut eng = engine.lock().await;
    tokio::task::block_in_place(|| {
        let cb = move |done: u32, total: u32| {
            let _ = channel.send(LoadProgress { tensors_done: done, tensors_total: total, done: false });
        };
        eng.load(&path, Some(&cb)).map_err(|e| {
            error!("Failed to load model: {:#}", e);
            format!("{:#}", e)
        })
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

#[tauri::command]
pub async fn delete_model(
    path: String,
    engine: State<'_, EngineHandle>,
    models_dir: State<'_, ModelsDir>,
) -> Result<(), String> {
    let canonical = fs::canonicalize(&path)
        .map_err(|e| format!("invalid path: {e}"))?;
    let models_canonical = fs::canonicalize(&*models_dir.0.lock().unwrap())
        .map_err(|e| format!("models dir error: {e}"))?;
    if !canonical.starts_with(&models_canonical) {
        return Err("path is outside the models directory".into());
    }

    // If this model is currently loaded, unload it first.
    {
        let eng = engine.lock().await;
        if eng.model_info().map(|i| i.path.as_str()) == Some(&path) {
            drop(eng);
            let mut eng = engine.lock().await;
            eng.unload().map_err(|e| e.to_string())?;
        }
    }
    info!("Deleting model file: {}", path);
    fs::remove_file(&canonical).map_err(|e| format!("delete_model: {e}"))?;
    Ok(())
}

// ── Model download ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: Option<u64>,
    pub done: bool,
    pub error: Option<String>,
}

#[tauri::command]
pub async fn download_model(
    url: String,
    filename: String,
    channel: Channel<DownloadProgress>,
    models_dir: State<'_, ModelsDir>,
) -> Result<(), String> {
    if !url.starts_with("https://") {
        return Err("only HTTPS URLs are allowed".into());
    }
    let safe_name = Path::new(&filename)
        .file_name()
        .ok_or_else(|| "invalid filename".to_string())?
        .to_str()
        .ok_or_else(|| "filename is not valid UTF-8".to_string())?
        .to_owned();
    let dest = models_dir.0.lock().unwrap().join(&safe_name);

    info!("Downloading model from {} → {}", url, dest.display());

    let resp = reqwest::get(&url)
        .await
        .map_err(|e| format!("download request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("server returned {}", resp.status()));
    }

    let total = resp.content_length();
    let mut stream = resp.bytes_stream();

    // Create/truncate destination file.
    let mut file = tokio::fs::File::create(&dest)
        .await
        .map_err(|e| format!("cannot create {}: {e}", dest.display()))?;

    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("stream error: {e}"))?;
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
            .await
            .map_err(|e| format!("write error: {e}"))?;
        downloaded += chunk.len() as u64;

        channel
            .send(DownloadProgress { downloaded, total, done: false, error: None })
            .map_err(|e| format!("channel: {e}"))?;
    }

    channel
        .send(DownloadProgress { downloaded, total, done: true, error: None })
        .map_err(|e| format!("channel: {e}"))?;

    info!("Download complete: {} bytes → {}", downloaded, dest.display());
    Ok(())
}

// ── Inference ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamEvent {
    pub token: String,
    pub done: bool,
}

#[tauri::command]
pub async fn run_inference(
    prompt: String,
    system_prompt: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    repeat_penalty: Option<f32>,
    channel: Channel<StreamEvent>,
    engine: State<'_, EngineHandle>,
    cancel: State<'_, CancelFlag>,
) -> Result<(), String> {
    cancel.0.store(false, Ordering::Relaxed);

    // Prepend system prompt when provided.
    let full_prompt = match &system_prompt {
        Some(sp) if !sp.trim().is_empty() => format!("{}\n\n{}", sp.trim(), prompt),
        _ => prompt,
    };

    info!("Starting inference, prompt_len={}", full_prompt.len());

    let params = InferenceParams {
        prompt: full_prompt,
        system_prompt,
        max_tokens: max_tokens.unwrap_or(512),
        temperature: temperature.unwrap_or(0.7),
        top_p: top_p.unwrap_or(0.9),
        repeat_penalty: repeat_penalty.unwrap_or(1.1),
    };

    let eng = engine.lock().await;

    if !eng.is_loaded() {
        return Err("No model loaded. Select and load a model first.".to_string());
    }

    let cancel_flag = Arc::clone(&cancel.0);
    let result = tokio::task::block_in_place(|| {
        eng.infer(&params, &mut |tok| {
            if cancel_flag.load(Ordering::Relaxed) {
                anyhow::bail!("inference cancelled");
            }
            channel
                .send(StreamEvent { token: tok.token, done: tok.done })
                .map_err(|e| anyhow!("channel send error: {}", e))
        })
    });

    match result {
        Ok(()) => Ok(()),
        Err(e) if e.to_string().contains("inference cancelled") => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
pub async fn cancel_inference(cancel: State<'_, CancelFlag>) -> Result<(), String> {
    cancel.0.store(true, Ordering::Relaxed);
    Ok(())
}

// ── Conversation history ─────────────────────────────────────────────────────

#[tauri::command]
pub async fn save_conv(
    conversation: Conversation,
    history_dir: State<'_, HistoryDir>,
) -> Result<(), String> {
    save_conversation(&history_dir.0, &conversation).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_convs(
    history_dir: State<'_, HistoryDir>,
) -> Result<Vec<Conversation>, String> {
    list_conversations(&history_dir.0).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn load_conv(
    id: String,
    history_dir: State<'_, HistoryDir>,
) -> Result<Conversation, String> {
    load_conversation(&history_dir.0, &id).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_conv(
    id: String,
    history_dir: State<'_, HistoryDir>,
) -> Result<(), String> {
    delete_conversation(&history_dir.0, &id).map_err(|e| e.to_string())
}

// ── Models directory picker ──────────────────────────────────────────────────

#[tauri::command]
pub async fn pick_models_dir(
    app_handle: tauri::AppHandle,
    models_dir: State<'_, ModelsDir>,
    config_path: State<'_, ConfigPath>,
) -> Result<Option<String>, String> {
    use crate::config::AppConfig;
    use tauri_plugin_dialog::DialogExt;
    use tokio::sync::oneshot;

    let (tx, rx) = oneshot::channel::<Option<tauri_plugin_dialog::FilePath>>();
    app_handle.dialog().file().pick_folder(move |result| {
        let _ = tx.send(result);
    });

    match rx.await.map_err(|e| format!("dialog closed: {e}"))? {
        Some(folder_path) => {
            let path_buf = folder_path
                .into_path()
                .map_err(|e| format!("invalid path: {e}"))?;
            fs::create_dir_all(&path_buf)
                .map_err(|e| format!("cannot create dir: {e}"))?;
            info!("Models directory changed to: {}", path_buf.display());
            *models_dir.0.lock().unwrap() = path_buf.clone();

            let cfg = AppConfig { models_dir: Some(path_buf.to_string_lossy().to_string()) };
            if let Err(e) = cfg.save(&config_path.0) {
                tracing::warn!("Failed to persist models dir config: {e}");
            }

            Ok(Some(path_buf.to_string_lossy().to_string()))
        }
        None => Ok(None),
    }
}

// ── Shell helpers ─────────────────────────────────────────────────────────────

#[tauri::command]
#[allow(deprecated)]
pub async fn open_external_url(url: String, app_handle: tauri::AppHandle) -> Result<(), String> {
    use tauri_plugin_shell::ShellExt;
    app_handle.shell().open(url, None).map_err(|e| e.to_string())
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
        backend: if cfg!(feature = "engine-wgpu") {
            "wgpu".to_string()
        } else if cfg!(feature = "engine-candle") {
            "candle-cpu".to_string()
        } else {
            "stub".to_string()
        },
        engine_loaded,
        model_name,
    })
}
