use crate::conversation::{
    delete_conversation, list_conversations, load_conversation, save_conversation, Conversation,
};
use crate::engine::{EngineHandle, InferenceParams, ModelInfo};
use crate::models::{save_models_json, scan_models_dir, ModelEntry};
use anyhow::anyhow;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{ipc::Channel, State};
use tracing::{error, info};

pub struct CancelFlag(pub Arc<AtomicBool>);

pub struct ModelsDir(pub Mutex<PathBuf>);
pub struct HistoryDir(pub PathBuf);
pub struct ConfigPath(pub PathBuf);
pub struct BenchmarksPath(pub PathBuf);

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

    // Refresh the shared models index.
    save_models_json(&models_dir.0.lock().unwrap());

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

    // Refresh the shared models index so other TPT tools see the new model.
    save_models_json(&models_dir.0.lock().unwrap());

    Ok(())
}

// ── Inference ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamEvent {
    pub token: String,
    pub done: bool,
    // Timing fields — only populated on the final done:true event, null otherwise.
    pub tokens_generated: Option<u32>,
    pub ttft_ms: Option<u64>,
    pub prefill_ms: Option<u64>,
    pub decode_ms: Option<u64>,
    pub tokens_per_sec: Option<f64>,
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
        let inference_start = Instant::now();
        let mut ttft_ms: Option<u64> = None;
        let mut prefill_end: Option<Instant> = None;
        let mut tokens_generated: u32 = 0;

        eng.infer(&params, &mut |tok| {
            if cancel_flag.load(Ordering::Relaxed) {
                anyhow::bail!("inference cancelled");
            }

            if !tok.done && ttft_ms.is_none() {
                ttft_ms = Some(inference_start.elapsed().as_millis() as u64);
                prefill_end = Some(Instant::now());
            }
            if !tok.done {
                tokens_generated += 1;
            }

            let event = if tok.done {
                let decode_ms = prefill_end.map(|pe| pe.elapsed().as_millis() as u64);
                let tokens_per_sec = decode_ms
                    .filter(|&d| d > 0)
                    .map(|d| tokens_generated as f64 / (d as f64 / 1000.0));
                StreamEvent {
                    token: tok.token,
                    done: true,
                    tokens_generated: Some(tokens_generated),
                    ttft_ms,
                    prefill_ms: ttft_ms,
                    decode_ms,
                    tokens_per_sec,
                }
            } else {
                StreamEvent {
                    token: tok.token,
                    done: false,
                    tokens_generated: None,
                    ttft_ms: None,
                    prefill_ms: None,
                    decode_ms: None,
                    tokens_per_sec: None,
                }
            };

            channel.send(event).map_err(|e| anyhow!("channel send error: {}", e))
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

// ── Benchmarks ───────────────────────────────────────────────────────────────

const BENCHMARK_PROMPT: &str =
    "Explain the difference between a stack and a heap in memory management. \
     Be concise and use a simple analogy.";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResult {
    pub id: String,
    pub model_name: String,
    pub backend: String,
    pub model_size_bytes: u64,
    pub prompt_tokens: u32,
    pub prompt_label: String,
    pub tokens_generated: u32,
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub total_ms: u64,
    pub tokens_per_sec: f64,
    pub toks_per_sec_per_gb: f64,
    pub ttft_ms: u64,
    pub timestamp: String,
}

fn load_benchmarks_file(path: &Path) -> Vec<BenchmarkResult> {
    if !path.exists() {
        return vec![];
    }
    match fs::read_to_string(path).and_then(|s| {
        serde_json::from_str(&s)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }) {
        Ok(results) => results,
        Err(e) => {
            tracing::warn!("Failed to read benchmarks file: {}", e);
            vec![]
        }
    }
}

fn save_benchmarks_file(path: &Path, results: &[BenchmarkResult]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(results)?)?;
    Ok(())
}

fn append_benchmark_result(path: &Path, result: &BenchmarkResult) -> anyhow::Result<()> {
    let mut results = load_benchmarks_file(path);
    results.insert(0, result.clone());
    save_benchmarks_file(path, &results)
}

// ── Crucible benchmark export ─────────────────────────────────────────────────

/// Minimal schema read by tpt-crucible to validate compiled edge-target performance.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct CrucibleBenchmarkRecord<'a> {
    model: &'a str,
    tokens_per_second: f64,
    time_to_first_token_ms: u64,
    prompt_tokens: u32,
    completion_tokens: u32,
    engine: &'a str,
    gpu_name: String,
    timestamp: &'a str,
}

fn write_crucible_benchmark(result: &BenchmarkResult) {
    let benchmarks_dir = dirs_next::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".tpt")
        .join("benchmarks");

    if let Err(e) = fs::create_dir_all(&benchmarks_dir) {
        tracing::warn!("Crucible benchmark dir: {e}");
        return;
    }

    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let path = benchmarks_dir.join(format!("spark-{date}.json"));

    let gpu_name = if result.backend.contains("wgpu") || result.backend.contains("tpt-gpu") {
        // Try to read GPU name from wgpu context; fall back to backend string.
        result.backend.clone()
    } else {
        "cpu".to_string()
    };

    let record = CrucibleBenchmarkRecord {
        model: &result.model_name,
        tokens_per_second: result.tokens_per_sec,
        time_to_first_token_ms: result.ttft_ms,
        prompt_tokens: result.prompt_tokens,
        completion_tokens: result.tokens_generated,
        engine: &result.backend,
        gpu_name,
        timestamp: &result.timestamp,
    };

    match serde_json::to_string_pretty(&record) {
        Ok(json) => {
            if let Err(e) = fs::write(&path, json) {
                tracing::warn!("Failed to write Crucible benchmark: {e}");
            } else {
                info!("Crucible benchmark written → {}", path.display());
            }
        }
        Err(e) => tracing::warn!("Crucible benchmark serialisation: {e}"),
    }
}

#[tauri::command]
pub async fn run_benchmark(
    max_tokens: u32,
    custom_prompt: Option<String>,
    engine: State<'_, EngineHandle>,
    benchmarks_path: State<'_, BenchmarksPath>,
) -> Result<BenchmarkResult, String> {
    let eng = engine.lock().await;

    if !eng.is_loaded() {
        return Err("No model loaded. Load a model before running a benchmark.".to_string());
    }

    let model_info = eng.model_info().unwrap().clone();

    let (prompt_text, prompt_label) = if let Some(ref cp) = custom_prompt {
        let label = {
            let chars: String = cp.chars().take(40).collect();
            if cp.chars().count() > 40 { format!("{chars}…") } else { chars }
        };
        (cp.clone(), label)
    } else {
        let label = match max_tokens {
            64  => "short-64".to_string(),
            128 => "medium-128".to_string(),
            256 => "long-256".to_string(),
            n   => format!("custom-{n}"),
        };
        (BENCHMARK_PROMPT.to_string(), label)
    };

    let params = InferenceParams {
        prompt: prompt_text.clone(),
        system_prompt: None,
        max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        repeat_penalty: 1.0,
    };

    info!(
        "Benchmark starting: model={} backend={} max_tokens={}",
        model_info.name, model_info.backend, max_tokens
    );

    let model_info_clone = model_info.clone();
    let bpath = benchmarks_path.0.clone();

    let result = tokio::task::block_in_place(|| {
        // Warm-up pass — discard output, not timed.
        info!("Benchmark warm-up pass");
        eng.infer(&params, &mut |_| Ok(()))?;

        // Timed pass.
        info!("Benchmark timed pass");
        let start = Instant::now();
        let mut ttft_ms: u64 = 0;
        let mut prefill_end: Option<Instant> = None;
        let mut tokens_generated: u32 = 0;

        eng.infer(&params, &mut |tok| {
            if !tok.done && prefill_end.is_none() {
                ttft_ms = start.elapsed().as_millis() as u64;
                prefill_end = Some(Instant::now());
            }
            if !tok.done {
                tokens_generated += 1;
            }
            Ok(())
        })?;

        let total_ms = start.elapsed().as_millis() as u64;
        let decode_ms = prefill_end.map(|pe| pe.elapsed().as_millis() as u64).unwrap_or(0);
        let tokens_per_sec = if decode_ms > 0 {
            tokens_generated as f64 / (decode_ms as f64 / 1000.0)
        } else {
            0.0
        };
        let model_size_gb = model_info_clone.size_bytes as f64 / 1_000_000_000.0;
        let toks_per_sec_per_gb = if model_size_gb > 0.0 { tokens_per_sec / model_size_gb } else { 0.0 };

        Ok::<BenchmarkResult, anyhow::Error>(BenchmarkResult {
            id: uuid::Uuid::new_v4().to_string(),
            model_name: model_info_clone.name.clone(),
            backend: model_info_clone.backend.clone(),
            model_size_bytes: model_info_clone.size_bytes,
            prompt_tokens: (prompt_text.len() / 4) as u32,
            prompt_label,
            tokens_generated,
            prefill_ms: ttft_ms,
            decode_ms,
            total_ms,
            tokens_per_sec,
            toks_per_sec_per_gb,
            ttft_ms,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    });

    match result {
        Ok(r) => {
            if let Err(e) = append_benchmark_result(&bpath, &r) {
                tracing::warn!("Failed to save benchmark result: {}", e);
            }
            write_crucible_benchmark(&r);
            info!("Benchmark complete: {:.1} tok/s", r.tokens_per_sec);
            Ok(r)
        }
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
pub async fn list_benchmarks(
    benchmarks_path: State<'_, BenchmarksPath>,
) -> Result<Vec<BenchmarkResult>, String> {
    Ok(load_benchmarks_file(&benchmarks_path.0))
}

#[tauri::command]
pub async fn delete_benchmark(
    id: String,
    benchmarks_path: State<'_, BenchmarksPath>,
) -> Result<(), String> {
    let mut results = load_benchmarks_file(&benchmarks_path.0);
    let before = results.len();
    results.retain(|r| r.id != id);
    if results.len() < before {
        save_benchmarks_file(&benchmarks_path.0, &results).map_err(|e| e.to_string())?;
    }
    Ok(())
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
