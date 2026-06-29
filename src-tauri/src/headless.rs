// Headless JSON-RPC 2.0 server for tpt-spark.
//
// Activated by passing --headless (or setting TPT_SPARK_HEADLESS=1) before the
// Tauri GUI starts. The server listens on:
//   Windows : \\.\pipe\tpt-spark
//   macOS   : /tmp/tpt-spark.sock
//   Linux   : $XDG_RUNTIME_DIR/tpt-spark.sock  (falls back to /tmp/tpt-spark.sock)
//
// Protocol: newline-delimited JSON-RPC 2.0.
// Streaming methods (spark_infer) emit multiple response lines with the same id
// before the final done:true response.

use crate::engine::{default_engine, EngineHandle, InferenceParams};
use crate::models::scan_models_dir;
use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{error, info};

// ── Socket path ──────────────────────────────────────────────────────────────

pub fn socket_path() -> String {
    #[cfg(target_os = "windows")]
    return r"\\.\pipe\tpt-spark".to_string();

    #[cfg(target_os = "macos")]
    return "/tmp/tpt-spark.sock".to_string();

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let runtime = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
        format!("{}/tpt-spark.sock", runtime)
    }
}

// ── JSON-RPC types ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct RpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

fn ok_response(id: &Option<Value>, result: Value) -> String {
    let resp = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    });
    resp.to_string()
}

fn err_response(id: &Option<Value>, code: i32, message: &str) -> String {
    let resp = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    });
    resp.to_string()
}

// ── Entry point ──────────────────────────────────────────────────────────────

pub async fn run_headless(models_dir: PathBuf) -> Result<()> {
    let engine = default_engine();
    let cancel = Arc::new(AtomicBool::new(false));

    info!("Headless server starting on {}", socket_path());

    #[cfg(target_os = "windows")]
    run_named_pipe(engine, cancel, models_dir).await?;

    #[cfg(not(target_os = "windows"))]
    run_unix_socket(engine, cancel, models_dir).await?;

    Ok(())
}

// ── Windows named-pipe server ────────────────────────────────────────────────

#[cfg(target_os = "windows")]
async fn run_named_pipe(
    engine: EngineHandle,
    cancel: Arc<AtomicBool>,
    models_dir: PathBuf,
) -> Result<()> {
    use tokio::net::windows::named_pipe::ServerOptions;

    let pipe_name = socket_path();

    loop {
        let server = ServerOptions::new()
            .first_pipe_instance(false)
            .create(&pipe_name)?;

        server.connect().await?;
        info!("Headless client connected");

        let engine = Arc::clone(&engine);
        let cancel = Arc::clone(&cancel);
        let models_dir = models_dir.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_pipe_connection(server, engine, cancel, models_dir).await {
                error!("Headless connection error: {e:#}");
            }
        });
    }
}

#[cfg(target_os = "windows")]
async fn handle_pipe_connection(
    pipe: tokio::net::windows::named_pipe::NamedPipeServer,
    engine: EngineHandle,
    cancel: Arc<AtomicBool>,
    models_dir: PathBuf,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let (reader, mut writer) = tokio::io::split(pipe);
    let mut lines = BufReader::new(reader).lines();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_owned();
        if line.is_empty() {
            continue;
        }
        let responses =
            dispatch(&line, &engine, &cancel, &models_dir).await;
        for resp in responses {
            writer.write_all(resp.as_bytes()).await?;
            writer.write_all(b"\n").await?;
        }
        writer.flush().await?;
    }
    Ok(())
}

// ── Unix socket server ───────────────────────────────────────────────────────

#[cfg(not(target_os = "windows"))]
async fn run_unix_socket(
    engine: EngineHandle,
    cancel: Arc<AtomicBool>,
    models_dir: PathBuf,
) -> Result<()> {
    use tokio::net::UnixListener;

    let path = socket_path();
    // Remove stale socket file if it exists.
    let _ = std::fs::remove_file(&path);

    let listener = UnixListener::bind(&path)?;
    info!("Headless server listening on {}", path);

    loop {
        let (stream, _) = listener.accept().await?;
        info!("Headless client connected");

        let engine = Arc::clone(&engine);
        let cancel = Arc::clone(&cancel);
        let models_dir = models_dir.clone();

        tokio::spawn(async move {
            if let Err(e) =
                handle_unix_connection(stream, engine, cancel, models_dir).await
            {
                error!("Headless connection error: {e:#}");
            }
        });
    }
}

#[cfg(not(target_os = "windows"))]
async fn handle_unix_connection(
    stream: tokio::net::UnixStream,
    engine: EngineHandle,
    cancel: Arc<AtomicBool>,
    models_dir: PathBuf,
) -> Result<()> {
    let (reader, mut writer) = tokio::io::split(stream);
    let mut lines = BufReader::new(reader).lines();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_owned();
        if line.is_empty() {
            continue;
        }
        let responses = dispatch(&line, &engine, &cancel, &models_dir).await;
        for resp in responses {
            writer.write_all(resp.as_bytes()).await?;
            writer.write_all(b"\n").await?;
        }
        writer.flush().await?;
    }
    Ok(())
}

// ── Dispatcher ───────────────────────────────────────────────────────────────

/// Parse one JSON-RPC request line and return one or more response lines.
/// Streaming methods produce many lines; all other methods produce exactly one.
async fn dispatch(
    raw: &str,
    engine: &EngineHandle,
    cancel: &Arc<AtomicBool>,
    models_dir: &PathBuf,
) -> Vec<String> {
    let req: RpcRequest = match serde_json::from_str(raw) {
        Ok(r) => r,
        Err(e) => {
            return vec![err_response(&None, -32700, &format!("parse error: {e}"))];
        }
    };

    match req.method.as_str() {
        "spark_listModels" => {
            let result = handle_list_models(models_dir);
            vec![match result {
                Ok(v) => ok_response(&req.id, v),
                Err(e) => err_response(&req.id, -32000, &e.to_string()),
            }]
        }
        "spark_loadModel" => {
            let result = handle_load_model(&req.params, engine).await;
            vec![match result {
                Ok(v) => ok_response(&req.id, v),
                Err(e) => err_response(&req.id, -32000, &e.to_string()),
            }]
        }
        "spark_infer" => handle_infer(&req.id, &req.params, engine, cancel).await,
        "spark_cancel" => {
            cancel.store(true, Ordering::Relaxed);
            vec![ok_response(&req.id, json!({"ok": true}))]
        }
        "spark_lastBenchmark" => {
            let result = handle_last_benchmark();
            vec![match result {
                Ok(v) => ok_response(&req.id, v),
                Err(e) => err_response(&req.id, -32000, &e.to_string()),
            }]
        }
        other => vec![err_response(
            &req.id,
            -32601,
            &format!("method not found: {other}"),
        )],
    }
}

// ── Method handlers ──────────────────────────────────────────────────────────

fn handle_list_models(models_dir: &PathBuf) -> Result<Value> {
    let entries = scan_models_dir(models_dir)?;
    let list: Vec<Value> = entries
        .iter()
        .map(|e| {
            json!({
                "name": e.name,
                "arch": null,
                "size_gb": e.size_bytes as f64 / 1_073_741_824.0,
            })
        })
        .collect();
    Ok(json!(list))
}

async fn handle_load_model(params: &Option<Value>, engine: &EngineHandle) -> Result<Value> {
    let name = params
        .as_ref()
        .and_then(|p| p.get("name"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("params.name is required"))?
        .to_owned();

    let mut eng = engine.lock().await;
    tokio::task::block_in_place(|| eng.load(&name, None))?;
    Ok(json!({"ok": true}))
}

async fn handle_infer(
    id: &Option<Value>,
    params: &Option<Value>,
    engine: &EngineHandle,
    cancel: &Arc<AtomicBool>,
) -> Vec<String> {
    let (prompt, system_prompt, max_tokens) = match extract_infer_params(params) {
        Ok(v) => v,
        Err(e) => return vec![err_response(id, -32602, &e.to_string())],
    };

    cancel.store(false, Ordering::Relaxed);

    let infer_params = InferenceParams {
        prompt,
        system_prompt,
        max_tokens,
        temperature: 0.7,
        top_p: 0.9,
        repeat_penalty: 1.1,
    };

    let eng = engine.lock().await;

    if !eng.is_loaded() {
        return vec![err_response(id, -32000, "no model loaded")];
    }

    let cancel_flag = Arc::clone(cancel);
    let mut responses: Vec<String> = Vec::new();
    let id_clone = id.clone();

    let result = tokio::task::block_in_place(|| {
        eng.infer(&infer_params, &mut |tok| {
            if cancel_flag.load(Ordering::Relaxed) {
                anyhow::bail!("cancelled");
            }
            let line = if tok.done {
                ok_response(
                    &id_clone,
                    json!({"token": tok.token, "done": true}),
                )
            } else {
                // Intermediate events use a notification-style object (no id echo needed
                // but we include it for easy client-side correlation).
                json!({
                    "jsonrpc": "2.0",
                    "id": id_clone,
                    "result": {"token": tok.token, "done": false}
                })
                .to_string()
            };
            responses.push(line);
            Ok(())
        })
    });

    if let Err(e) = result {
        let msg = e.to_string();
        if !msg.contains("cancelled") {
            responses.push(err_response(id, -32000, &msg));
        }
    }

    responses
}

fn extract_infer_params(
    params: &Option<Value>,
) -> Result<(String, Option<String>, u32)> {
    let p = params
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("params required"))?;
    let prompt = p
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("params.prompt required"))?
        .to_owned();
    let system_prompt = p
        .get("system_prompt")
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned());
    let max_tokens = p
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as u32;
    Ok((prompt, system_prompt, max_tokens))
}

fn handle_last_benchmark() -> Result<Value> {
    let benchmarks_dir = dirs_next::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".tpt")
        .join("benchmarks");

    // Find the most recent spark-*.json file.
    let mut entries: Vec<_> = std::fs::read_dir(&benchmarks_dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("spark-")
        })
        .collect();

    entries.sort_by_key(|e| {
        e.metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });

    match entries.last() {
        None => Ok(Value::Null),
        Some(entry) => {
            let content = std::fs::read_to_string(entry.path())?;
            Ok(serde_json::from_str(&content)?)
        }
    }
}
