mod commands;
mod config;
mod conversation;
mod engine;
mod headless;
mod models;

use commands::{BenchmarksPath, CancelFlag, ConfigPath, HistoryDir, ModelsDir};
use config::{default_config_path, AppConfig};
use conversation::history_dir;
use engine::default_engine;
use models::{default_models_dir, legacy_models_dir, migrate_from_legacy_dir, save_models_json};
use std::sync::{atomic::AtomicBool, Arc, Mutex};
use tauri::Manager;
use tracing::info;

/// Run the headless JSON-RPC server (no GUI).
/// Called from `main.rs` when `--headless` or `TPT_SPARK_HEADLESS=1` is detected.
pub fn run_headless_blocking() {
    init_logging();
    info!("TPT Spark headless mode");

    let config_path = default_config_path();
    let cfg = AppConfig::load(&config_path);
    let models_dir = resolve_models_dir(&cfg);

    migrate_from_legacy_dir(&legacy_models_dir(), &models_dir);
    info!("Models directory: {}", models_dir.display());

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        if let Err(e) = headless::run_headless(models_dir).await {
            eprintln!("Headless server error: {e:#}");
            std::process::exit(1);
        }
    });
}

fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "tpt_spark_lib=info".parse().unwrap()),
        )
        .init();
}

fn resolve_models_dir(cfg: &AppConfig) -> std::path::PathBuf {
    cfg.models_dir
        .as_deref()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
        .unwrap_or_else(default_models_dir)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_logging();

    info!("TPT Spark starting");

    let config_path = default_config_path();
    let cfg = AppConfig::load(&config_path);

    // Migrate any models from the legacy per-app directory to ~/.tpt/models/.
    let new_models_default = default_models_dir();
    migrate_from_legacy_dir(&legacy_models_dir(), &new_models_default);
    save_models_json(&new_models_default);

    let models_dir = resolve_models_dir(&cfg);

    info!("Models directory: {}", models_dir.display());

    let data_dir = dirs_next::data_dir().unwrap_or_else(|| std::path::PathBuf::from("."));

    let hist_dir = history_dir(&data_dir);

    // Internal benchmarks list (full history, used by the GUI benchmark tab).
    let benchmarks_path = data_dir.join("tpt-spark").join("benchmarks.json");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            app.manage(default_engine());
            app.manage(ModelsDir(Mutex::new(models_dir)));
            app.manage(HistoryDir(hist_dir));
            app.manage(ConfigPath(config_path));
            app.manage(CancelFlag(Arc::new(AtomicBool::new(false))));
            app.manage(BenchmarksPath(benchmarks_path));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_models,
            commands::get_models_dir,
            commands::load_model,
            commands::unload_model,
            commands::get_loaded_model,
            commands::delete_model,
            commands::run_inference,
            commands::download_model,
            commands::save_conv,
            commands::list_convs,
            commands::load_conv,
            commands::delete_conv,
            commands::get_system_info,
            commands::cancel_inference,
            commands::pick_models_dir,
            commands::open_external_url,
            commands::run_benchmark,
            commands::list_benchmarks,
            commands::delete_benchmark,
        ])
        .run(tauri::generate_context!())
        .expect("error while running TPT Spark");
}
