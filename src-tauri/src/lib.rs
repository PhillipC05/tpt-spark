mod commands;
mod conversation;
mod engine;
mod models;

use commands::{CancelFlag, HistoryDir, ModelsDir};
use conversation::history_dir;
use engine::default_engine;
use models::default_models_dir;
use std::sync::{atomic::AtomicBool, Arc};
use tauri::Manager;
use tracing::info;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "tpt_spark_lib=info".parse().unwrap()),
        )
        .init();

    info!("TPT Spark starting");

    let models_dir = default_models_dir();
    let hist_dir = history_dir(
        &dirs_next::data_dir().unwrap_or_else(|| std::path::PathBuf::from(".")),
    );

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            app.manage(default_engine());
            app.manage(ModelsDir(models_dir));
            app.manage(HistoryDir(hist_dir));
            app.manage(CancelFlag(Arc::new(AtomicBool::new(false))));
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running TPT Spark");
}
