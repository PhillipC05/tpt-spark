mod commands;
mod engine;
mod models;

use commands::ModelsDir;
use engine::default_engine;
use models::default_models_dir;
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

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            app.manage(default_engine());
            app.manage(ModelsDir(default_models_dir()));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_models,
            commands::get_models_dir,
            commands::load_model,
            commands::unload_model,
            commands::get_loaded_model,
            commands::run_inference,
            commands::get_system_info,
        ])
        .run(tauri::generate_context!())
        .expect("error while running TPT Spark");
}
