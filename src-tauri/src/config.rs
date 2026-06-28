use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct AppConfig {
    pub models_dir: Option<String>,
}

impl AppConfig {
    pub fn load(config_path: &Path) -> Self {
        std::fs::read_to_string(config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, config_path: &Path) -> Result<()> {
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(config_path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

pub fn default_config_path() -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tpt-spark")
        .join("config.json")
}
