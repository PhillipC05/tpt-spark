use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    pub name: String,
    pub filename: String,
    pub path: String,
    pub size_bytes: u64,
    pub size_human: String,
}

/// Scan `dir` for .gguf files and return metadata for each one.
pub fn scan_models_dir(dir: &Path) -> Result<Vec<ModelEntry>> {
    let mut entries = Vec::new();

    if !dir.exists() {
        fs::create_dir_all(dir)?;
        return Ok(entries);
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            let meta = fs::metadata(&path)?;
            let size_bytes = meta.len();
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&filename)
                .to_string();

            entries.push(ModelEntry {
                name,
                filename,
                path: path.to_string_lossy().to_string(),
                size_bytes,
                size_human: format_size(size_bytes),
            });
        }
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

pub fn default_models_dir() -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tpt-spark")
        .join("models")
}

fn format_size(bytes: u64) -> String {
    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    } else {
        format!("{} KB", bytes / 1024)
    }
}
