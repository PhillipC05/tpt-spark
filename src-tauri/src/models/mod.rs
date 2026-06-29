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

/// Canonical shared models directory used by all TPT suite tools.
pub fn tpt_models_dir() -> PathBuf {
    dirs_next::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".tpt")
        .join("models")
}

/// Legacy per-app models directory (kept for migration).
pub fn legacy_models_dir() -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tpt-spark")
        .join("models")
}

pub fn default_models_dir() -> PathBuf {
    tpt_models_dir()
}

/// Path to the models index inside the shared models directory.
pub fn models_json_path(models_dir: &Path) -> PathBuf {
    models_dir.join("models.json")
}

/// Write a models.json index to `models_dir` reflecting its current contents.
pub fn save_models_json(models_dir: &Path) {
    match scan_models_dir(models_dir) {
        Ok(entries) => {
            if let Ok(json) = serde_json::to_string_pretty(&entries) {
                let _ = fs::write(models_json_path(models_dir), json);
            }
        }
        Err(e) => tracing::warn!("save_models_json: {e}"),
    }
}

/// Load the models index from disk; falls back to a fresh scan when missing or stale.
#[allow(dead_code)]
pub fn load_models_with_cache(models_dir: &Path) -> Result<Vec<ModelEntry>> {
    let json_path = models_json_path(models_dir);
    if json_path.exists() {
        if let Ok(content) = fs::read_to_string(&json_path) {
            if let Ok(entries) = serde_json::from_str::<Vec<ModelEntry>>(&content) {
                return Ok(entries);
            }
        }
    }
    // Cache missing or invalid — fall back to scanning.
    scan_models_dir(models_dir)
}

/// Move .gguf (and adjacent tokenizer.json) files from `old_dir` to `new_dir`.
/// Skips migration when `new_dir` already contains any .gguf files.
pub fn migrate_from_legacy_dir(old_dir: &Path, new_dir: &Path) {
    if !old_dir.exists() {
        return;
    }

    // Check if new_dir already has models — if so, skip.
    let already_populated = new_dir.exists()
        && fs::read_dir(new_dir)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("gguf"));

    if already_populated {
        return;
    }

    if let Err(e) = fs::create_dir_all(new_dir) {
        tracing::warn!("migrate_from_legacy_dir: cannot create {}: {e}", new_dir.display());
        return;
    }

    let entries: Vec<_> = fs::read_dir(old_dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .collect();

    for entry in &entries {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            let dest = new_dir.join(path.file_name().unwrap());
            if let Err(e) = fs::rename(&path, &dest) {
                tracing::warn!("migrate: failed to move {}: {e}", path.display());
            } else {
                tracing::info!("migrate: moved {} → {}", path.display(), dest.display());
            }
        }
    }

    // Move any tokenizer.json files that sit alongside the models.
    for entry in &entries {
        let path = entry.path();
        if path.file_name().and_then(|n| n.to_str()) == Some("tokenizer.json") {
            let dest = new_dir.join("tokenizer.json");
            let _ = fs::rename(&path, &dest);
        }
    }
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
