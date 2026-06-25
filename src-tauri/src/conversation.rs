//! Conversation history: save/load/list/delete chat sessions on disk.
//!
//! Each conversation is a JSON file in `{data_dir}/tpt-spark/history/<id>.json`.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

fn validate_conv_id(id: &str) -> Result<()> {
    if id.is_empty() || id.contains(['/', '\\', '.']) {
        anyhow::bail!("invalid conversation id: {id:?}");
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub messages: Vec<ConversationMessage>,
    pub model_name: String,
    pub system_prompt: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

pub fn history_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("tpt-spark").join("history")
}

pub fn save_conversation(dir: &Path, conv: &Conversation) -> Result<()> {
    validate_conv_id(&conv.id)?;
    fs::create_dir_all(dir).context("creating history dir")?;
    let path = dir.join(format!("{}.json", conv.id));
    let json = serde_json::to_string_pretty(conv)?;
    fs::write(path, json)?;
    Ok(())
}

pub fn list_conversations(dir: &Path) -> Result<Vec<Conversation>> {
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut convs = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            match fs::read_to_string(&path).and_then(|s| {
                serde_json::from_str::<Conversation>(&s)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            }) {
                Ok(c) => convs.push(c),
                Err(e) => tracing::warn!("Skipping malformed conversation {:?}: {}", path, e),
            }
        }
    }
    convs.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(convs)
}

pub fn load_conversation(dir: &Path, id: &str) -> Result<Conversation> {
    validate_conv_id(id)?;
    let path = dir.join(format!("{}.json", id));
    let json = fs::read_to_string(&path)
        .with_context(|| format!("reading conversation {id}"))?;
    Ok(serde_json::from_str(&json)?)
}

pub fn delete_conversation(dir: &Path, id: &str) -> Result<()> {
    validate_conv_id(id)?;
    let path = dir.join(format!("{}.json", id));
    fs::remove_file(path).with_context(|| format!("deleting conversation {id}"))?;
    Ok(())
}
