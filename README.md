# ⚡ TPT Spark — Project Forge

A lean, native, cross-platform LLM runtime. No daemons. No HTTP overhead. No Python.

Load a `.gguf` model and start chatting — everything runs in a single binary, fully on-device.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Features

- **GPU inference** via [wgpu](https://wgpu.rs/) (Vulkan / Metal / DirectX 12) with custom WGSL compute shaders
- **CPU fallback** via [HuggingFace candle](https://github.com/huggingface/candle) when no GPU adapter is found
- **Zero-copy mmap** — model weights stream from disk directly into VRAM; no RAM copy
- **Real-time token streaming** — tokens appear word-by-word as they are generated
- **Stop generation** — cancel an in-flight inference at any time
- **Conversation history** — sessions persist to disk as JSON and can be resumed
- **System prompt / persona** — configurable per conversation
- **In-app model download** — fetch GGUF files directly from any HTTPS URL
- **Fully local** — no telemetry, no cloud, no network required after download
- Single binary built with **Rust + Tauri v2** — ~10 MB app overhead

---

## Supported model architectures

`llama` and `mistral` GGUF files (quantized or full precision). Tested with:

- LLaMA 3 (1B, 3B, 8B)
- Mistral 7B
- Phi-3 Mini

---

## Quick start

### 1. Prerequisites

| Requirement | Version |
|---|---|
| [Rust](https://rustup.rs/) | 1.77+ |
| [Node.js](https://nodejs.org/) | 20+ |
| Platform libs | [Tauri prerequisites](https://tauri.app/start/prerequisites/) |

### 2. Build

```bash
git clone https://github.com/PhillipC05/tpt-spark
cd tpt-spark
npm install

# Dev mode — hot-reload frontend + Rust watch
npm run tauri dev

# Release binary — stub engine (UI smoke-test, no AI)
npm run tauri build

# Release binary — wgpu GPU engine (recommended for real inference)
npm run tauri build -- --features engine-wgpu
```

### 3. Add a model

Place a `.gguf` file **and its matching `tokenizer.json`** in the models directory:

| OS | Path |
|---|---|
| Windows | `%LOCALAPPDATA%\tpt-spark\models\` |
| macOS | `~/Library/Application Support/tpt-spark/models/` |
| Linux | `~/.local/share/tpt-spark/models/` |

Or use the **Download Model** panel in the app sidebar to fetch a GGUF directly by HTTPS URL.

> **Note**: `tokenizer.json` must sit next to the `.gguf` file. Download it from the same HuggingFace model repository.

### 4. Chat

1. Open the app, select your model from the sidebar dropdown, click **Load**.
2. Weights upload to VRAM (GPU path) or RAM (CPU fallback) — a few seconds for a 4B model.
3. Type a message and press **Enter** to start chatting.
4. Click **Stop** to cancel generation at any time.

---

## Engine feature flags

| Cargo feature | Engine | Notes |
|---|---|---|
| `engine-stub` *(default)* | `StubEngine` | Echoes mock tokens — compiles everywhere, no native deps |
| `engine-candle` | `CandleEngine` | Real GGUF CPU inference via HuggingFace candle |
| `engine-wgpu` | `WgpuEngine` | GPU inference via wgpu + WGSL shaders; falls back to candle if no GPU |

---

## Architecture

```
Frontend (TypeScript + Vite)
    ↕  Tauri IPC (Channel — zero-copy token streaming)
Backend (Rust)
    ├── engine/mod.rs         LlmEngine trait + EngineHandle (Arc<Mutex<...>>)
    ├── engine/wgpu_engine    GPU path: mmap → VRAM → WGSL kernels
    ├── engine/candle_engine  CPU path: HuggingFace candle GGUF
    ├── engine/stub           Mock echo (default)
    ├── engine/shaders/       WGSL: gemm, attention, rope, rms_norm, silu, dequant
    ├── commands.rs           Tauri IPC commands
    ├── conversation.rs       History persistence (JSON files)
    └── models/               GGUF directory scanner
```

---

## Why not Ollama?

Ollama wraps llama.cpp in a **Go HTTP daemon**. TPT Spark replaces the HTTP layer with Tauri IPC
and the Go daemon with a Rust process — cutting ~100 MB RAM overhead and the requirement for a
running background service while keeping the same GGUF model support.

---

## License

[Apache 2.0](LICENSE) — © 2024 TPT Solutions
