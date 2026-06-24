# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install JS deps
npm install

# Dev mode (hot-reload frontend + Rust backend watch)
npm run tauri dev

# Release build (output: src-tauri/target/release/bundle/)
npm run tauri build

# Build with real candle CPU inference (default is stub engine)
npm run tauri build -- --features engine-candle

# Rust-only checks (faster than full Tauri build)
cd src-tauri && cargo check
cd src-tauri && cargo check --features engine-candle
cd src-tauri && cargo clippy --features engine-candle
cd src-tauri && cargo test
```

There are no frontend tests currently. No linter is configured beyond TypeScript type-checking via `tsc`.

## Architecture

TPT Spark is a Tauri v2 desktop app: a TypeScript/Vite frontend rendered in the OS WebView, backed by a Rust process.

### Feature-gated engines

The Rust backend selects an engine at **compile time** via Cargo features:

| Feature flag | Engine | Notes |
|---|---|---|
| `engine-stub` *(default)* | `StubEngine` | Echoes mock tokens, no native deps |
| `engine-candle` | `CandleEngine` | Real GGUF CPU inference via HuggingFace candle |

`default_engine()` in [src-tauri/src/engine/mod.rs](src-tauri/src/engine/mod.rs) constructs the active engine at startup. Adding a new backend means implementing `LlmEngine`, gating it behind a new feature, and wiring it in there.

### `LlmEngine` trait

All engines implement this trait (`engine/mod.rs`):
- `load(&mut self, path)` — read model into memory, parse metadata
- `unload(&mut self)` — free memory
- `infer(&self, params, on_token)` — blocking, synchronous; calls `on_token` for each generated token

`EngineHandle = Arc<tokio::sync::Mutex<Box<dyn LlmEngine>>>` is stored in Tauri state and shared across commands.

### IPC / streaming

Tauri commands in [src-tauri/src/commands.rs](src-tauri/src/commands.rs) are the only bridge between frontend and backend. Inference uses a `tauri::ipc::Channel<StreamEvent>` for zero-copy token streaming. CPU-bound calls (`load_model`, `run_inference`) use `tokio::task::block_in_place` to keep the async runtime responsive.

### CandleEngine inference design

`load()` reads the entire GGUF file into `Vec<u8>` (warm page cache). Each `infer()` call re-parses the header and reconstructs `ModelWeights` from those bytes — this gives a fresh KV cache per request at the cost of re-parsing overhead. Phase 3 will replace this with mmap + wgpu.

**Tokenizer**: a `tokenizer.json` (HuggingFace format) must sit next to the `.gguf` file. `find_tokenizer()` checks the same directory and a subdirectory matching the model stem.

**Supported architectures**: `llama` and `mistral` only — validated from the `general.architecture` GGUF metadata key.

### Model discovery

[src-tauri/src/models/mod.rs](src-tauri/src/models/mod.rs) scans a platform-specific directory for `.gguf` files:
- Windows: `%LOCALAPPDATA%\tpt-spark\models`
- macOS/Linux: `~/.local/share/tpt-spark/models`

### Frontend

Plain TypeScript + Vite, no framework. Calls Tauri commands via `@tauri-apps/api/core`. The channel listener in the frontend appends tokens word-by-word to the chat view as they stream in.

## Key constraints

- `tokenizer.json` must be placed next to any `.gguf` model file — the engine will fail with a clear error if it is missing.
- Only `llama` and `mistral` GGUF architectures are supported by `CandleEngine`; other families need explicit additions to the match in `candle_engine.rs`.
- The candle crates (`candle-core`, `candle-transformers`, `tokenizers`) add significant compile time and are only pulled in under `--features engine-candle`.
