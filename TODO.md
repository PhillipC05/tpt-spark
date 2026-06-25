# TPT Spark — Project Forge TODO

## Phase 1: Foundation ✅
- [x] Initialize Tauri v2 project with TypeScript frontend
- [x] Define `LlmEngine` trait and `EngineHandle` type
- [x] Implement stub engine (compiles everywhere, streams mock tokens)
- [x] GGUF model scanner (`scan_models_dir`)
- [x] Tauri IPC commands: `list_models`, `get_models_dir`, `load_model`, `unload_model`, `get_loaded_model`, `run_inference`, `get_system_info`
- [x] Tauri Channel streaming (zero-copy token delivery to UI)
- [x] Dark-themed chat UI with sidebar (model selector, generation params)
- [x] Word-by-word streaming display with blinking cursor
- [x] Push initial commit to `claude/hopeful-maxwell-ytvggl`

## Phase 2: Engine Integration (wgpu + candle) ✅
- [x] Add `candle` (Hugging Face) as the AI math engine dependency
- [x] Implement `CandleEngine` struct satisfying `LlmEngine` trait
- [x] Parse GGUF file format and load weights via zero-copy `mmap`
- [x] Implement tokenizer (BPE / SentencePiece) in Rust
- [x] CPU inference via `candle` — prove single-binary pipeline end-to-end
- [ ] Test loading a real `.gguf` model (e.g. Llama 3.2 1B or Phi-3 mini)
- [ ] Validate streaming token output in the UI
- [x] Handle context length limits and truncation gracefully

## Phase 3: GPU Acceleration (wgpu) ✅ (core implementation)
- [x] Implement `wgpu` compute shader pipeline for matrix multiply (GEMM)
- [x] Write WGSL shaders for attention (causal, GQA), feed-forward (SwiGLU), RMS norm, RoPE, dequant
- [x] Pre-allocate `wgpu` GPU buffers for model weights on load (`wgpu_loader.rs`)
- [x] Stream weight chunks from `mmap` directly into VRAM (zero RAM copy)
- [x] Implement explicit `wgpu` buffer `.destroy()` on model unload
- [x] CPU fallback via `CandleEngine` when no GPU adapter is detected (`cpu_fallback.rs`)
- [ ] Test GPU dispatch on Windows (DirectX 12 / Vulkan)
- [ ] Test GPU dispatch on Linux (Vulkan)
- [ ] Test GPU dispatch on macOS (Metal)
- [ ] Benchmark VRAM usage and tokens/sec vs Ollama baseline

## Phase 4: Polish & Open Source Launch
- [x] Model management UI: download models from any direct URL (reqwest streaming + progress bar)
- [x] Model management UI: delete models from disk with confirmation dialog
- [x] Conversation history (persist across sessions to disk as JSON)
- [x] System prompt / persona support (collapsible panel, persisted per conversation)
- [x] Open source repository (README.md, CONTRIBUTING.md, LICENSE)
- [x] Optimize frontend bundle size — removed 3 unused plugin packages; esnext target; manual chunk split (tauri-api isolated); CSS minify; total JS 11.7 kB → 11.7 kB split into cacheable chunks
- [x] App icon and branding assets — custom lightning bolt SVG → all sizes generated via `tauri icon` (PNG, ICO, ICNS, APPX, Android, iOS)
- [x] Bump to v1.0.0 (package.json, Cargo.toml, tauri.conf.json)
- [ ] Release v1.0 binaries (`.exe`, `.app`, `.AppImage`) — run `npm run tauri build` on each target platform
- [ ] Publish to GitHub Releases
