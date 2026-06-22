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

## Phase 2: Engine Integration
- [ ] Add `llama-cpp-2` Rust bindings behind `engine-llama` feature flag
- [ ] Implement `LlamaEngine` struct satisfying `LlmEngine` trait
- [ ] CPU-only inference — prove the single-binary pipeline works end-to-end
- [ ] Test loading a real `.gguf` model (e.g. Llama 3.2 1B or Phi-3 mini)
- [ ] Validate streaming token output in the UI
- [ ] Handle context length limits and truncation gracefully

## Phase 3: GPU Acceleration
- [ ] Configure `llama.cpp` Vulkan backend via `wgpu` / Rust
- [ ] Test VRAM allocation on Linux (Vulkan)
- [ ] Test VRAM allocation on macOS (Metal)
- [ ] Test VRAM allocation on Windows (DirectX 12 / Vulkan)
- [ ] Implement explicit `wgpu` buffer `.destroy()` on model unload
- [ ] Benchmark VRAM vs RAM usage vs Ollama baseline
- [ ] Implement zero-copy `mmap` streaming from SSD → GPU buffer

## Phase 4: Polish & Open Source Launch
- [ ] Model management UI: download models from HuggingFace Hub
- [ ] Model management UI: delete models from disk with confirmation
- [ ] Conversation history (persist across sessions to disk)
- [ ] System prompt / persona support
- [ ] Optimize frontend bundle size (tree-shake unused CSS/JS)
- [ ] App icon and branding assets
- [ ] CI pipeline (GitHub Actions): build on Windows, macOS, Linux
- [ ] Release v1.0 binaries (`.exe`, `.app`, `.AppImage`)
- [ ] Open source repository (README, CONTRIBUTING, LICENSE)
- [ ] Publish to GitHub Releases
