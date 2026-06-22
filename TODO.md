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

## Phase 2: Engine Integration (wgpu + candle)
- [ ] Add `candle` (Hugging Face) as the AI math engine dependency
- [ ] Implement `CandleEngine` struct satisfying `LlmEngine` trait
- [ ] Parse GGUF file format and load weights via zero-copy `mmap`
- [ ] Implement tokenizer (BPE / SentencePiece) in Rust
- [ ] CPU inference via `candle` — prove single-binary pipeline end-to-end
- [ ] Test loading a real `.gguf` model (e.g. Llama 3.2 1B or Phi-3 mini)
- [ ] Validate streaming token output in the UI
- [ ] Handle context length limits and truncation gracefully

## Phase 3: GPU Acceleration (wgpu)
- [ ] Implement `wgpu` compute shader pipeline for matrix multiply (GEMM)
- [ ] Write WGSL shaders for attention, feed-forward, and RMS norm layers
- [ ] Pre-allocate `wgpu` GPU buffers for model weights on load
- [ ] Stream weight chunks from `mmap` directly into VRAM (zero RAM copy)
- [ ] Test GPU dispatch on Linux (Vulkan)
- [ ] Test GPU dispatch on macOS (Metal)
- [ ] Test GPU dispatch on Windows (DirectX 12 / Vulkan)
- [ ] Implement explicit `wgpu` buffer `.destroy()` on model unload
- [ ] CPU fallback via AVX/NEON when no GPU is detected
- [ ] Benchmark VRAM usage and tokens/sec vs Ollama baseline

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
