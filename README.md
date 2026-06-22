# TPT Spark — Project Forge

> A lean, native, cross-platform LLM runtime. No daemons. No HTTP overhead. No proprietary AI drivers.

**TPT Spark** is an open-source desktop application for running Large Language Models locally.
It compiles to a **single binary** and runs on Windows, macOS, and Linux using standard display
drivers via `wgpu` — no CUDA or ROCm required.

---

## Architecture

| Layer | Technology | Purpose |
|---|---|---|
| UI / Frontend | TypeScript + Vite | Chat interface rendered by OS WebView (WebView2 / WebKit) |
| App Core / Bridge | Rust + Tauri v2 | Window management, OS integration, IPC |
| Async Runtime | Tokio | Non-blocking, zero-copy weight streaming |
| Compute Backend | `wgpu` / Vulkan / Metal / DirectX 12 | GPU dispatch without CUDA |
| AI Math Engine | `llama-cpp-rs` (V1) / `candle` (V2) | Optimized GGUF inference |

### Data Flow

```
User prompt → Tauri IPC → Rust tokenizer → wgpu GPUBuffer
    → Vulkan / Metal compute shaders → predicted token
    → IPC stream → TS frontend (word-by-word)
```

---

## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) 1.77+
- [Node.js](https://nodejs.org/) 18+
- Platform system libs:
  - **Linux**: `libgtk-3-dev libwebkit2gtk-4.1-dev librsvg2-dev`
  - **macOS**: Xcode Command Line Tools
  - **Windows**: WebView2 (ships with Windows 11)

### Run in development

```bash
npm install
npm run tauri dev
```

### Build a release binary

```bash
npm run tauri build
```

The output binary lives in `src-tauri/target/release/bundle/`.

---

## Adding Models

1. Open the app and note the **Models directory** shown in the sidebar.
2. Copy any `.gguf` model file into that directory.
3. Click **⟳ Refresh** in the sidebar.
4. Select the model and click **Load**.

Popular GGUF sources: [HuggingFace](https://huggingface.co/models?library=gguf)

---

## Engine Backends

| Feature flag | Backend | Status |
|---|---|---|
| `engine-stub` *(default)* | Mock streaming | Compiles everywhere, no native deps |
| `engine-llama` | llama.cpp via `llama-cpp-2` | Full GGUF inference |

Enable real inference:

```bash
npm run tauri build -- --features engine-llama
```

> Requires `cmake` and a C++ compiler for the llama.cpp native build.

---

## Roadmap

- [x] Phase 1 — Foundation: Tauri v2 project, Rust backend, IPC streaming
- [ ] Phase 2 — Engine Integration: llama-cpp-rs CPU inference
- [ ] Phase 3 — GPU Acceleration: wgpu Vulkan / Metal dispatch, VRAM management
- [ ] Phase 4 — Polish: model download manager, bundle size optimisation, v1.0 release

---

## Why not Ollama?

Ollama wraps llama.cpp in a **Go HTTP server + daemon**. TPT Spark wraps it directly in Rust,
stripping out the network stack, daemon process, and ~100 MB RAM overhead while keeping the same
GGUF model support and CPU/GPU inference speed.

## License

MIT
