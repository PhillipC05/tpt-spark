# tpt-spark Integration Todos

These items were identified by analysing the three-repo TPT AI compute suite (tpt-gpu, tpt-spark, tpt-crucible) for cross-repo synergies. None of these are required for tpt-spark to work standalone — they are optional improvements that strengthen the suite.

---

## 1. Add `TptGpuEngine` Cargo feature (depends on tpt-gpu item 3)

**Why:** The current `WgpuEngine` uses hand-written WGSL shaders. tpt-gpu's Layer 4 runtime benchmarks above cuBLAS for GEMM and matches FlashAttention v2 for attention. Replacing WGSL shaders with the tpt-gpu runtime would give Spark production-quality inference without maintaining custom GPU kernels.

**What to do:**
- Wait for tpt-gpu to publish `tpt-gpu-runtime` crate (see tpt-gpu `todo1.md` item 3)
- Add a `tpt-gpu-engine` Cargo feature to `Cargo.toml`
- Implement `LlmEngine` for a new `TptGpuEngine` struct that wraps `tpt_gpu_runtime::LlmInference`
- Add engine selection logic: if `tpt-gpu-engine` feature is enabled and a supported GPU is detected, use `TptGpuEngine`; otherwise fall back to `WgpuEngine` → `CandleEngine`
- Add a CI matrix entry that builds with `--features tpt-gpu-engine`

---

## 2. Define a stable headless IPC API

**Why:** tpt-crucible already references "Spark IPC" as an optional LLM backend for AI-assisted design (driver generation, RTL synthesis hints, topology advising). The IPC protocol is currently implicit. Formalising it allows Crucible — and any other tool — to use Spark as a headless local inference engine without the GUI.

**What to do:**
- Add a `--headless` CLI flag (or a separate `spark-server` binary) that skips the Tauri GUI and starts a Unix socket (Linux/macOS) or named pipe (Windows) server
- Define a minimal JSON-RPC 2.0 protocol:
  - `spark_listModels` → `[{ name, arch, size_gb }]`
  - `spark_loadModel` `{ name }` → `{ ok }`
  - `spark_infer` `{ prompt, system_prompt?, max_tokens? }` → streaming `{ token }` events + `{ done }` final event
  - `spark_cancel` → `{ ok }`
- Write a `HEADLESS_API.md` spec document at the repo root
- Implement a minimal Rust JSON-RPC server in `src-tauri/src/headless.rs` using `tokio` (already a dependency)
- Publish the socket path convention: `$XDG_RUNTIME_DIR/tpt-spark.sock` on Linux, `\\.\pipe\tpt-spark` on Windows, `/tmp/tpt-spark.sock` on macOS

---

## 3. Adopt shared model registry (`~/.tpt/models/`)

**Why:** tpt-gpu and tpt-crucible will both adopt `~/.tpt/models/` as the canonical GGUF directory (see their respective `todo1.md` files). Aligning Spark to the same convention means models downloaded via Spark's UI are immediately available for Crucible compilation and tpt-gpu benchmarks — no duplicate downloads.

**What to do:**
- Change the default model scan directory from the current OS-specific path to `~/.tpt/models/`
- On first launch (when `~/.tpt/models/` doesn't exist), create it and migrate any models from the old directory
- When downloading a model, write it to `~/.tpt/models/` and update `~/.tpt/models/models.json`
- Read `models.json` on startup to pre-populate the model list without rescanning the filesystem
- Add a Settings UI option to override the model directory (for users who want a different path or a shared network drive)

---

## 4. Benchmark export for Crucible baseline comparison

**Why:** tpt-crucible's emulator uses baseline GPU benchmarks from Spark to validate that compiled edge targets (FPGA, MCU swarm, analog) are within expected performance bounds relative to a known GPU reference.

**What to do:**
- After each inference run, write a benchmark record to `~/.tpt/benchmarks/spark-{date}.json`: `{ model, tokens_per_second, time_to_first_token_ms, prompt_tokens, completion_tokens, engine, gpu_name }`
- Expose this via the headless IPC as `spark_lastBenchmark` → the most recent record
- Document the schema in `HEADLESS_API.md` so Crucible can read it without tight coupling
