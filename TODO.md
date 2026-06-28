# TPT Spark — TODO

## Engine: Quantization

- [x] Wire up all quantization formats via dtype-aware dequantization dispatch
  - Q4_K (rewritten shader + wired up via GpuTensor dtype dispatch)
  - Q4_0, Q5_0, Q5_1 — new shaders, wired
  - Q2_K, Q3_K, Q5_K, Q6_K — new shaders, wired
  - F16 → F32 conversion shader, wired
  - F32 passthrough (buffer copy, no shader)

## Engine: Architecture & Correctness

- [x] Add fused QKV support (`attn_qkv.weight` — Phi-3, MiMo, some Qwen3)
- [x] Add missing architectures: starcoder2, phi2, solar, baichuan, baichuan2, grok, falcon
- [x] Read RMSNorm epsilon from GGUF metadata (currently hardcoded 1e-5)
- [x] Support attention bias terms (`attn_q.bias`, `attn_output.bias`, etc.)
- [x] Read RoPE scaling metadata (linear + NTK/YaRN; linear passes scale through rope.wgsl, NTK bakes into freq_base at load time)

## Engine: Features

- [x] Multi-turn KV cache (prompt-prefix caching — skip re-running tokens already in GPU cache)
  - `cached_tokens: Vec<u32>` in `KvCache` tracks what's resident in GPU memory
  - Each `infer()` finds the longest matching prefix and starts prefill from there
  - No IPC changes — frontend still sends the full prompt; backend skips the matching prefix silently
  - Cancel `done: true` bug fixed: UI `isGenerating` now resets correctly on stop

## Frontend

- [x] Show model load progress (channel-based per-tensor progress from loader → command → frontend status bar)
- [x] Conversation history / multi-turn UI (save/load/delete conversations, buildHistoryPrompt for context)
