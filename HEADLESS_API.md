# TPT Spark Headless API

TPT Spark can run without its GUI as a local LLM inference server. This is the
protocol used by tpt-crucible (and any other tool in the TPT suite) to drive
Spark programmatically.

---

## Starting headless mode

```bash
# CLI flag
tpt-spark --headless

# Environment variable (useful from scripts / service managers)
TPT_SPARK_HEADLESS=1 tpt-spark
```

---

## Transport

| Platform | Socket type    | Address                                       |
|----------|----------------|-----------------------------------------------|
| Windows  | Named pipe     | `\\.\pipe\tpt-spark`                          |
| macOS    | Unix domain    | `/tmp/tpt-spark.sock`                         |
| Linux    | Unix domain    | `$XDG_RUNTIME_DIR/tpt-spark.sock`            |

Connect with any socket client. The stale socket file on Linux/macOS is removed
automatically on startup.

---

## Protocol

**Newline-delimited JSON-RPC 2.0.**

Each client message is one JSON object on a single line followed by `\n`.
Each server response is one or more JSON objects (one per line) followed by `\n`.

Standard request format:
```json
{"jsonrpc":"2.0","id":1,"method":"spark_listModels","params":null}
```

Standard success response:
```json
{"jsonrpc":"2.0","id":1,"result":[...]}
```

Standard error response:
```json
{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"no model loaded"}}
```

---

## Methods

### `spark_listModels`

Returns the models available in `~/.tpt/models/`.

**Request params:** `null`

**Result:** array of model descriptors

```json
[
  { "name": "llama-3-8b-q4_k_m", "arch": null, "size_gb": 4.68 }
]
```

---

### `spark_loadModel`

Loads a model into memory. `name` is the bare filename (without `.gguf` extension)
or the full path returned by `spark_listModels`.

**Request params:**
```json
{ "name": "llama-3-8b-q4_k_m" }
```

**Result:**
```json
{ "ok": true }
```

---

### `spark_infer`

Runs inference and streams tokens back. This method sends **multiple response
lines** with the same `id` before the final `done:true` line.

**Request params:**
```json
{
  "prompt": "Explain the difference between RAM and storage.",
  "system_prompt": "You are a helpful assistant.",
  "max_tokens": 256
}
```

`system_prompt` and `max_tokens` are optional (defaults: none, 512).

**Streaming token events** (one per line, emitted during generation):
```json
{"jsonrpc":"2.0","id":1,"result":{"token":"The","done":false}}
{"jsonrpc":"2.0","id":1,"result":{"token":" difference","done":false}}
...
```

**Final event** (signals end of stream):
```json
{"jsonrpc":"2.0","id":1,"result":{"token":"","done":true}}
```

Use `spark_cancel` to abort an in-progress generation.

---

### `spark_cancel`

Cancels the currently running `spark_infer` call. Safe to call even when no
inference is running.

**Request params:** `null`

**Result:**
```json
{ "ok": true }
```

---

### `spark_lastBenchmark`

Returns the most recent Crucible-compatible benchmark record written to
`~/.tpt/benchmarks/spark-{date}.json`, or `null` if none exists.

**Request params:** `null`

**Result:**
```json
{
  "model": "llama-3-8b-q4_k_m",
  "tokens_per_second": 28.4,
  "time_to_first_token_ms": 312,
  "prompt_tokens": 14,
  "completion_tokens": 128,
  "engine": "wgpu",
  "gpu_name": "NVIDIA GeForce RTX 3080",
  "timestamp": "2026-06-29T04:12:00Z"
}
```

---

## Benchmark export schema

After every benchmark run, Spark writes a record to:

```
~/.tpt/benchmarks/spark-{YYYY-MM-DD}.json
```

One file per calendar day; each file contains the single most recent record for
that day (overwritten on each run). tpt-crucible reads this file to establish a
GPU reference baseline for compiled edge-target validation.

Schema (all fields always present):

| Field                    | Type    | Description                                      |
|--------------------------|---------|--------------------------------------------------|
| `model`                  | string  | Model stem name                                  |
| `tokens_per_second`      | float   | Decode throughput (completion tokens / decode s) |
| `time_to_first_token_ms` | integer | Prefill latency in milliseconds                  |
| `prompt_tokens`          | integer | Approximate prompt token count                   |
| `completion_tokens`      | integer | Tokens generated in the decode phase             |
| `engine`                 | string  | `"wgpu"`, `"candle-cpu"`, `"tpt-gpu"`, `"stub"` |
| `gpu_name`               | string  | GPU adapter name, or `"cpu"` for CPU-only        |
| `timestamp`              | string  | RFC 3339 UTC timestamp                           |

---

## Error codes

| Code    | Meaning                                      |
|---------|----------------------------------------------|
| -32700  | Parse error — request was not valid JSON     |
| -32601  | Method not found                             |
| -32602  | Invalid params                               |
| -32000  | Server error (model not loaded, I/O failure) |
