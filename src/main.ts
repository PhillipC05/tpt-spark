import { invoke, Channel } from "@tauri-apps/api/core";

// ── Types ─────────────────────────────────────────────────────────────────────

interface ModelEntry {
  name: string;
  filename: string;
  path: string;
  sizeBytes: number;
  sizeHuman: string;
}

interface ModelInfo {
  name: string;
  path: string;
  sizeBytes: number;
  backend: string;
}

interface StreamEvent {
  token: string;
  done: boolean;
  tokensGenerated: number | null;
  ttftMs: number | null;
  prefillMs: number | null;
  decodeMs: number | null;
  tokensPerSec: number | null;
}

interface BenchmarkResult {
  id: string;
  modelName: string;
  backend: string;
  modelSizeBytes: number;
  promptTokens: number;
  promptLabel: string;
  tokensGenerated: number;
  prefillMs: number;
  decodeMs: number;
  totalMs: number;
  tokensPerSec: number;
  toksPerSecPerGb: number;
  ttftMs: number;
  timestamp: string;
}

interface DownloadProgress {
  downloaded: number;
  total: number | null;
  done: boolean;
  error: string | null;
}

interface LoadProgress {
  tensorsDone: number;
  tensorsTotal: number;
  done: boolean;
}

interface SystemInfo {
  backend: string;
  engineLoaded: boolean;
  modelName: string | null;
}

interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

interface Conversation {
  id: string;
  title: string;
  messages: ConversationMessage[];
  modelName: string;
  systemPrompt: string | null;
  createdAt: string;
  updatedAt: string;
}

// ── State ─────────────────────────────────────────────────────────────────────

let isGenerating = false;
let currentConv: Conversation | null = null;
let loadedModelName = "";
let benchmarkResults: BenchmarkResult[] = [];
let selectedBenchmarkTokens = 64;

// ── DOM refs ──────────────────────────────────────────────────────────────────

const modelSelect      = q<HTMLSelectElement>("#model-select");
const modelStatus      = q<HTMLDivElement>("#model-status");
const modelStatusTxt   = q<HTMLSpanElement>("#model-status-text");
const modelMeta        = q<HTMLDivElement>("#model-meta");
const btnLoad          = q<HTMLButtonElement>("#btn-load");
const btnUnload        = q<HTMLButtonElement>("#btn-unload");
const btnDeleteModel   = q<HTMLButtonElement>("#btn-delete-model");
const btnRefresh       = q<HTMLButtonElement>("#btn-refresh");
const modelsDirEl      = q<HTMLDivElement>("#models-dir");
const backendBadge     = q<HTMLDivElement>("#backend-badge");

const promptInput      = q<HTMLTextAreaElement>("#prompt-input");
const btnSend          = q<HTMLButtonElement>("#btn-send");
const btnStop          = q<HTMLButtonElement>("#btn-stop");
const messagesEl       = q<HTMLDivElement>("#messages");
const statusBar        = q<HTMLDivElement>("#status-bar");
const welcomeScreen    = document.getElementById("welcome-screen");

const paramMaxTokens     = q<HTMLInputElement>("#param-max-tokens");
const paramTemperature   = q<HTMLInputElement>("#param-temperature");
const paramTopP          = q<HTMLInputElement>("#param-top-p");
const paramRepeatPenalty = q<HTMLInputElement>("#param-repeat-penalty");
const tempVal  = q<HTMLSpanElement>("#temp-val");
const toppVal  = q<HTMLSpanElement>("#topp-val");
const repVal   = q<HTMLSpanElement>("#rep-val");

const btnNewConv       = q<HTMLButtonElement>("#btn-new-conv");
const convListEl       = q<HTMLDivElement>("#conv-list");

const systemPromptInput = q<HTMLTextAreaElement>("#system-prompt");
const btnToggleSysPrompt = q<HTMLButtonElement>("#btn-toggle-sysprompt");
const syspromptPanel   = q<HTMLDivElement>("#sysprompt-panel");

const btnBrowseModels  = q<HTMLButtonElement>("#btn-browse-models");

const btnToggleDownload = q<HTMLButtonElement>("#btn-toggle-download");
const downloadPanel    = q<HTMLDivElement>("#download-panel");
const btnHF            = q<HTMLButtonElement>("#btn-hf");
const dlUrl            = q<HTMLInputElement>("#dl-url");
const dlFilename       = q<HTMLInputElement>("#dl-filename");
const btnDownload      = q<HTMLButtonElement>("#btn-download");
const dlProgressWrap   = q<HTMLDivElement>("#dl-progress-wrap");
const dlProgress       = q<HTMLProgressElement>("#dl-progress");
const dlProgressTxt    = q<HTMLSpanElement>("#dl-progress-txt");

const confirmDialog    = q<HTMLDialogElement>("#confirm-dialog");
const confirmText      = q<HTMLParagraphElement>("#confirm-text");
const confirmOk        = q<HTMLButtonElement>("#confirm-ok");
const confirmCancel    = q<HTMLButtonElement>("#confirm-cancel");

const btnToggleBenchmark  = q<HTMLButtonElement>("#btn-toggle-benchmark");
const benchmarkPanel      = q<HTMLDivElement>("#benchmark-panel");
const benchCustomPrompt   = q<HTMLTextAreaElement>("#bench-custom-prompt");
const btnRunBenchmark     = q<HTMLButtonElement>("#btn-run-benchmark");
const benchmarkStatus     = q<HTMLDivElement>("#benchmark-status");
const btnCopyResults      = q<HTMLButtonElement>("#btn-copy-results");
const benchmarkResultsEl  = q<HTMLDivElement>("#benchmark-results");

function q<T extends Element>(sel: string): T {
  return document.querySelector(sel) as T;
}

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  await Promise.all([loadSystemInfo(), loadModelsDir(), refreshModelList(), refreshConvList()]);

  paramTemperature.addEventListener("input", () => (tempVal.textContent = paramTemperature.value));
  paramTopP.addEventListener("input", () => (toppVal.textContent = paramTopP.value));
  paramRepeatPenalty.addEventListener("input", () => (repVal.textContent = paramRepeatPenalty.value));

  btnRefresh.addEventListener("click", refreshModelList);
  btnLoad.addEventListener("click", handleLoadModel);
  btnUnload.addEventListener("click", handleUnloadModel);
  btnDeleteModel.addEventListener("click", handleDeleteModel);
  btnSend.addEventListener("click", handleSend);
  btnStop.addEventListener("click", handleStop);
  btnNewConv.addEventListener("click", startNewConversation);

  btnBrowseModels.addEventListener("click", handleBrowseModels);

  btnToggleSysPrompt.addEventListener("click", () => togglePanel(syspromptPanel, btnToggleSysPrompt));
  btnToggleDownload.addEventListener("click", () => togglePanel(downloadPanel, btnToggleDownload));
  btnDownload.addEventListener("click", handleDownload);
  btnHF.addEventListener("click", () => invoke("open_external_url", { url: "https://huggingface.co/models?search=gguf&sort=downloads&library=gguf" }));

  modelSelect.addEventListener("change", () => {
    btnLoad.disabled = !modelSelect.value;
    btnDeleteModel.disabled = !modelSelect.value;
  });

  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  });

  promptInput.addEventListener("input", () => {
    promptInput.style.height = "auto";
    promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + "px";
  });

  confirmCancel.addEventListener("click", () => confirmDialog.close());

  btnToggleBenchmark.addEventListener("click", () => togglePanel(benchmarkPanel, btnToggleBenchmark));
  btnRunBenchmark.addEventListener("click", handleRunBenchmark);
  btnCopyResults.addEventListener("click", handleCopyBenchmarkResults);

  document.querySelectorAll<HTMLButtonElement>(".bench-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".bench-preset").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const tokens = parseInt(btn.dataset.tokens ?? "64", 10);
      selectedBenchmarkTokens = tokens;
      if (tokens === 0) {
        benchCustomPrompt.classList.remove("hidden");
      } else {
        benchCustomPrompt.classList.add("hidden");
      }
    });
  });

  await loadBenchmarkHistory();
}

// ── System info ───────────────────────────────────────────────────────────────

async function loadSystemInfo() {
  try {
    const info: SystemInfo = await invoke("get_system_info");
    backendBadge.textContent = `backend: ${info.backend}`;
    if (info.engineLoaded && info.modelName) {
      loadedModelName = info.modelName;
      setModelLoaded({ name: info.modelName, path: "", sizeBytes: 0, backend: info.backend });
    }
  } catch (e) {
    console.error("get_system_info failed", e);
  }
}

async function loadModelsDir() {
  try {
    const dir: string = await invoke("get_models_dir");
    modelsDirEl.textContent = dir;
  } catch {
    modelsDirEl.textContent = "unavailable";
  }
}

async function handleBrowseModels() {
  try {
    const dir: string | null = await invoke("pick_models_dir");
    if (dir) {
      modelsDirEl.textContent = dir;
      setStatus("Models directory updated. Refreshing…");
      await refreshModelList();
    }
  } catch (e) {
    setStatus(`Could not change models directory: ${e}`);
  }
}

// ── Model list ────────────────────────────────────────────────────────────────

async function refreshModelList() {
  setStatus("Scanning for models…");
  modelSelect.disabled = true;
  modelSelect.innerHTML = '<option value="">Scanning…</option>';
  btnLoad.disabled = true;
  btnDeleteModel.disabled = true;

  try {
    const models: ModelEntry[] = await invoke("list_models");
    modelSelect.innerHTML = "";

    if (models.length === 0) {
      modelSelect.innerHTML = '<option value="">No .gguf files found</option>';
      setStatus("No models found. Add .gguf files to the models directory.");
    } else {
      const ph = document.createElement("option");
      ph.value = "";
      ph.textContent = "— Select a model —";
      modelSelect.appendChild(ph);

      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m.path;
        opt.textContent = `${m.name} (${m.sizeHuman})`;
        modelSelect.appendChild(opt);
      }

      modelSelect.disabled = false;
      btnLoad.disabled = false;
      setStatus(`Found ${models.length} model${models.length !== 1 ? "s" : ""}.`);
    }
  } catch (e) {
    modelSelect.innerHTML = '<option value="">Error scanning</option>';
    setStatus(`Scan error: ${e}`);
  }
}

// ── Load / unload / delete model ─────────────────────────────────────────────

async function handleLoadModel() {
  const path = modelSelect.value;
  if (!path) return;

  setModelLoading();
  setStatus("Loading model… (uploading weights to VRAM)");

  const channel = new Channel<LoadProgress>();
  channel.onmessage = (ev) => {
    if (!ev.done && ev.tensorsTotal > 0) {
      const pct = Math.round((ev.tensorsDone / ev.tensorsTotal) * 100);
      setStatus(`Loading model… ${pct}% (${ev.tensorsDone}/${ev.tensorsTotal} tensors)`);
    }
  };

  try {
    const info: ModelInfo = await invoke("load_model", { path, channel });
    loadedModelName = info.name;
    setModelLoaded(info);
    setStatus(`Model loaded: ${info.name}`);
    enableChat();
  } catch (e) {
    setModelUnloaded();
    appendError(`Failed to load model: ${e}`);
    setStatus("Model load failed.");
  }
}

async function handleUnloadModel() {
  try { await invoke("unload_model"); } catch (e) { console.error("unload error", e); }
  loadedModelName = "";
  setModelUnloaded();
  disableChat();
  setStatus("Model unloaded.");
}

async function handleDeleteModel() {
  const path = modelSelect.value;
  if (!path) return;
  const name = modelSelect.options[modelSelect.selectedIndex]?.text ?? path;

  const confirmed = await confirmPrompt(`Delete "${name}" from disk? This cannot be undone.`);
  if (!confirmed) return;

  try {
    await invoke("delete_model", { path });
    setStatus(`Deleted ${name}.`);
    if (loadedModelName && name.startsWith(loadedModelName)) {
      loadedModelName = "";
      setModelUnloaded();
      disableChat();
    }
    await refreshModelList();
  } catch (e) {
    appendError(`Delete failed: ${e}`);
  }
}

function setModelLoading() {
  modelStatus.className = "model-status loading";
  modelStatusTxt.textContent = "Loading…";
  btnLoad.disabled = true;
  btnUnload.disabled = true;
  btnDeleteModel.disabled = true;
}

function setModelLoaded(info: ModelInfo) {
  modelStatus.className = "model-status loaded";
  modelStatusTxt.textContent = info.name;
  btnLoad.disabled = false;
  btnUnload.disabled = false;
  btnDeleteModel.disabled = !modelSelect.value;

  if (info.sizeBytes > 0) {
    const sizeGb = (info.sizeBytes / 1_073_741_824).toFixed(1);
    modelMeta.textContent = `Backend: ${info.backend} · Size: ${sizeGb} GB`;
    modelMeta.classList.remove("hidden");
  }
}

function setModelUnloaded() {
  modelStatus.className = "model-status not-loaded";
  modelStatusTxt.textContent = "No model loaded";
  btnLoad.disabled = !modelSelect.value;
  btnUnload.disabled = true;
  btnDeleteModel.disabled = !modelSelect.value;
  modelMeta.classList.add("hidden");
}

// ── Chat enable / disable ─────────────────────────────────────────────────────

function enableChat() {
  promptInput.disabled = false;
  btnSend.disabled = false;
  btnRunBenchmark.disabled = false;
  promptInput.placeholder = "Send a message… (Enter to send, Shift+Enter for newline)";
  promptInput.focus();
}

function disableChat() {
  promptInput.disabled = true;
  btnSend.disabled = true;
  btnRunBenchmark.disabled = true;
  promptInput.placeholder = "Load a model to start chatting…";
}

// ── Conversation history ──────────────────────────────────────────────────────

function startNewConversation() {
  currentConv = null;
  messagesEl.innerHTML = "";
  if (welcomeScreen) {
    const ws = welcomeScreen.cloneNode(true) as HTMLElement;
    ws.id = "welcome-screen";
    messagesEl.appendChild(ws);
  } else {
    const ws = document.createElement("div");
    ws.className = "welcome-screen";
    ws.innerHTML = `<div class="welcome-icon">⚡</div><h1 class="welcome-title">New conversation</h1>`;
    messagesEl.appendChild(ws);
  }
  setStatus("New conversation started.");
}

async function refreshConvList() {
  try {
    const convs: Conversation[] = await invoke("list_convs");
    convListEl.innerHTML = "";
    if (convs.length === 0) {
      convListEl.innerHTML = '<div class="conv-empty">No saved conversations</div>';
      return;
    }
    for (const c of convs) {
      const el = document.createElement("div");
      el.className = "conv-item";
      el.dataset.id = c.id;
      el.innerHTML = `
        <span class="conv-title" title="${escapeHtml(c.title)}">${escapeHtml(c.title)}</span>
        <span class="conv-model">${escapeHtml(c.modelName)}</span>
        <button class="conv-delete btn btn-ghost btn-xs" title="Delete conversation">×</button>
      `;
      el.querySelector(".conv-title")!.addEventListener("click", () => loadConversation(c.id));
      el.querySelector(".conv-delete")!.addEventListener("click", async (e) => {
        e.stopPropagation();
        const ok = await confirmPrompt(`Delete conversation "${c.title}"?`);
        if (!ok) return;
        await invoke("delete_conv", { id: c.id });
        if (currentConv?.id === c.id) startNewConversation();
        await refreshConvList();
      });
      convListEl.appendChild(el);
    }
  } catch (e) {
    console.error("list_convs failed", e);
  }
}

async function loadConversation(id: string) {
  try {
    const conv: Conversation = await invoke("load_conv", { id });
    currentConv = conv;

    messagesEl.innerHTML = "";
    for (const msg of conv.messages) {
      if (msg.role === "user") appendUserMessage(msg.content);
      else appendAssistantMessageStatic(msg.content);
    }

    if (conv.systemPrompt) {
      systemPromptInput.value = conv.systemPrompt;
      showPanel(syspromptPanel, btnToggleSysPrompt);
    }

    setStatus(`Loaded: ${conv.title}`);
    scrollToBottom();
  } catch (e) {
    appendError(`Failed to load conversation: ${e}`);
  }
}

async function persistConversation() {
  if (!currentConv) return;
  currentConv.updatedAt = new Date().toISOString();
  try {
    await invoke("save_conv", { conversation: currentConv });
    await refreshConvList();
  } catch (e) {
    console.error("save_conv failed", e);
  }
}

// ── Download ──────────────────────────────────────────────────────────────────

async function handleDownload() {
  const url = dlUrl.value.trim();
  const filename = dlFilename.value.trim();
  if (!url || !filename) {
    setStatus("Please enter a URL and filename.");
    return;
  }

  btnDownload.disabled = true;
  dlProgressWrap.classList.remove("hidden");
  dlProgress.value = 0;
  dlProgressTxt.textContent = "0%";
  setStatus("Downloading…");

  const channel = new Channel<DownloadProgress>();
  channel.onmessage = (ev) => {
    if (ev.done) {
      dlProgress.value = 100;
      dlProgressTxt.textContent = "Done";
      btnDownload.disabled = false;
      setStatus(`Downloaded ${filename}`);
      refreshModelList();
      setTimeout(() => dlProgressWrap.classList.add("hidden"), 2000);
    } else {
      const pct = ev.total ? Math.round((ev.downloaded / ev.total) * 100) : -1;
      if (pct >= 0) {
        dlProgress.value = pct;
        dlProgressTxt.textContent = `${pct}%`;
      } else {
        const mb = (ev.downloaded / 1_048_576).toFixed(1);
        dlProgressTxt.textContent = `${mb} MB`;
      }
    }
  };

  try {
    await invoke("download_model", { url, filename, channel });
  } catch (e) {
    appendError(`Download failed: ${e}`);
    btnDownload.disabled = false;
    setStatus("Download failed.");
  }
}

// ── Inference ─────────────────────────────────────────────────────────────────

async function handleSend() {
  if (isGenerating) return;
  const userText = promptInput.value.trim();
  if (!userText) return;

  promptInput.value = "";
  promptInput.style.height = "auto";
  isGenerating = true;
  btnSend.disabled = true;
  btnStop.classList.remove("hidden");

  hideWelcome();
  appendUserMessage(userText);
  const { bubble, statsBar } = appendAssistantMessage();
  setStatus("Generating…");

  // Build up the full conversation prompt from history.
  const historyPrompt = buildHistoryPrompt(userText);

  const channel = new Channel<StreamEvent>();
  let assistantContent = "";

  channel.onmessage = (event) => {
    if (event.done) {
      bubble.classList.remove("streaming");
      renderInferenceStats(statsBar, event);
      isGenerating = false;
      btnSend.disabled = false;
      btnStop.classList.add("hidden");
      setStatus("");

      // Persist the turn.
      if (!currentConv) {
        currentConv = newConversation(loadedModelName, systemPromptInput.value.trim() || null);
      }
      currentConv.messages.push(
        { role: "user", content: userText, timestamp: new Date().toISOString() },
        { role: "assistant", content: assistantContent, timestamp: new Date().toISOString() },
      );
      if (currentConv.messages.length === 2) autoTitle(currentConv, userText);
      persistConversation();
    } else {
      assistantContent += event.token;
      bubble.textContent += event.token;
      scrollToBottom();
    }
  };

  const systemPrompt = systemPromptInput.value.trim() || undefined;

  try {
    await invoke("run_inference", {
      prompt: historyPrompt,
      systemPrompt,
      maxTokens: parseInt(paramMaxTokens.value, 10),
      temperature: parseFloat(paramTemperature.value),
      topP: parseFloat(paramTopP.value),
      repeatPenalty: parseFloat(paramRepeatPenalty.value),
      channel,
    });
  } catch (e) {
    bubble.classList.remove("streaming");
    bubble.textContent = "";
    appendError(`Inference error: ${e}`);
    isGenerating = false;
    btnSend.disabled = false;
    btnStop.classList.add("hidden");
    setStatus("Error during generation.");
  }
}

async function handleStop() {
  try {
    await invoke("cancel_inference");
    setStatus("Generation stopped.");
  } catch (e) {
    console.error("cancel_inference failed", e);
  }
}

function buildHistoryPrompt(latestUserText: string): string {
  if (!currentConv || currentConv.messages.length === 0) {
    return latestUserText;
  }
  const lines: string[] = [];
  for (const m of currentConv.messages) {
    lines.push(m.role === "user" ? `User: ${m.content}` : `Assistant: ${m.content}`);
  }
  lines.push(`User: ${latestUserText}`);
  lines.push("Assistant:");
  return lines.join("\n");
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

function hideWelcome() {
  document.getElementById("welcome-screen")?.remove();
}

function appendUserMessage(text: string) {
  const msg = document.createElement("div");
  msg.className = "message user";
  msg.innerHTML = `
    <div class="message-avatar">U</div>
    <div class="message-bubble">${escapeHtml(text)}</div>
  `;
  messagesEl.appendChild(msg);
  scrollToBottom();
}

function appendAssistantMessage(): { bubble: HTMLDivElement; statsBar: HTMLDivElement } {
  const msg = document.createElement("div");
  msg.className = "message assistant";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = "⚡";

  const content = document.createElement("div");
  content.className = "message-content";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble streaming";

  const statsBar = document.createElement("div");
  statsBar.className = "inference-stats hidden";

  content.appendChild(bubble);
  content.appendChild(statsBar);
  msg.appendChild(avatar);
  msg.appendChild(content);
  msg.appendChild(makeCopyButton(() => bubble.textContent ?? ""));

  messagesEl.appendChild(msg);
  scrollToBottom();
  return { bubble, statsBar };
}

function appendAssistantMessageStatic(text: string) {
  const msg = document.createElement("div");
  msg.className = "message assistant";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  msg.innerHTML = `<div class="message-avatar">⚡</div>`;
  msg.appendChild(bubble);
  msg.appendChild(makeCopyButton(() => text));
  messagesEl.appendChild(msg);
}

function makeCopyButton(getText: () => string): HTMLButtonElement {
  const btn = document.createElement("button");
  btn.className = "copy-btn";
  btn.title = "Copy to clipboard";
  btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(getText());
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`;
      setTimeout(() => {
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
      }, 1500);
    } catch {
      // Clipboard API may not be available in all contexts
    }
  });
  return btn;
}

function appendError(text: string) {
  const el = document.createElement("div");
  el.className = "message-error";
  el.innerHTML = `<div class="error-bubble">${escapeHtml(text)}</div>`;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function renderInferenceStats(bar: HTMLDivElement, event: StreamEvent): void {
  if (event.tokensGenerated === null) return;
  const parts: string[] = [`${event.tokensGenerated} tok`];
  if (event.tokensPerSec !== null) parts.push(`${event.tokensPerSec.toFixed(1)} tok/s`);
  if (event.ttftMs !== null) parts.push(`TTFT ${event.ttftMs}ms`);
  if (event.decodeMs !== null) parts.push(`decode ${event.decodeMs}ms`);
  bar.textContent = parts.join(" · ");
  bar.classList.remove("hidden");
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

async function loadBenchmarkHistory(): Promise<void> {
  try {
    const results: BenchmarkResult[] = await invoke("list_benchmarks");
    benchmarkResults = results;
    renderAllBenchmarks();
  } catch (e) {
    console.error("list_benchmarks failed", e);
  }
}

function renderAllBenchmarks(): void {
  benchmarkResultsEl.innerHTML = "";
  if (benchmarkResults.length === 0) {
    btnCopyResults.classList.add("hidden");
    return;
  }
  const bestTps = Math.max(...benchmarkResults.map((r) => r.tokensPerSec));
  for (const r of benchmarkResults) {
    benchmarkResultsEl.appendChild(renderBenchmarkRow(r, r.tokensPerSec === bestTps));
  }
  btnCopyResults.classList.remove("hidden");
}

function renderBenchmarkRow(r: BenchmarkResult, isBest: boolean): HTMLDivElement {
  const row = document.createElement("div");
  row.className = "benchmark-row" + (isBest ? " bench-best" : "");

  const date = new Date(r.timestamp);
  const timeStr = date.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  const sizeGb = (r.modelSizeBytes / 1_000_000_000).toFixed(1);

  row.innerHTML = `
    <div class="bench-header">
      <span class="bench-model" title="${escapeHtml(r.modelName)}">${escapeHtml(r.modelName)}</span>
      <div class="bench-header-right">
        <span class="bench-preset-badge">${escapeHtml(r.promptLabel)}</span>
        <span class="bench-timestamp">${timeStr}</span>
        <button class="bench-delete" title="Delete result">×</button>
      </div>
    </div>
    <div class="bench-grid">
      <div class="bench-metric">
        <span class="bench-val">${r.tokensPerSec.toFixed(1)}</span>
        <span class="bench-label">tok/s</span>
      </div>
      <div class="bench-metric">
        <span class="bench-val">${r.toksPerSecPerGb.toFixed(1)}</span>
        <span class="bench-label">tok/s/GB</span>
      </div>
      <div class="bench-metric">
        <span class="bench-val">${r.ttftMs}</span>
        <span class="bench-label">TTFT ms</span>
      </div>
      <div class="bench-metric">
        <span class="bench-val">${r.tokensGenerated}</span>
        <span class="bench-label">tokens</span>
      </div>
      <div class="bench-metric">
        <span class="bench-val">${r.totalMs}</span>
        <span class="bench-label">total ms</span>
      </div>
    </div>
    <div class="bench-footer">${escapeHtml(r.backend)} · ${sizeGb} GB</div>
  `;

  row.querySelector<HTMLButtonElement>(".bench-delete")!.addEventListener("click", async () => {
    try {
      await invoke("delete_benchmark", { id: r.id });
      benchmarkResults = benchmarkResults.filter((x) => x.id !== r.id);
      renderAllBenchmarks();
    } catch (e) {
      console.error("delete_benchmark failed", e);
    }
  });

  return row;
}

async function handleRunBenchmark(): Promise<void> {
  if (isGenerating) {
    setStatus("Cannot run benchmark while generating.");
    return;
  }

  const customPrompt = selectedBenchmarkTokens === 0 ? benchCustomPrompt.value.trim() : undefined;
  if (selectedBenchmarkTokens === 0 && !customPrompt) {
    setBenchmarkStatus("Enter a custom prompt first.");
    return;
  }

  btnRunBenchmark.disabled = true;
  setBenchmarkStatus("Running benchmark (warm-up + timed pass)…");

  try {
    const result: BenchmarkResult = await invoke("run_benchmark", {
      maxTokens: selectedBenchmarkTokens === 0 ? 128 : selectedBenchmarkTokens,
      customPrompt: customPrompt ?? null,
    });
    benchmarkResults.unshift(result);
    renderAllBenchmarks();
    setBenchmarkStatus(`Done · ${result.tokensPerSec.toFixed(1)} tok/s`, 3000);
    setStatus(`Benchmark complete: ${result.tokensPerSec.toFixed(1)} tok/s`);
  } catch (e) {
    setBenchmarkStatus(`Error: ${e}`);
    setStatus("Benchmark failed.");
  } finally {
    btnRunBenchmark.disabled = false;
  }
}

async function handleCopyBenchmarkResults(): Promise<void> {
  if (benchmarkResults.length === 0) return;
  const header = "| Model | Backend | tok/s | tok/s/GB | TTFT ms | Tokens | Total ms | Prompt |";
  const sep    = "|---|---|---|---|---|---|---|---|";
  const rows = benchmarkResults.map((r) =>
    `| ${r.modelName} | ${r.backend} | ${r.tokensPerSec.toFixed(1)} | ${r.toksPerSecPerGb.toFixed(1)} | ${r.ttftMs} | ${r.tokensGenerated} | ${r.totalMs} | ${r.promptLabel} |`
  );
  const table = [header, sep, ...rows].join("\n");
  try {
    await navigator.clipboard.writeText(table);
    setBenchmarkStatus("Copied to clipboard!", 2000);
  } catch {
    setBenchmarkStatus("Copy failed — clipboard unavailable.");
  }
}

function setBenchmarkStatus(text: string, clearAfterMs?: number): void {
  benchmarkStatus.textContent = text;
  benchmarkStatus.classList.remove("hidden");
  if (clearAfterMs) {
    setTimeout(() => benchmarkStatus.classList.add("hidden"), clearAfterMs);
  }
}

function setStatus(text: string) { statusBar.textContent = text; }
function scrollToBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }
function escapeHtml(s: string) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ── Collapsible panels ────────────────────────────────────────────────────────

function togglePanel(panel: HTMLElement, btn: HTMLButtonElement) {
  const open = !panel.classList.contains("hidden");
  if (open) { panel.classList.add("hidden"); btn.textContent = "▸"; }
  else       { panel.classList.remove("hidden"); btn.textContent = "▾"; }
}

function showPanel(panel: HTMLElement, btn: HTMLButtonElement) {
  panel.classList.remove("hidden");
  btn.textContent = "▾";
}

// ── Confirm dialog ────────────────────────────────────────────────────────────

function confirmPrompt(message: string): Promise<boolean> {
  return new Promise((resolve) => {
    confirmText.textContent = message;
    confirmDialog.showModal();
    const ok = () => { confirmDialog.close(); cleanup(); resolve(true); };
    const cancel = () => { confirmDialog.close(); cleanup(); resolve(false); };
    const cleanup = () => {
      confirmOk.removeEventListener("click", ok);
      confirmCancel.removeEventListener("click", cancel);
    };
    confirmOk.addEventListener("click", ok, { once: true });
    confirmCancel.addEventListener("click", cancel, { once: true });
  });
}

// ── Conversation helpers ──────────────────────────────────────────────────────

function newConversation(modelName: string, systemPrompt: string | null): Conversation {
  const now = new Date().toISOString();
  return {
    id: crypto.randomUUID(),
    title: "New conversation",
    messages: [],
    modelName,
    systemPrompt,
    createdAt: now,
    updatedAt: now,
  };
}

function autoTitle(conv: Conversation, firstUserMsg: string) {
  conv.title = firstUserMsg.length > 60
    ? firstUserMsg.slice(0, 60) + "…"
    : firstUserMsg;
}

// ── Boot ──────────────────────────────────────────────────────────────────────

function waitForTauri(): Promise<void> {
  return new Promise((resolve) => {
    if ((window as any).__TAURI_INTERNALS__) { resolve(); return; }
    const id = setInterval(() => {
      if ((window as any).__TAURI_INTERNALS__) { clearInterval(id); resolve(); }
    }, 10);
    setTimeout(() => { clearInterval(id); resolve(); }, 8000);
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  await waitForTauri();
  if (!(window as any).__TAURI_INTERNALS__) {
    document.body.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0">
        <div style="text-align:center;max-width:400px;padding:2rem">
          <div style="font-size:3rem;margin-bottom:1rem">⚡</div>
          <h2 style="margin:0 0 .5rem;font-size:1.4rem">TPT Spark</h2>
          <p style="margin:0 0 1rem;color:#94a3b8;font-size:.9rem">
            This app must run inside the Tauri desktop shell.<br/>
            Open it with <code style="background:#1e2533;padding:2px 6px;border-radius:4px">npm run tauri dev</code> instead.
          </p>
        </div>
      </div>`;
    return;
  }
  await init();
});
