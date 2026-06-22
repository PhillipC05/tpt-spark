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
}

interface SystemInfo {
  backend: string;
  engineLoaded: boolean;
  modelName: string | null;
}

// ── State ─────────────────────────────────────────────────────────────────────

let isGenerating = false;

// ── DOM refs ──────────────────────────────────────────────────────────────────

const modelSelect    = document.getElementById("model-select")    as HTMLSelectElement;
const modelStatus    = document.getElementById("model-status")    as HTMLDivElement;
const modelStatusTxt = document.getElementById("model-status-text") as HTMLSpanElement;
const modelMeta      = document.getElementById("model-meta")      as HTMLDivElement;
const btnLoad        = document.getElementById("btn-load")        as HTMLButtonElement;
const btnUnload      = document.getElementById("btn-unload")      as HTMLButtonElement;
const btnRefresh     = document.getElementById("btn-refresh")     as HTMLButtonElement;
const modelsDirEl    = document.getElementById("models-dir")      as HTMLDivElement;
const backendBadge   = document.getElementById("backend-badge")   as HTMLDivElement;

const promptInput    = document.getElementById("prompt-input")    as HTMLTextAreaElement;
const btnSend        = document.getElementById("btn-send")        as HTMLButtonElement;
const messagesEl     = document.getElementById("messages")        as HTMLDivElement;
const statusBar      = document.getElementById("status-bar")      as HTMLDivElement;
const welcomeScreen  = document.getElementById("welcome-screen")  as HTMLDivElement;

const paramMaxTokens    = document.getElementById("param-max-tokens")    as HTMLInputElement;
const paramTemperature  = document.getElementById("param-temperature")   as HTMLInputElement;
const paramTopP         = document.getElementById("param-top-p")         as HTMLInputElement;
const paramRepeatPenalty = document.getElementById("param-repeat-penalty") as HTMLInputElement;
const tempVal  = document.getElementById("temp-val")!;
const toppVal  = document.getElementById("topp-val")!;
const repVal   = document.getElementById("rep-val")!;

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  await Promise.all([loadSystemInfo(), loadModelsDir(), refreshModelList()]);

  paramTemperature.addEventListener("input", () => (tempVal.textContent = paramTemperature.value));
  paramTopP.addEventListener("input", () => (toppVal.textContent = paramTopP.value));
  paramRepeatPenalty.addEventListener("input", () => (repVal.textContent = paramRepeatPenalty.value));

  btnRefresh.addEventListener("click", refreshModelList);
  btnLoad.addEventListener("click", handleLoadModel);
  btnUnload.addEventListener("click", handleUnloadModel);
  btnSend.addEventListener("click", handleSend);

  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  promptInput.addEventListener("input", () => {
    promptInput.style.height = "auto";
    promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + "px";
  });
}

// ── System info ───────────────────────────────────────────────────────────────

async function loadSystemInfo() {
  try {
    const info: SystemInfo = await invoke("get_system_info");
    backendBadge.textContent = `backend: ${info.backend}`;
    if (info.engineLoaded && info.modelName) {
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
  } catch (e) {
    modelsDirEl.textContent = "unavailable";
  }
}

// ── Model list ────────────────────────────────────────────────────────────────

async function refreshModelList() {
  setStatus("Scanning for models…");
  modelSelect.disabled = true;
  modelSelect.innerHTML = '<option value="">Scanning…</option>';
  btnLoad.disabled = true;

  try {
    const models: ModelEntry[] = await invoke("list_models");

    modelSelect.innerHTML = "";
    if (models.length === 0) {
      modelSelect.innerHTML = '<option value="">No .gguf files found</option>';
      setStatus("No models found. Add .gguf files to the models directory.");
    } else {
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "— Select a model —";
      modelSelect.appendChild(placeholder);

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

// ── Load / unload ─────────────────────────────────────────────────────────────

async function handleLoadModel() {
  const path = modelSelect.value;
  if (!path) return;

  setModelLoading();
  setStatus("Loading model…");
  btnLoad.disabled = true;
  btnUnload.disabled = true;

  try {
    const info: ModelInfo = await invoke("load_model", { path });
    setModelLoaded(info);
    setStatus(`Model loaded: ${info.name}`);
    enableChat();
  } catch (e: unknown) {
    setModelUnloaded();
    appendError(`Failed to load model: ${e}`);
    setStatus("Model load failed.");
  }
}

async function handleUnloadModel() {
  try {
    await invoke("unload_model");
  } catch (e) {
    console.error("unload error", e);
  }
  setModelUnloaded();
  disableChat();
  setStatus("Model unloaded.");
}

function setModelLoading() {
  modelStatus.className = "model-status loading";
  modelStatusTxt.textContent = "Loading…";
  btnLoad.disabled = true;
  btnUnload.disabled = true;
}

function setModelLoaded(info: ModelInfo) {
  modelStatus.className = "model-status loaded";
  modelStatusTxt.textContent = info.name;
  btnLoad.disabled = false;
  btnUnload.disabled = false;

  if (info.sizeBytes > 0) {
    const sizeGb = (info.sizeBytes / 1_073_741_824).toFixed(1);
    modelMeta.textContent = `Backend: ${info.backend} · Size: ${sizeGb} GB`;
    modelMeta.classList.remove("hidden");
  }
}

function setModelUnloaded() {
  modelStatus.className = "model-status not-loaded";
  modelStatusTxt.textContent = "No model loaded";
  btnLoad.disabled = modelSelect.value === "";
  btnUnload.disabled = true;
  modelMeta.classList.add("hidden");
}

// ── Chat enable / disable ─────────────────────────────────────────────────────

function enableChat() {
  promptInput.disabled = false;
  btnSend.disabled = false;
  promptInput.placeholder = "Send a message… (Enter to send, Shift+Enter for newline)";
  promptInput.focus();
}

function disableChat() {
  promptInput.disabled = true;
  btnSend.disabled = true;
  promptInput.placeholder = "Load a model to start chatting…";
}

// ── Inference ─────────────────────────────────────────────────────────────────

async function handleSend() {
  if (isGenerating) return;
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  promptInput.value = "";
  promptInput.style.height = "auto";
  isGenerating = true;
  btnSend.disabled = true;

  hideWelcome();
  appendUserMessage(prompt);
  const bubble = appendAssistantMessage();

  setStatus("Generating…");

  const channel = new Channel<StreamEvent>();
  channel.onmessage = (event) => {
    if (event.done) {
      bubble.classList.remove("streaming");
      isGenerating = false;
      btnSend.disabled = false;
      setStatus("");
    } else {
      bubble.textContent += event.token;
      scrollToBottom();
    }
  };

  try {
    await invoke("run_inference", {
      prompt,
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
    setStatus("Error during generation.");
  }
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

function hideWelcome() {
  welcomeScreen?.remove();
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

function appendAssistantMessage(): HTMLDivElement {
  const msg = document.createElement("div");
  msg.className = "message assistant";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble streaming";
  msg.innerHTML = `<div class="message-avatar">⚡</div>`;
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  scrollToBottom();
  return bubble;
}

function appendError(text: string) {
  const el = document.createElement("div");
  el.className = "message-error";
  el.innerHTML = `<div class="error-bubble">${escapeHtml(text)}</div>`;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function setStatus(text: string) {
  statusBar.textContent = text;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", init);
