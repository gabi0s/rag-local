const API_BASE = "http://localhost:8000";

const chatForm = document.getElementById("chatForm");
const chatText = document.getElementById("chatText");
const chatBody = document.getElementById("chatBody");
const docList = document.getElementById("docList");
const sourcesList = document.getElementById("sourcesList");
const refreshDocs = document.getElementById("refreshDocs");
const fileInput = document.getElementById("fileInput");
const runIngest = document.getElementById("runIngest");
const chunkSize = document.getElementById("chunkSize");
const chunkOverlap = document.getElementById("chunkOverlap");
const ingestStatus = document.getElementById("ingestStatus");
const newChat = document.getElementById("newChat");

const emptyState = document.getElementById("emptyState");
const typingHint = document.getElementById("typingHint");
const charCount = document.getElementById("charCount");
const sendBtn = document.getElementById("sendBtn");
const stopStream = document.getElementById("stopStream");
const toast = document.getElementById("toast");
const connBadge = document.getElementById("connBadge");
const ingestPill = document.getElementById("ingestPill");
const dropzone = document.getElementById("dropzone");

const clearSources = document.getElementById("clearSources");
const openRightPanel = document.getElementById("openRightPanel");
const closeRightPanel = document.getElementById("closeRightPanel");
const rightPanel = document.getElementById("rightPanel");
const deviceButtons = Array.from(document.querySelectorAll(".device-btn"));

let selectedDevice = "cpu";

let activeEventSource = null;
let isStreaming = false;

function showToast(message) {
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast._t);
  showToast._t = window.setTimeout(() => toast.classList.remove("show"), 2200);
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatMessageSafe(text) {
  const escaped = escapeHtml(text);
  return escaped.replaceAll("\n", "<br/>");
}

function setStreaming(next) {
  isStreaming = next;
  sendBtn.disabled = next;
  stopStream.disabled = !next;
  chatText.disabled = next;
  typingHint.textContent = next ? "Réponse en cours (streaming)..." : "Prêt.";
  ingestPill.textContent = next ? "busy" : ingestPill.textContent;
}

function updateEmptyState() {
  const hasAnyMsg = chatBody.querySelector(".msg");
  if (!emptyState) return;
  emptyState.style.display = hasAnyMsg ? "none" : "block";
}

function nowTimeLabel() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addMessage(text, role) {
  const msg = document.createElement("div");
  msg.className = `msg msg-${role}`;

  const row = document.createElement("div");
  row.className = "msg-row";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "A";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = formatMessageSafe(text);

  if (role === "user") {
    row.appendChild(bubble);
    row.appendChild(avatar);
  } else {
    row.appendChild(avatar);
    row.appendChild(bubble);
  }

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = `${role === "user" ? "You" : "Assistant"} • ${nowTimeLabel()}`;

  const actions = document.createElement("div");
  actions.className = "msg-actions";

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "action";
  copyBtn.textContent = "Copy";
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bubble.textContent || "");
      showToast("Copié ✅");
    } catch {
      showToast("Copie impossible");
    }
  });

  actions.appendChild(copyBtn);

  msg.appendChild(row);
  msg.appendChild(meta);
  msg.appendChild(actions);

  chatBody.appendChild(msg);
  chatBody.scrollTop = chatBody.scrollHeight;

  updateEmptyState();
  return bubble;
}

function renderDocs(docs) {
  docList.innerHTML = "";

  if (!docs.length) {
    const li = document.createElement("li");
    li.className = "doc-item";
    li.innerHTML = `<span class="doc-title">No documents</span><span class="doc-meta">Upload + ingest</span>`;
    docList.appendChild(li);
    return;
  }

  docs.forEach((doc) => {
    const li = document.createElement("li");
    const isIndexed = Boolean(doc.chunks);
    li.className = isIndexed ? "doc-item" : "doc-item unindexed";

    const sizeKb = doc.size ? Math.max(1, Math.round(doc.size / 1024)) : null;
    const meta = doc.chunks
      ? `chunks: ${doc.chunks}`
      : sizeKb
        ? `${sizeKb} KB`
        : "not indexed";

    li.innerHTML = `
      <span class="doc-title">${escapeHtml(doc.name)}</span>
      <span class="doc-meta">${escapeHtml(meta)}</span>
    `;
    docList.appendChild(li);
  });
}

async function loadDocs() {
  try {
    const res = await fetch(`${API_BASE}/api/docs`, { method: "GET" });
    if (!res.ok) throw new Error("docs fetch failed");
    const data = await res.json();
    renderDocs(data.docs || []);
    connBadge.textContent = "●";
    connBadge.style.color = "var(--ok)";
    connBadge.title = "Backend OK";
  } catch (err) {
    renderDocs([]);
    connBadge.textContent = "●";
    connBadge.style.color = "var(--danger)";
    connBadge.title = "Backend inaccessible";
  }
}

function renderSources(sources) {
  sourcesList.innerHTML = "";

  if (!sources.length) {
    const li = document.createElement("li");
    li.className = "source-item";
    li.innerHTML = `
      <div class="source-top">
        <div class="source-name">No sources yet</div>
        <div class="source-page"></div>
      </div>
      <div class="source-sub">Les sources apparaissent après la réponse.</div>
    `;
    sourcesList.appendChild(li);
    return;
  }

  sources.forEach((s) => {
    const li = document.createElement("li");
    li.className = "source-item";

    const name = s.source ? String(s.source) : "source";
    const page = s.page ? `p.${s.page}` : "";
    const extra = s.text ? String(s.text) : "";

    li.innerHTML = `
      <div class="source-top">
        <div class="source-name">${escapeHtml(name)}</div>
        <div class="source-page">${escapeHtml(page)}</div>
      </div>
      ${extra ? `<div class="source-sub">${escapeHtml(extra)}</div>` : ""}
    `;
    sourcesList.appendChild(li);
  });
}

async function uploadFiles(files) {
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));

  ingestStatus.textContent = "Uploading...";
  ingestPill.textContent = "uploading";

  try {
    const res = await fetch(`${API_BASE}/api/docs`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      ingestStatus.textContent = "Upload failed";
      ingestPill.textContent = "error";
      showToast("Upload failed");
      return;
    }

    ingestStatus.textContent = "Upload done. Run ingest.";
    ingestPill.textContent = "uploaded";
    showToast("Upload terminé");
    await loadDocs();
  } catch {
    ingestStatus.textContent = "Upload failed";
    ingestPill.textContent = "error";
    showToast("Backend inaccessible");
  }
}

async function runIngestJob() {
  ingestStatus.textContent = "Indexing...";
  ingestPill.textContent = "indexing";

  try {
    const res = await fetch(`${API_BASE}/api/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chunk_size: Number(chunkSize.value) || 900,
        chunk_overlap: Number(chunkOverlap.value) || 150,
      }),
    });

    if (!res.ok) {
      ingestStatus.textContent = "Ingest failed";
      ingestPill.textContent = "error";
      showToast("Ingest failed");
      return;
    }

    const data = await res.json();
    ingestStatus.textContent = `Indexed ${data.chunks} chunks`;
    ingestPill.textContent = "done";
    showToast("Indexation OK");
    await loadDocs();
  } catch {
    ingestStatus.textContent = "Ingest failed";
    ingestPill.textContent = "error";
    showToast("Backend inaccessible");
  }
}

function startStreaming(bubble) {
  const cursor = document.createElement("span");
  cursor.className = "cursor";

  bubble.innerHTML = "";
  bubble.appendChild(cursor);

  return {
    append: (text) => {
      cursor.before(document.createTextNode(text));
      chatBody.scrollTop = chatBody.scrollHeight;
    },
    done: () => {
      cursor.remove();
    },
  };
}

function stopActiveStream(reason = "Stopped") {
  if (activeEventSource) {
    try { activeEventSource.close(); } catch {}
    activeEventSource = null;
  }
  if (isStreaming) {
    setStreaming(false);
    ingestPill.textContent = "idle";
    showToast(reason);
  }
}

function autosizeTextarea() {
  chatText.style.height = "auto";
  chatText.style.height = Math.min(chatText.scrollHeight, 160) + "px";
}

function setInputCounters() {
  const v = chatText.value || "";
  charCount.textContent = String(v.length);
}

async function sendMessage(text) {
  if (!text) return;

  stopActiveStream("New request");

  addMessage(text, "user");
  renderSources([]);

  const assistantBubble = addMessage("", "assistant");
  const stream = startStreaming(assistantBubble);

  const url = new URL(`${API_BASE}/api/chat/stream`);
  url.searchParams.set("question", text);
  url.searchParams.set("device", selectedDevice);

  setStreaming(true);

  const es = new EventSource(url);
  activeEventSource = es;

  es.addEventListener("token", (e) => {
    stream.append(e.data);
  });

  es.addEventListener("sources", (e) => {
    try {
      const parsed = JSON.parse(e.data);
      renderSources(parsed);
    } catch (err) {
      renderSources([]);
    }
  });

  es.addEventListener("done", () => {
    stream.done();
    stopActiveStream("Done");
    ingestPill.textContent = "idle";
  });

  es.onerror = () => {
    stream.append("\n[stream error]");
    stream.done();
    stopActiveStream("Stream error");
    ingestPill.textContent = "error";
  };
}

/* Events */
chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = (chatText.value || "").trim();
  if (!text) return;

  chatText.value = "";
  autosizeTextarea();
  setInputCounters();
  sendMessage(text);
});

chatText.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    const text = (chatText.value || "").trim();
    if (!text || isStreaming) return;
    chatForm.requestSubmit();
  }
});

chatText.addEventListener("input", () => {
  autosizeTextarea();
  setInputCounters();
});

refreshDocs.addEventListener("click", loadDocs);

fileInput.addEventListener("change", (event) => {
  const files = event.target.files;
  if (!files || !files.length) return;
  uploadFiles(files);
  event.target.value = "";
});

function setDropzoneActive(active) {
  if (!dropzone) return;
  dropzone.classList.toggle("dragover", active);
}

if (dropzone) {
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    setDropzoneActive(true);
  });

  dropzone.addEventListener("dragleave", () => {
    setDropzoneActive(false);
  });

  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    setDropzoneActive(false);
    const files = event.dataTransfer?.files;
    if (!files || !files.length) return;
    uploadFiles(files);
  });
}

runIngest.addEventListener("click", runIngestJob);

newChat.addEventListener("click", () => {
  stopActiveStream("Stopped");
  chatBody.innerHTML = "";
  renderSources([]);
  updateEmptyState();
});

stopStream.addEventListener("click", () => {
  stopActiveStream("Stopped");
  ingestPill.textContent = "idle";
});

clearSources.addEventListener("click", () => {
  renderSources([]);
  showToast("Sources effacées");
});

document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    const prompt = btn.getAttribute("data-prompt") || "";
    if (!prompt || isStreaming) return;
    chatText.value = prompt;
    autosizeTextarea();
    setInputCounters();
    chatForm.requestSubmit();
  });
});

// Mobile right panel
function openPanel() { rightPanel.classList.add("open"); }
function closePanel() { rightPanel.classList.remove("open"); }

async function requestShutdown() {
  try {
    await fetch(`${API_BASE}/api/shutdown`, { method: "POST", keepalive: true });
  } catch {}
}

function attemptCloseTab() {
  try {
    window.open("", "_self");
    window.close();
  } catch {}
}

if (openRightPanel) openRightPanel.addEventListener("click", openPanel);
if (closeRightPanel) {
  closeRightPanel.addEventListener("click", () => {
    closePanel();
    showToast("Shutting down...");
    requestShutdown();
    window.setTimeout(attemptCloseTab, 150);
  });
}

function setDevice(device) {
  selectedDevice = device;
  deviceButtons.forEach((btn) => {
    const isActive = btn.dataset.device === device;
    btn.classList.toggle("is-active", isActive);
    btn.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
  showToast(`LLM: ${device.toUpperCase()}`);
}

deviceButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const device = btn.dataset.device;
    if (!device || device === selectedDevice) return;
    setDevice(device);
  });
});

setDevice(selectedDevice);

// Initial load
loadDocs();
renderSources([]);
updateEmptyState();
autosizeTextarea();
setInputCounters();
