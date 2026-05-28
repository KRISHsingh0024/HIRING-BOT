const queryInput = document.getElementById("queryInput");
const topKInput = document.getElementById("topKInput");
const rerankerInput = document.getElementById("rerankerInput");
const sendButton = document.getElementById("sendButton");
const copyButton = document.getElementById("copyButton");
const resetButton = document.getElementById("resetButton");
const healthValue = document.getElementById("healthValue");
const serverStatus = document.getElementById("serverStatus");
const actionValue = document.getElementById("actionValue");
const turnCountValue = document.getElementById("turnCountValue");
const endValue = document.getElementById("endValue");
const provenanceValue = document.getElementById("provenanceValue");
const clarifyValue = document.getElementById("clarifyValue");
const recommendationsValue = document.getElementById("recommendationsValue");
const retrievedAssessmentsValue = document.getElementById("retrievedAssessmentsValue");
const jsonValue = document.getElementById("jsonValue");
const chatMessages = document.getElementById("chatMessages");

let lastPayload = null;
const STORAGE_KEYS = {
  query: "shl.query",
  topK: "shl.topK",
  reranker: "shl.reranker",
  messages: "shl.messages",
};

let messages = [];

function loadSetting(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value === null ? fallback : value;
  } catch (error) {
    return fallback;
  }
}

function saveSetting(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (error) {
    // Ignore storage failures in private or restricted browsing modes.
  }
}

function setTextContentIfPresent(node, value) {
  if (node) {
    node.textContent = value;
  }
}

function setStatus(text, tone = "ok") {
  serverStatus.textContent = text;
  serverStatus.style.background =
    tone === "ok"
      ? "rgba(94, 234, 212, 0.12)"
      : tone === "warn"
      ? "rgba(251, 191, 36, 0.12)"
      : "rgba(251, 113, 133, 0.12)";
  serverStatus.style.color =
    tone === "ok" ? "#5eead4" : tone === "warn" ? "#fde68a" : "#fecdd3";
}

function autoResizeTextarea() {
  if (!queryInput) {
    return;
  }
  queryInput.style.height = "auto";
  queryInput.style.height = `${Math.max(queryInput.scrollHeight, 56)}px`;
}

function formatTime(timestamp) {
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch (error) {
    return "";
  }
}

function persistMessages() {
  try {
    localStorage.setItem(STORAGE_KEYS.messages, JSON.stringify(messages));
  } catch (error) {
    // ignore
  }
}

function loadMessages() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.messages);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter((m) => m && typeof m.role === "string" && typeof m.content === "string")
      .map((m) => ({ role: m.role, content: m.content, ts: m.ts || Date.now() }));
  } catch (error) {
    return [];
  }
}

function renderChat() {
  if (!chatMessages) {
    return;
  }

  if (!Array.isArray(messages) || messages.length === 0) {
    chatMessages.innerHTML = `
      <div class="chat-empty">
        Start a conversation. Example: <strong>I need a backend-leaning Python developer assessment</strong>.
      </div>
    `;
    return;
  }

  chatMessages.innerHTML = messages
    .map((m) => {
      const isUser = m.role === "user";
      const label = isUser ? "You" : "Assistant";
      const time = m.ts ? formatTime(m.ts) : "";
      return `
        <div class="chat-message ${isUser ? "user" : "assistant"}">
          <div class="chat-meta"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(time)}</span></div>
          <div class="bubble">${escapeHtml(m.content)}</div>
        </div>
      `;
    })
    .join("");

  // Scroll to bottom
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderDetailGrid(target, entries) {
  if (!target) {
    return;
  }

  if (!Array.isArray(entries) || entries.length === 0) {
    target.classList.add("empty-state");
    target.textContent = "—";
    return;
  }

  target.classList.remove("empty-state");
  target.innerHTML = entries
    .map(
      ({ label, value }) => `
        <div class="detail-item">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `,
    )
    .join("");
}

function renderRecommendations(recommendations) {
  if (!Array.isArray(recommendations) || recommendations.length === 0) {
    recommendationsValue.innerHTML = '<div class="cards empty-state">No recommendations yet.</div>';
    return;
  }

  recommendationsValue.innerHTML = recommendations
    .map((item) => {
      const meta = item.description || item.title || "";
      const rank = item.final_rank || item.rank || "—";
      const scoreBits = [
        item.hybrid_score !== undefined ? `Hybrid ${Number(item.hybrid_score).toFixed(3)}` : null,
        item.rerank_score !== undefined ? `Rerank ${Number(item.rerank_score).toFixed(3)}` : null,
        item.metadata_score !== undefined ? `Meta ${Number(item.metadata_score).toFixed(3)}` : null,
      ].filter(Boolean);

      return `
        <div class="card">
          <h3>#${rank} ${escapeHtml(item.title || item.id || "Assessment")}</h3>
          <p>${escapeHtml(meta)}</p>
          <div class="badge-row">
            <span class="badge">${escapeHtml(item.id || "unknown")}</span>
            ${scoreBits.map((bit) => `<span class="badge warn">${escapeHtml(bit)}</span>`).join("")}
          </div>
        </div>
      `;
    })
    .join("");
}

function renderRetrievedAssessments(retrievedAssessments) {
  if (!Array.isArray(retrievedAssessments) || retrievedAssessments.length === 0) {
    retrievedAssessmentsValue.innerHTML = '<div class="cards empty-state">No retrieved assessments yet.</div>';
    return;
  }

  retrievedAssessmentsValue.innerHTML = retrievedAssessments
    .map((item) => {
      const meta = item.meta || {};
      const scoreBits = [
        item.hybrid_score !== undefined ? `Hybrid ${Number(item.hybrid_score).toFixed(3)}` : null,
        item.vector_score !== undefined ? `Vector ${Number(item.vector_score).toFixed(3)}` : null,
        item.bm25_score !== undefined ? `BM25 ${Number(item.bm25_score).toFixed(3)}` : null,
        item.metadata_score !== undefined ? `Meta ${Number(item.metadata_score).toFixed(3)}` : null,
      ].filter(Boolean);

      return `
        <div class="card">
          <h3>#${escapeHtml(item.rank ?? item.final_rank ?? "—")} ${escapeHtml(item.title || item.id || "Assessment")}</h3>
          <p>${escapeHtml(meta.description || item.description || "")}</p>
          <div class="badge-row">
            <span class="badge">${escapeHtml(item.id || "unknown")}</span>
            ${scoreBits.map((bit) => `<span class="badge warn">${escapeHtml(bit)}</span>`).join("")}
          </div>
        </div>
      `;
    })
    .join("");
}

function renderPayload(data) {
  lastPayload = data;
  actionValue.textContent = data.action || "—";
  turnCountValue.textContent = data.turn_count ?? "—";
  endValue.textContent = data.end_of_conversation === true ? "true" : data.end_of_conversation === false ? "false" : "—";
  clarifyValue.textContent = data.clarify_prompt || "—";
  clarifyValue.classList.toggle("empty", !data.clarify_prompt);
  renderRecommendations(data.recommendations);
  renderRetrievedAssessments(data.retrieved_assessments);
  renderDetailGrid(provenanceValue, data.provenance
    ? Object.entries(data.provenance).map(([label, value]) => ({ label, value: value === null || value === undefined ? "—" : String(value) }))
    : []);
  jsonValue.textContent = JSON.stringify(data, null, 2);
}

async function fetchHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    const idx = data.indices ? ` · ${data.indices}` : "";
    healthValue.textContent = `${data.status} (${data.backend})${idx}`;
    setStatus("Ready", "ok");
  } catch (error) {
    healthValue.textContent = "offline";
    setStatus("Backend offline", "danger");
  }
}

function addMessage(role, content) {
  if (!content) {
    return;
  }
  messages.push({ role, content, ts: Date.now() });
  persistMessages();
  renderChat();
}

function resetChat() {
  messages = [];
  persistMessages();
  renderChat();
  renderPayload({ action: "—", recommendations: [], retrieved_assessments: [] });
  setStatus("Ready", "ok");
}

async function runTest() {
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Enter a query first", "warn");
    return;
  }

  saveSetting(STORAGE_KEYS.query, query);
  saveSetting(STORAGE_KEYS.topK, String(topKInput.value || 5));
  saveSetting(STORAGE_KEYS.reranker, rerankerInput.checked ? "true" : "false");

  addMessage("user", query);
  queryInput.value = "";
  autoResizeTextarea();

  const payload = {
    messages: messages.map((m) => ({ role: m.role, content: m.content })),
    top_k: Number(topKInput.value || 5),
    use_reranker: rerankerInput.checked,
  };

  sendButton.disabled = true;
  sendButton.textContent = "Running…";
  setStatus("Sending request…", "warn");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    let data;
    try {
      data = await response.json();
    } catch (error) {
      data = { detail: "Non-JSON response" };
    }

    if (!response.ok) {
      const detail = data && (data.detail || data.message) ? String(data.detail || data.message) : JSON.stringify(data);
      data = {
        action: "error",
        reply: detail,
        recommendations: [],
        retrieved_assessments: [],
        turn_count: 1,
        end_of_conversation: true,
      };
    }

    renderPayload(data);

    if (data.action === "clarify" && data.clarify_prompt) {
      addMessage("assistant", data.clarify_prompt);
    } else if (data.reply) {
      addMessage("assistant", data.reply);
    } else if (data.reason) {
      addMessage("assistant", String(data.reason));
    }

    setStatus(response.ok ? "Response received" : "Request returned an error", response.ok ? "ok" : "danger");
  } catch (error) {
    renderPayload({ action: "error", reply: String(error), recommendations: [], end_of_conversation: true });
    addMessage("assistant", `Request failed: ${String(error)}`);
    setStatus("Request failed", "danger");
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Send";
  }
}

async function copyJson() {
  if (!lastPayload) {
    setStatus("Nothing to copy yet", "warn");
    return;
  }

  try {
    await navigator.clipboard.writeText(JSON.stringify(lastPayload, null, 2));
    setStatus("JSON copied", "ok");
  } catch (error) {
    setStatus("Copy failed", "danger");
  }
}

queryInput.value = loadSetting(STORAGE_KEYS.query, queryInput.value);
topKInput.value = loadSetting(STORAGE_KEYS.topK, topKInput.value);
rerankerInput.checked = loadSetting(STORAGE_KEYS.reranker, "true") !== "false";
autoResizeTextarea();

messages = loadMessages();
renderChat();

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query || "";
    saveSetting(STORAGE_KEYS.query, queryInput.value);
    autoResizeTextarea();
    queryInput.focus();
  });
});

sendButton.addEventListener("click", runTest);
copyButton.addEventListener("click", copyJson);
if (resetButton) {
  resetButton.addEventListener("click", resetChat);
}
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    runTest();
  }
});
queryInput.addEventListener("input", () => {
  autoResizeTextarea();
  saveSetting(STORAGE_KEYS.query, queryInput.value);
});
topKInput.addEventListener("change", () => saveSetting(STORAGE_KEYS.topK, String(topKInput.value || 5)));
rerankerInput.addEventListener("change", () => saveSetting(STORAGE_KEYS.reranker, rerankerInput.checked ? "true" : "false"));

fetchHealth();
