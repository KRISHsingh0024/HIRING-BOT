const queryInput = document.getElementById("queryInput");
const topKInput = document.getElementById("topKInput");
const rerankerInput = document.getElementById("rerankerInput");
const sendButton = document.getElementById("sendButton");
const copyButton = document.getElementById("copyButton");
const healthValue = document.getElementById("healthValue");
const serverStatus = document.getElementById("serverStatus");
const actionValue = document.getElementById("actionValue");
const turnCountValue = document.getElementById("turnCountValue");
const endValue = document.getElementById("endValue");
const replyValue = document.getElementById("replyValue");
const clarifyValue = document.getElementById("clarifyValue");
const recommendationsValue = document.getElementById("recommendationsValue");
const jsonValue = document.getElementById("jsonValue");

let lastPayload = null;

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

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
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

function renderPayload(data) {
  lastPayload = data;
  actionValue.textContent = data.action || "—";
  turnCountValue.textContent = data.turn_count ?? "—";
  endValue.textContent = data.end_of_conversation === true ? "true" : data.end_of_conversation === false ? "false" : "—";
  replyValue.textContent = data.reply || data.reason || "—";
  replyValue.classList.toggle("empty", !(data.reply || data.reason));
  clarifyValue.textContent = data.clarify_prompt || "—";
  clarifyValue.classList.toggle("empty", !data.clarify_prompt);
  renderRecommendations(data.recommendations);
  jsonValue.textContent = JSON.stringify(data, null, 2);
}

async function fetchHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    healthValue.textContent = `${data.status} (${data.backend})`;
    setStatus("Ready", "ok");
  } catch (error) {
    healthValue.textContent = "offline";
    setStatus("Backend offline", "danger");
  }
}

async function runTest() {
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Enter a query first", "warn");
    return;
  }

  const payload = {
    messages: [{ role: "user", content: query }],
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

    const data = await response.json();
    renderPayload(data);
    setStatus(response.ok ? "Response received" : "Request returned an error", response.ok ? "ok" : "danger");
  } catch (error) {
    renderPayload({ action: "error", reply: String(error), recommendations: [], end_of_conversation: true });
    setStatus("Request failed", "danger");
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Run Test";
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

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query || "";
    queryInput.focus();
  });
});

sendButton.addEventListener("click", runTest);
copyButton.addEventListener("click", copyJson);
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    runTest();
  }
});

fetchHealth();
