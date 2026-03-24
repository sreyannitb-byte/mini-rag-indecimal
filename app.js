const statusText = document.getElementById("statusText");
const reindexBtn = document.getElementById("reindexBtn");
const askBtn = document.getElementById("askBtn");
const queryInput = document.getElementById("queryInput");
const topKInput = document.getElementById("topK");
const answerOutput = document.getElementById("answerOutput");
const contextOutput = document.getElementById("contextOutput");

async function fetchStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  statusText.textContent = `Indexed sources: ${data.indexed_sources} | Indexed chunks: ${data.indexed_chunks}`;
}

async function reindex() {
  reindexBtn.disabled = true;
  try {
    const res = await fetch("/api/reindex", { method: "POST" });
    const data = await res.json();
    statusText.textContent = `Reindexed successfully. Chunks: ${data.indexed_chunks}`;
  } catch (err) {
    statusText.textContent = `Reindex failed: ${err}`;
  } finally {
    reindexBtn.disabled = false;
    fetchStatus();
  }
}

function renderContext(chunks) {
  contextOutput.innerHTML = "";
  if (!chunks || chunks.length === 0) {
    contextOutput.textContent = "No retrieved chunks.";
    return;
  }

  chunks.forEach((chunk) => {
    const card = document.createElement("div");
    card.className = "context-item";
    card.innerHTML = `
      <div class="meta">
        Source: <b>${chunk.source}</b> | Chunk: ${chunk.chunk_id} | Score: ${chunk.score}
      </div>
      <div>${chunk.text}</div>
    `;
    contextOutput.appendChild(card);
  });
}

async function askQuestion() {
  const query = queryInput.value.trim();
  if (!query) return;

  askBtn.disabled = true;
  answerOutput.textContent = "Generating answer...";
  contextOutput.textContent = "Retrieving context...";

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: Number(topKInput.value || 4),
      }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Request failed");
    }

    answerOutput.textContent = data.answer;
    renderContext(data.retrieved_context);
  } catch (err) {
    answerOutput.textContent = `Error: ${err.message || err}`;
    contextOutput.textContent = "No context due to error.";
  } finally {
    askBtn.disabled = false;
  }
}

reindexBtn.addEventListener("click", reindex);
askBtn.addEventListener("click", askQuestion);
fetchStatus();
