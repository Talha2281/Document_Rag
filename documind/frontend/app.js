const uploadForm = document.querySelector("#upload-form");
const pdfInput = document.querySelector("#pdf-file");
const fileName = document.querySelector("#file-name");
const uploadButton = document.querySelector("#upload-button");
const uploadStatus = document.querySelector("#upload-status");
const uploadBox = document.querySelector(".upload-box");

const questionForm = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const askButton = document.querySelector("#ask-button");
const queryStatus = document.querySelector("#query-status");
const chat = document.querySelector("#chat");

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function setUploadStatus(text, kind = "") {
  uploadStatus.textContent = text;
  uploadStatus.className = `message ${kind}`.trim();
}

function showFile() {
  const name = pdfInput.files[0]?.name;
  fileName.textContent = name || "or drop it here";
  uploadBox.classList.toggle("has-file", Boolean(name));
}

pdfInput.addEventListener("change", showFile);

// Drag and drop
["dragenter", "dragover"].forEach((type) =>
  uploadBox.addEventListener(type, (e) => {
    e.preventDefault();
    uploadBox.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((type) =>
  uploadBox.addEventListener(type, (e) => {
    e.preventDefault();
    uploadBox.classList.remove("drag");
  })
);
uploadBox.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  pdfInput.files = transfer.files;
  showFile();
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    questionInput.focus();
  });
});

// Enter sends, Shift+Enter adds a new line
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    questionForm.requestSubmit();
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = pdfInput.files[0];

  if (!file) {
    setUploadStatus("Choose a PDF first.", "error");
    return;
  }
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setUploadStatus("Only PDF files are supported.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  uploadButton.disabled = true;
  uploadButton.classList.add("loading");
  setUploadStatus("Uploading and indexing…");

  try {
    const response = await fetch("/documents/upload", {
      method: "POST",
      body: formData
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `Upload failed (${response.status})`);
    }
    setUploadStatus(data.message || "Upload completed successfully.", "ok");
  } catch (error) {
    setUploadStatus(error.message || "Upload failed.", "error");
  } finally {
    uploadButton.disabled = false;
    uploadButton.classList.remove("loading");
  }
});

function addMessage(label, text, className) {
  const row = document.createElement("article");
  row.className = "message-row" + (className === "user-bubble" ? " mine" : "");

  const heading = document.createElement("div");
  heading.className = "message-label";
  heading.textContent = label;

  const bubble = document.createElement("div");
  bubble.className = `bubble ${className}`;
  bubble.textContent = text;

  row.append(heading, bubble);
  chat.append(row);
  scrollToEnd();
  return row;
}

function scrollToEnd() {
  chat.scrollTop = chat.scrollHeight;
}

function addThinking() {
  const row = addMessage("DocuMind", "", "answer-bubble");
  const dots = document.createElement("span");
  dots.className = "thinking";
  dots.innerHTML = "<i></i><i></i><i></i>";
  row.querySelector(".bubble").append(dots);
  return row;
}

// Types the answer out quickly; skipped when reduced motion is on
function typeText(element, text) {
  return new Promise((resolve) => {
    if (reducedMotion || text.length > 1500) {
      element.textContent = text;
      resolve();
      return;
    }
    element.textContent = "";
    element.classList.add("caret");
    let i = 0;
    const step = Math.max(2, Math.ceil(text.length / 220));
    const tick = () => {
      i = Math.min(text.length, i + step);
      element.textContent = text.slice(0, i);
      scrollToEnd();
      if (i < text.length) {
        requestAnimationFrame(tick);
      } else {
        element.classList.remove("caret");
        resolve();
      }
    };
    requestAnimationFrame(tick);
  });
}

function addMeta(row, data) {
  const chips = [];
  if (typeof data.latency_ms === "number") chips.push(`${(data.latency_ms / 1000).toFixed(1)}s`);
  if (typeof data.retries === "number" && data.retries > 0) chips.push(`${data.retries} rewrite${data.retries > 1 ? "s" : ""}`);
  if (data.abstained) chips.push("Not enough context");
  if (!chips.length) return;
  const meta = document.createElement("div");
  meta.className = "meta";
  chips.forEach((c) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = c;
    meta.append(chip);
  });
  row.append(meta);
}

questionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  document.querySelector(".welcome")?.remove();
  addMessage("You", question, "user-bubble");
  questionInput.value = "";
  askButton.disabled = true;
  askButton.classList.add("loading");
  queryStatus.textContent = "Searching documents and generating an answer…";
  const thinkingRow = addThinking();

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(data.detail || `Query failed (${response.status})`);
    }

    const answer = data.answer || data.response || "No answer was returned.";
    const bubble = thinkingRow.querySelector(".bubble");
    await typeText(bubble, answer);
    const row = thinkingRow;

    addMeta(row, data);

    const sources = data.sources || data.contexts || [];
    if (Array.isArray(sources) && sources.length) {
      const section = document.createElement("section");
      section.className = "sources";
      const heading = document.createElement("h3");
      heading.textContent = "Retrieved sources";
      section.append(heading);

      sources.forEach((source, index) => {
        const item = document.createElement("div");
        item.className = "source";
        item.style.animationDelay = `${index * 80}ms`;
        if (typeof source === "string") {
          item.textContent = source;
        } else {
          const name = source.source || source.filename || source.file || "Document";
          const page = source.page ?? source.page_number;
          item.textContent = `${index + 1}. ${name}${page ? ` · Page ${page}` : ""}`;
        }
        section.append(item);
      });
      row.append(section);
    }

    queryStatus.textContent = "";
    scrollToEnd();
  } catch (error) {
    thinkingRow.remove();
    const row = addMessage("Error", error.message || "Something went wrong.", "answer-bubble");
    row.querySelector(".bubble").classList.add("error");
    queryStatus.textContent = "Please check that the API is running.";
  } finally {
    askButton.disabled = false;
    askButton.classList.remove("loading");
    questionInput.focus();
  }
});