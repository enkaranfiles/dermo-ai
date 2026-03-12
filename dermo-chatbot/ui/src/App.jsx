import { useEffect, useRef, useState } from "react";

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "";
const isLocalHost = typeof window !== "undefined" && ["localhost", "127.0.0.1"].includes(window.location.hostname);
const API_BASE_URL = configuredApiBaseUrl || (isLocalHost ? "http://127.0.0.1:8000" : "");

const initialMessages = [];
const emptySymptomState = {
  location: null,
  symptoms: [],
  duration: null,
  appearance: null,
  pain: null,
  itching: null,
  growth: null,
};

function formatValue(value) {
  if (Array.isArray(value)) {
    return value.length ? value.join(", ") : "Belirtilmedi";
  }
  return value || "Belirtilmedi";
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderInline(text) {
  const parts = text.split(/(\*\*.*?\*\*)/g);

  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>;
    }
    return <span key={`${part}-${index}`}>{part}</span>;
  });
}

function parseTable(block) {
  const rows = block
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (rows.length < 2 || !rows[0].includes("|")) {
    return null;
  }

  const header = rows[0]
    .split("|")
    .map((cell) => cell.trim())
    .filter(Boolean);

  const divider = rows[1].replace(/\|/g, "").trim();
  if (!header.length || !/^[-:\s]+$/.test(divider)) {
    return null;
  }

  const body = rows.slice(2).map((row) =>
    row
      .split("|")
      .map((cell) => cell.trim())
      .filter(Boolean),
  );

  return { header, body };
}

function renderInlineToHtml(text) {
  return escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function renderMessageContent(content) {
  const blocks = content.split(/\n\s*\n/).filter((block) => block.trim());

  return blocks.map((block, blockIndex) => {
    const table = parseTable(block);
    if (table) {
      return (
        <div className="markdown-table-wrap" key={`table-${blockIndex}`}>
          <table className="markdown-table">
            <thead>
              <tr>
                {table.header.map((cell, index) => (
                  <th key={`head-${blockIndex}-${index}`}>{renderInline(cell)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.body.map((row, rowIndex) => (
                <tr key={`row-${blockIndex}-${rowIndex}`}>
                  {row.map((cell, cellIndex) => (
                    <td key={`cell-${blockIndex}-${rowIndex}-${cellIndex}`}>{renderInline(cell)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }

    const lines = block.split("\n").filter(Boolean);
    if (lines.every((line) => /^\d+\.\s+/.test(line.trim()))) {
      return (
        <ol className="markdown-list" key={`list-${blockIndex}`}>
          {lines.map((line, index) => (
            <li key={`item-${blockIndex}-${index}`}>{renderInline(line.replace(/^\d+\.\s+/, ""))}</li>
          ))}
        </ol>
      );
    }

    if (lines.every((line) => line.trim().startsWith(">"))) {
      return (
        <blockquote className="markdown-quote" key={`quote-${blockIndex}`}>
          {lines.map((line, index) => (
            <p key={`quote-line-${blockIndex}-${index}`}>{renderInline(line.replace(/^>\s?/, ""))}</p>
          ))}
        </blockquote>
      );
    }

    if (lines.length === 1 && lines[0].startsWith("## ")) {
      return <h3 className="markdown-heading" key={`h2-${blockIndex}`}>{renderInline(lines[0].slice(3))}</h3>;
    }

    if (lines.length === 1 && lines[0].startsWith("# ")) {
      return <h2 className="markdown-heading" key={`h1-${blockIndex}`}>{renderInline(lines[0].slice(2))}</h2>;
    }

    return (
      <div className="markdown-paragraph-group" key={`paragraph-${blockIndex}`}>
        {lines.map((line, index) => (
          <p key={`line-${blockIndex}-${index}`}>{renderInline(line)}</p>
        ))}
      </div>
    );
  });
}

function renderMessageContentToHtml(content) {
  const blocks = content.split(/\n\s*\n/).filter((block) => block.trim());

  return blocks
    .map((block) => {
      const table = parseTable(block);
      if (table) {
        const header = table.header.map((cell) => `<th>${renderInlineToHtml(cell)}</th>`).join("");
        const body = table.body
          .map((row) => `<tr>${row.map((cell) => `<td>${renderInlineToHtml(cell)}</td>`).join("")}</tr>`)
          .join("");

        return `<div class="export-table-wrap"><table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table></div>`;
      }

      const lines = block.split("\n").filter(Boolean);
      if (lines.every((line) => /^\d+\.\s+/.test(line.trim()))) {
        return `<ol>${lines.map((line) => `<li>${renderInlineToHtml(line.replace(/^\d+\.\s+/, ""))}</li>`).join("")}</ol>`;
      }

      if (lines.every((line) => line.trim().startsWith(">"))) {
        return `<blockquote>${lines.map((line) => `<p>${renderInlineToHtml(line.replace(/^>\s?/, ""))}</p>`).join("")}</blockquote>`;
      }

      if (lines.length === 1 && lines[0].startsWith("## ")) {
        return `<h3>${renderInlineToHtml(lines[0].slice(3))}</h3>`;
      }

      if (lines.length === 1 && lines[0].startsWith("# ")) {
        return `<h2>${renderInlineToHtml(lines[0].slice(2))}</h2>`;
      }

      return lines.map((line) => `<p>${renderInlineToHtml(line)}</p>`).join("");
    })
    .join("");
}

function App() {
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState(initialMessages);
  const [draft, setDraft] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedImagePreview, setSelectedImagePreview] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [diagnosisDone, setDiagnosisDone] = useState(false);
  const [symptomState, setSymptomState] = useState(emptySymptomState);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    startSession();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function startSession() {
    setIsLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/chat/start`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("Oturum baslatilamadi.");
      }

      const data = await response.json();
      setSessionId(data.session_id);
      setMessages([{ role: "assistant", content: data.message }]);
      setDiagnosisDone(false);
      setSymptomState(emptySymptomState);
      setSelectedImage(null);
    } catch (sessionError) {
      setError("Baglanti kurulurken bir sorun olustu. API calisiyor mu kontrol edin.");
    } finally {
      setIsLoading(false);
    }
  }

  async function sendMessage(event) {
    if (event) {
      event.preventDefault();
    }
    if (!draft.trim() || !sessionId || isLoading) {
      return;
    }

    const userMessage = draft.trim();
    const nextMessages = [
      ...messages,
      {
        role: "user",
        content: userMessage,
        imagePreview: selectedImagePreview,
        imageName: selectedImage?.name || "",
      },
    ];
    setMessages(nextMessages);
    setDraft("");
    setError("");
    setIsLoading(true);

    try {
      const response = selectedImage
        ? await sendMultipartMessage(userMessage, selectedImage)
        : await sendJsonMessage(userMessage);

      setMessages([...nextMessages, { role: "assistant", content: response.reply }]);
      setSymptomState(response.symptom_state);
      setDiagnosisDone(response.diagnosis_done);
      setSelectedImage(null);
    } catch (messageError) {
      setMessages(messages);
      setError("Mesaj gonderilemedi. API yanitini kontrol edin.");
    } finally {
      setIsLoading(false);
    }
  }

  async function sendJsonMessage(message) {
    const response = await fetch(`${API_BASE_URL}/chat/message`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: sessionId,
        message,
      }),
    });

    if (!response.ok) {
      throw new Error("Mesaj istegi basarisiz.");
    }

    return response.json();
  }

  async function sendMultipartMessage(message, imageFile) {
    const formData = new FormData();
    formData.append("session_id", sessionId);
    formData.append("message", message);
    formData.append("image", imageFile);

    const response = await fetch(`${API_BASE_URL}/chat/message/image`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Gorselli mesaj istegi basarisiz.");
    }

    return response.json();
  }

  function handleImageSelection(event) {
    const file = event.target.files?.[0];
    if (!file) {
      setSelectedImage(null);
      setSelectedImagePreview("");
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      setSelectedImage(file);
      setSelectedImagePreview(typeof reader.result === "string" ? reader.result : "");
    };
    reader.readAsDataURL(file);
  }

  function handleComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function exportSessionAsPdf() {
    if (!messages.length) {
      return;
    }

    const exportWindow = window.open("", "_blank", "width=960,height=720");
    if (!exportWindow) {
      setError("PDF olusturmak icin acilan pencereye izin verin.");
      return;
    }

    const exportedAt = new Date().toLocaleString("tr-TR");
    const messageHtml = messages
      .map((message) => {
        const imageHtml = message.imagePreview
          ? `<img class="message-image" src="${message.imagePreview}" alt="${escapeHtml(message.imageName || "Yuklenen gorsel")}" />`
          : "";

        return `
          <article class="message ${message.role}">
            <div class="message-label">${message.role === "assistant" ? "Asistan" : "Kullanici"}</div>
            ${imageHtml}
            <div class="message-body">${renderMessageContentToHtml(message.content)}</div>
          </article>
        `;
      })
      .join("");

    exportWindow.document.open();
    exportWindow.document.write(`
      <!doctype html>
      <html lang="tr">
        <head>
          <meta charset="utf-8" />
          <title>Dermo AI Session</title>
          <style>
            :root {
              color-scheme: light;
              font-family: Arial, sans-serif;
            }
            body {
              margin: 0;
              padding: 32px;
              color: #1f312d;
              background: #ffffff;
            }
            .header {
              margin-bottom: 24px;
              padding-bottom: 16px;
              border-bottom: 1px solid #d9e4df;
            }
            .eyebrow {
              margin: 0 0 8px;
              font-size: 12px;
              text-transform: uppercase;
              letter-spacing: 0.14em;
              color: #55756c;
            }
            h1 {
              margin: 0;
              font-size: 28px;
            }
            .meta {
              margin-top: 8px;
              color: #5d756d;
              font-size: 14px;
            }
            .message {
              margin-bottom: 16px;
              padding: 18px;
              border-radius: 18px;
            }
            .message.assistant {
              background: #f2f8f5;
            }
            .message.user {
              background: #eef3ff;
            }
            .message-label {
              margin-bottom: 10px;
              font-size: 12px;
              font-weight: 700;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              color: #55756c;
            }
            .message-image {
              display: block;
              max-width: 260px;
              width: 100%;
              margin-bottom: 12px;
              border-radius: 14px;
            }
            .message-body {
              display: block;
            }
            .message-body p,
            .message-body h2,
            .message-body h3,
            .message-body ol,
            .message-body blockquote {
              margin-top: 0;
            }
            .message-body p:last-child,
            .message-body ol:last-child,
            .message-body blockquote:last-child,
            .message-body .export-table-wrap:last-child {
              margin-bottom: 0;
            }
            blockquote {
              margin: 0;
              padding: 12px 14px;
              border-left: 3px solid #9fb9b0;
              background: #ffffff;
            }
            ol {
              padding-left: 20px;
            }
            li + li {
              margin-top: 6px;
            }
            .export-table-wrap {
              overflow: hidden;
            }
            table {
              width: 100%;
              border-collapse: collapse;
              font-size: 14px;
            }
            th,
            td {
              padding: 10px 12px;
              border: 1px solid #d9e4df;
              text-align: left;
              vertical-align: top;
            }
            th {
              background: #f6faf8;
            }
            @media print {
              body {
                padding: 18px;
              }
              .message {
                break-inside: avoid;
              }
            }
          </style>
          <script>
            function waitForImages() {
              const images = Array.from(document.images);
              if (!images.length) {
                return Promise.resolve();
              }

              return Promise.all(
                images.map((image) => {
                  if (image.complete) {
                    return Promise.resolve();
                  }

                  return new Promise((resolve) => {
                    image.addEventListener("load", resolve, { once: true });
                    image.addEventListener("error", resolve, { once: true });
                  });
                }),
              );
            }

            window.addEventListener("load", () => {
              waitForImages().then(() => {
                window.setTimeout(() => {
                  window.focus();
                  window.print();
                }, 250);
              });
            });

            window.addEventListener("afterprint", () => {
              window.close();
            });
          </script>
        </head>
        <body>
          <header class="header">
            <p class="eyebrow">Dermo AI</p>
            <h1>Chat Session Export</h1>
            <div class="meta">Oturum: ${escapeHtml(sessionId || "N/A")} | Tarih: ${escapeHtml(exportedAt)}</div>
          </header>
          <main>${messageHtml}</main>
        </body>
      </html>
    `);
    exportWindow.document.close();
  }

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Dermo AI</p>
          <h1>Cildini anlat. Net bir on degerlendirme al.</h1>
        </div>

        <div className="hero-actions">
          <button className="ghost-button" type="button" onClick={exportSessionAsPdf} disabled={!messages.length}>
            PDF Al
          </button>
          <button className="ghost-button" type="button" onClick={startSession} disabled={isLoading}>
            Yeni Oturum
          </button>
        </div>
      </section>

      <section className="workspace">
        <div className="chat-card">
          <div className="chat-header">
            <div>
              <h2>Danisma</h2>
              <p>{sessionId ? `Oturum: ${sessionId.slice(0, 8)}...` : "Oturum hazirlaniyor"}</p>
            </div>
            <span className={`status-pill ${diagnosisDone ? "done" : ""}`}>
              {diagnosisDone ? "On degerlendirme hazir" : "Bilgi toplaniyor"}
            </span>
          </div>

          <div className="messages">
            {messages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
                <span>{message.role === "assistant" ? "Asistan" : "Siz"}</span>
                {message.imagePreview ? (
                  <figure className="message-image-wrap">
                    <img className="message-image" src={message.imagePreview} alt={message.imageName || "Yuklenen gorsel"} />
                  </figure>
                ) : null}
                <div className="message-content">{renderMessageContent(message.content)}</div>
              </article>
            ))}
            {isLoading ? (
              <article className="message assistant typing-message" aria-live="polite">
                <div className="typing-indicator" aria-label="Yaziyor">
                  <span className="typing-label">Yaziyor</span>
                  <span className="typing-ellipsis" aria-hidden="true">
                    <i>.</i>
                    <i>.</i>
                    <i>.</i>
                  </span>
                </div>
              </article>
            ) : null}
            <div ref={messagesEndRef} />
          </div>

          <form className="composer" onSubmit={sendMessage}>
            <label className="composer-input">
              <span className="sr-only">Mesajiniz</span>
              <textarea
                rows="3"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder="Sikayetinizi kisaca yazin..."
              />
            </label>

            <div className="composer-actions">
              <label className="upload-chip">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelection}
                />
                {selectedImage ? selectedImage.name : "Gorsel ekle"}
              </label>

              <button className="send-button" type="submit" disabled={isLoading || !draft.trim()}>
                {isLoading ? "Gonderiliyor" : "Gonder"}
              </button>
            </div>

            {selectedImagePreview ? (
              <div className="composer-preview">
                <img src={selectedImagePreview} alt={selectedImage?.name || "Secilen gorsel"} />
              </div>
            ) : null}

            {error ? <p className="feedback error">{error}</p> : null}
          </form>
        </div>

        <aside className="summary-card">
          <div className="summary-header">
            <h2>Semptom Ozeti</h2>
            <p>Backend tarafinda toplanan alanlar</p>
          </div>

          <dl className="summary-grid">
            <div>
              <dt>Bolge</dt>
              <dd>{formatValue(symptomState.location)}</dd>
            </div>
            <div>
              <dt>Semptomlar</dt>
              <dd>{formatValue(symptomState.symptoms)}</dd>
            </div>
            <div>
              <dt>Sure</dt>
              <dd>{formatValue(symptomState.duration)}</dd>
            </div>
            <div>
              <dt>Gorunum</dt>
              <dd>{formatValue(symptomState.appearance)}</dd>
            </div>
            <div>
              <dt>Kasinti</dt>
              <dd>{formatValue(symptomState.itching)}</dd>
            </div>
            <div>
              <dt>Agri</dt>
              <dd>{formatValue(symptomState.pain)}</dd>
            </div>
            <div>
              <dt>Buyume</dt>
              <dd>{formatValue(symptomState.growth)}</dd>
            </div>
          </dl>

          <p className="disclaimer">
            Bu arayuz bilgilendirme icindir. Tibbi tani veya ilac onerisi vermez.
          </p>
        </aside>
      </section>
    </main>
  );
}

export default App;
