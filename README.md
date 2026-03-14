# Dermo-AI

Collects symptoms through conversation, then reasons over a curated disease database to surface the most likely conditions. When a skin image is provided, a dedicated visual AI model (DermLIP) analyses the image immediately — its predictions guide the conversation and are synthesised with Claude's text-based reasoning in the final diagnosis.

> **Not a medical diagnosis tool.** For informational purposes only.

## Architecture

```
User message (text + optional image)
    │
    ├─ Image provided?
    │   YES → DermLIP runs immediately → predictions stored
    │         + injected into Claude's system prompt
    │   NO  → continue without visual context
    │
    ├─ Slot filling (current message)       extract_slots_from_message()
    ├─ Slot filling (full conversation)     extract_symptom_state()
    │       ↓
    │   SymptomState { location, symptoms, duration,
    │                  appearance, pain, itching, growth }
    │
    ├─ Sufficient? (location + ≥3 symptoms + ≥2 turns)
    │       │
    │       │ NO  → conversational follow-up
    │       │       Claude sees DermLIP predictions in system prompt,
    │       │       acknowledges the image, asks targeted questions
    │       │
    │       ↓ YES
    │
    ├─ Stage 1 — Group prediction               predict_groups()
    │       Selects 2–3 groups from 23 disease groups
    │       Fallback: all 47 diseases if result < 3
    │       ↓
    │   ┌─────────────────────────────────────────────────────┐
    │   │         DermLIP predictions available?               │
    │   ├─── YES (dual-model) ──────┬─── NO (text-only) ─────┤
    │   │                           │                         │
    │   │  Reuse stored DermLIP     │  Claude                 │
    │   │  predictions (top-5)      │  reason_over_diseases() │
    │   │         +                 │  (text only → top 3)    │
    │   │  Claude text reasoning    │                         │
    │   │  (text only → top 3)      └───────────┬─────────────┘
    │   │         ↓                              │
    │   │  Claude synthesis                      │
    │   │  merge visual + text                   │
    │   │  into final top 3                      │
    │   │         │                              │
    │   └─────────┼──────────────────────────────┘
    │             ↓
    └─ Formatted reply with risk levels (🔴 🟡 🟢) + follow-up questions
```

## Dual-Model Pipeline

The image is **never sent to Claude** — DermLIP handles all visual analysis locally, Claude works only with text.

### How it works

1. **Image upload** — DermLIP runs immediately on the uploaded image and produces top-5 condition predictions with confidence scores. These are stored for the rest of the session.

2. **Conversation** — DermLIP predictions are injected into Claude's system prompt. Claude acknowledges the image was received and uses the visual predictions to ask smarter, more targeted follow-up questions (without revealing raw predictions to the user).

3. **Diagnosis** — When enough symptoms are collected, two independent reasoning paths run:

| Path | Model | Input | Output |
|------|-------|-------|--------|
| **Visual** | DermLIP ViT-B/16 (CLIP-based) | Skin image (already predicted) | Top-5 conditions with confidence scores |
| **Text** | Claude (claude-haiku-4-5) | Symptom text only | Top-3 conditions with reasoning |

4. **Synthesis** — Claude receives both sets of results and produces a unified answer:
   - Conditions supported by **both** sources get higher confidence
   - Conditions from **only one** source are included with adjusted confidence
   - Each condition's reason states which source (visual/text/both) supports it
   - If the sources **conflict**, Claude notes it and suggests clarifying questions

### Logging

DermLIP predictions are logged to the terminal on every turn for debugging:

```
[DermLIP] Visual predictions:
  1. acne vulgaris                  confidence: 0.1523
  2. rosacea                        confidence: 0.0891
  3. folliculitis                   confidence: 0.0744
  4. impetigo                       confidence: 0.0612
  5. eczema                         confidence: 0.0489
```

### Graceful Degradation

| Scenario | Behaviour |
|----------|-----------|
| No image provided | Text-only pipeline (Claude handles everything) |
| DermLIP not installed (missing torch) | Text-only pipeline only |
| DermLIP prediction fails at runtime | Falls back to text-only result |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.13, FastAPI, Uvicorn |
| Text AI | Anthropic Claude API (claude-haiku-4-5) |
| Visual AI | DermLIP ViT-B/16 via OpenCLIP (PyTorch) |
| Frontend | React 18, Vite |
| Disease DB | 47 diseases across 23 groups (JSON) |

## Project Structure

```
dermo-chatbot/
├── api/
│   └── fastapi_app.py              # REST API + static UI serving
├── chat/
│   └── conversation_manager.py     # Multi-turn state, DermLIP-aware system prompt
├── pipeline/
│   └── diagnosis_pipeline.py       # Dual-model orchestration + DermLIP logging
├── services/
│   ├── claude_client.py            # Claude API: slots, groups, reasoning, synthesis
│   ├── dermlip_model.py            # DermLIPClassifier (CLIP ViT-B/16)
│   ├── dermlip_client.py           # DermLIP wrapper: init, predict from b64/path
│   ├── conditions.py               # 41 skin condition labels for DermLIP
│   ├── disease_filter.py           # Disease DB loader and group filtering
│   └── symptom_parser.py           # SymptomState dataclass and merge logic
├── data/
│   └── diseases.json               # 47 diseases, 23 groups
├── ui/                             # React frontend
└── main.py                         # CLI + server entry point
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for building the UI)
- Anthropic API key

### Install

```bash
# Clone and enter the project
cd dermo-ai

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (includes PyTorch + DermLIP)
pip install -r dermo-chatbot/requirements.txt

# Build the React UI
cd dermo-chatbot/ui
npm install
npx vite build
cd ../..

# Set your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > dermo-chatbot/.env
```

### Run

```bash
cd dermo-chatbot
source ../.venv/bin/activate

# Start the server (API + UI + DermLIP)
python main.py --serve --port 8000
```

Open **http://localhost:8000** in your browser.

### CLI Mode

```bash
python main.py                     # text-only chat
python main.py --image photo.jpg   # chat with skin image
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat/start` | Create a new chat session |
| POST | `/chat/message` | Send text (+ optional base64 image) |
| POST | `/chat/message/image` | Send text + image file upload |
| DELETE | `/chat/{session_id}` | End a session |
| POST | `/diagnose` | Stateless one-shot diagnosis |
| GET | `/health` | Health check (includes `dermlip_loaded` status) |

## Docker

```bash
docker build -t dermo-ai .
docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 10000:10000 dermo-ai
```

The DermLIP model is downloaded from HuggingFace on first startup and cached in `/app/.cache/huggingface` inside the container.

## Disease Database

47 dermatological conditions across 23 clinical groups, including:

- Acne and Rosacea
- Skin cancers (BCC, SCC, Melanoma)
- Eczema, Psoriasis, Dermatitis
- Fungal, viral, and bacterial infections
- Pigmentation disorders
- Autoimmune conditions
- And more

Each disease includes: name, group, symptoms, typical locations, risk level, and description — all in Turkish.
