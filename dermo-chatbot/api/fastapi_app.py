"""FastAPI REST API for the dermatology chatbot."""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # loads ANTHROPIC_API_KEY from .env
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chat.conversation_manager import ConversationManager
from pipeline.diagnosis_pipeline import process_user_message, get_diagnosis_result_json
from services.symptom_parser import SymptomState

# DermLIP is optional — app works without it (text-only mode)
try:
    from services import dermlip_client
    _HAS_DERMLIP = True
except ImportError:
    dermlip_client = None  # type: ignore[assignment]
    _HAS_DERMLIP = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models at startup."""
    if _HAS_DERMLIP:
        dermlip_client.init_dermlip()
    yield


app = FastAPI(
    title="Dermatoloji Chatbot API",
    description="Türkçe dermatoloji bilgi asistanı — PoC",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (PoC — use Redis/DB in production)
_sessions: dict[str, ConversationManager] = {}
UI_DIST_DIR = Path(__file__).resolve().parent.parent / "ui" / "dist"


# ── Schemas ──────────────────────────────────────────────────────────────────

class StartResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    image_b64: Optional[str] = None
    media_type: Optional[str] = "image/jpeg"


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    symptom_state: dict
    diagnosis_done: bool


class DiagnoseRequest(BaseModel):
    symptoms: list[str]
    location: Optional[str] = None
    duration: Optional[str] = None
    appearance: Optional[str] = None
    pain: Optional[str] = None
    itching: Optional[str] = None
    growth: Optional[str] = None
    image_b64: Optional[str] = None
    media_type: Optional[str] = "image/jpeg"


class DiagnoseResponse(BaseModel):
    candidate_conditions: list[dict]
    disclaimer: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat/start", response_model=StartResponse, tags=["Chat"])
def start_session():
    """Create a new chat session and return the opening greeting."""
    session_id = str(uuid.uuid4())
    manager = ConversationManager()
    greeting = manager.build_greeting()
    manager.add_assistant_message(greeting)
    _sessions[session_id] = manager
    return StartResponse(session_id=session_id, message=greeting)


@app.post("/chat/message", response_model=ChatResponse, tags=["Chat"])
def send_message(req: ChatRequest):
    """Send a user message and receive the assistant's reply."""
    manager = _sessions.get(req.session_id)
    if manager is None:
        raise HTTPException(status_code=404, detail="Oturum bulunamadı. Lütfen yeni bir oturum başlatın.")

    reply = process_user_message(
        manager=manager,
        user_input=req.message,
        image_b64=req.image_b64,
        media_type=req.media_type or "image/jpeg",
    )

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        symptom_state=manager.symptom_state.to_dict(),
        diagnosis_done=manager.diagnosis_done,
    )


@app.post("/chat/message/image", response_model=ChatResponse, tags=["Chat"])
async def send_message_with_image(
    session_id: str = Form(...),
    message: str = Form(...),
    image: UploadFile = File(...),
):
    """Send a user message with an uploaded skin image."""
    import base64

    manager = _sessions.get(session_id)
    if manager is None:
        raise HTTPException(status_code=404, detail="Oturum bulunamadı.")

    raw = await image.read()
    image_b64 = base64.standard_b64encode(raw).decode("utf-8")
    media_type = image.content_type or "image/jpeg"

    reply = process_user_message(
        manager=manager,
        user_input=message,
        image_b64=image_b64,
        media_type=media_type,
    )

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        symptom_state=manager.symptom_state.to_dict(),
        diagnosis_done=manager.diagnosis_done,
    )


@app.post("/diagnose", response_model=DiagnoseResponse, tags=["Diagnosis"])
def diagnose(req: DiagnoseRequest):
    """
    Stateless endpoint: submit symptoms directly and get candidate conditions.
    No session required.
    """
    state = SymptomState(
        location=req.location,
        symptoms=req.symptoms,
        duration=req.duration,
        appearance=req.appearance,
        pain=req.pain,
        itching=req.itching,
        growth=req.growth,
    )

    result = get_diagnosis_result_json(
        state=state,
        image_b64=req.image_b64,
        media_type=req.media_type or "image/jpeg",
    )

    return DiagnoseResponse(
        candidate_conditions=result.get("candidate_conditions", []),
        disclaimer=(
            "Bu sistem yalnızca bilgilendirme amaçlıdır. "
            "Kesin teşhis için dermatoloji uzmanına başvurunuz."
        ),
    )


@app.get("/health", tags=["System"])
def health():
    dermlip_loaded = _HAS_DERMLIP and dermlip_client.is_loaded()
    return {
        "status": "ok",
        "service": "dermo-chatbot",
        "dermlip_loaded": dermlip_loaded,
    }


@app.delete("/chat/{session_id}", tags=["Chat"])
def end_session(session_id: str):
    """Remove a chat session."""
    if session_id in _sessions:
        del _sessions[session_id]
    return {"status": "deleted"}


if UI_DIST_DIR.exists():
    assets_dir = UI_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="ui-assets")

    @app.get("/", include_in_schema=False)
    def serve_ui():
        return FileResponse(UI_DIST_DIR / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def ui_not_built():
        return JSONResponse(
            {
                "message": "UI build not found.",
                "ui_path": str(UI_DIST_DIR),
                "hint": "Run the React app in dermo-chatbot/ui or build it to ui/dist.",
            }
        )
