"""End-to-end diagnosis pipeline: symptom → group prediction → filter → Claude reasoning → result.

When an image is provided the pipeline runs two models in parallel:
  1. DermLIP (visual CLIP model)  → top-k predictions from the skin image
  2. Claude  (text-based reasoning) → top-3 conditions from symptom text
Claude then synthesises both results into a single unified answer.
"""

import json
import re
from typing import Optional

from services import claude_client
from services.disease_filter import (
    get_all_diseases_for_claude,
    get_all_groups,
    get_diseases_by_group_ids,
    get_risk_level,
)
from services.symptom_parser import SymptomState, update_symptom_state
from chat.conversation_manager import ConversationManager, DISCLAIMER

# DermLIP is optional — gracefully degrade if not available (e.g. missing torch)
try:
    from services import dermlip_client as _dermlip
    _HAS_DERMLIP = True
except ImportError:
    _dermlip = None  # type: ignore[assignment]
    _HAS_DERMLIP = False


def _parse_claude_json(raw: str) -> dict:
    """Extract and parse JSON from Claude's response text."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"candidate_conditions": []}


def process_user_message(
    manager: ConversationManager,
    user_input: str,
    image_path: Optional[str] = None,
    image_b64: Optional[str] = None,
    media_type: str = "image/jpeg",
) -> str:
    """
    Full pipeline for a single user turn:
    1. Record user message
    2. Claude slot filling on current message
    3. Claude slot filling on full conversation (cross-turn inference)
    4. If sufficient info → stage-1 group prediction → stage-2 disease reasoning
    5. Else → conversational follow-up (group-aware questions if groups known)

    Returns the assistant's reply (string, Turkish).
    """
    # 1. Add user message to history
    manager.add_user_message(user_input, image_b64=image_b64, media_type=media_type)

    # 1b. Run DermLIP immediately if a new image was provided
    if image_b64 and not manager.visual_predictions and _HAS_DERMLIP and _dermlip.is_loaded():
        try:
            manager.visual_predictions = _dermlip.predict_from_b64(image_b64, top_k=5)
        except Exception as e:
            print(f"[DermLIP] Visual prediction failed: {e}")
    elif image_path and not manager.visual_predictions and _HAS_DERMLIP and _dermlip.is_loaded():
        try:
            manager.visual_predictions = _dermlip.predict_from_path(image_path, top_k=5)
        except Exception as e:
            print(f"[DermLIP] Visual prediction failed: {e}")

    if manager.visual_predictions:
        print("\n[DermLIP] Visual predictions:")
        for i, p in enumerate(manager.visual_predictions, 1):
            print(f"  {i}. {p['condition']:30s} confidence: {p['confidence']:.4f}")
        print()

    # 2. Claude slot filling on current message (handles morphology, synonyms, slang)
    raw_slots = claude_client.extract_slots_from_message(user_input)
    manager.symptom_state = update_symptom_state(manager.symptom_state, raw_slots)

    # 3. Claude slot filling on full conversation context (catches cross-turn inferences)
    def _msg_text(m: dict) -> str:
        c = m["content"]
        if isinstance(c, list):
            return " ".join(b["text"] for b in c if b.get("type") == "text")
        return c

    conversation_text = "\n".join(
        f"{'Kullanıcı' if m['role'] == 'user' else 'Asistan'}: {_msg_text(m)}"
        for m in manager.get_messages()
    )
    raw_extraction = claude_client.extract_symptom_state(conversation_text)
    manager.symptom_state = update_symptom_state(manager.symptom_state, raw_extraction)

    # 4. Decide: proceed to diagnosis or ask follow-up?
    if manager.should_proceed_to_diagnosis() and not manager.diagnosis_done:
        reply = _run_diagnosis(manager)
        manager.diagnosis_done = True
    else:
        # Run group prediction early (even before diagnosis) to enable targeted questions
        if not manager.predicted_group_ids and manager.symptom_state.location:
            group_ids = claude_client.predict_groups(
                manager.symptom_state.to_text(), get_all_groups()
            )
            if group_ids:
                manager.predicted_group_ids = group_ids

        reply = _run_conversational_turn(manager)

    manager.add_assistant_message(reply)
    return reply


def _run_conversational_turn(manager: ConversationManager) -> str:
    """Generate a conversational follow-up using Claude."""
    system = manager.get_system_prompt()
    messages = manager.get_messages()
    response = claude_client.chat_with_claude(messages, system_prompt=system)
    return response + DISCLAIMER


def _run_diagnosis(manager: ConversationManager) -> str:
    """
    Diagnosis pipeline with dual-model support:

    **Without image** (text-only):
      Stage 1 — Claude predicts the most likely disease groups.
      Stage 2 — Claude reasons over diseases filtered to those groups.

    **With image** (dual-model):
      DermLIP predictions are already stored in manager.visual_predictions.
      Claude text reasoning runs separately, then both are synthesised.
    """
    state: SymptomState = manager.symptom_state

    # ── Stage 1: group prediction (reuse if already predicted) ────────────
    if not manager.predicted_group_ids:
        group_ids = claude_client.predict_groups(state.to_text(), get_all_groups())
        manager.predicted_group_ids = group_ids

    if manager.predicted_group_ids:
        candidate_diseases = get_diseases_by_group_ids(manager.predicted_group_ids)
        if len(candidate_diseases) < 3:
            candidate_diseases = get_all_diseases_for_claude()
    else:
        candidate_diseases = get_all_diseases_for_claude()

    # ── Dual-model path (visual predictions available) ────────────────────
    if manager.visual_predictions:
        # Claude text-based reasoning (WITHOUT image — text analysis only)
        raw_text_result = claude_client.reason_over_diseases(
            symptoms_text=state.to_text(),
            all_diseases=candidate_diseases,
        )
        text_result = _parse_claude_json(raw_text_result)

        # Synthesise DermLIP visual + Claude text predictions
        raw_synth = claude_client.synthesize_predictions(
            symptoms_text=state.to_text(),
            visual_predictions=manager.visual_predictions,
            text_conditions=text_result.get("conditions", []),
            next_questions=text_result.get("next_questions", []),
        )
        result = _parse_claude_json(raw_synth)
        return _format_diagnosis_result(result, dual_model=True)

    # ── Text-only path (no image or DermLIP unavailable) ──────────────────
    # Image is never sent to Claude — only DermLIP handles visual analysis.
    raw_result = claude_client.reason_over_diseases(
        symptoms_text=state.to_text(),
        all_diseases=candidate_diseases,
    )
    result = _parse_claude_json(raw_result)
    return _format_diagnosis_result(result, dual_model=False)


def _format_diagnosis_result(result: dict, dual_model: bool = False) -> str:
    """Format a diagnosis result dict into a user-facing Turkish string."""
    conditions = result.get("conditions", [])
    next_questions = result.get("next_questions", [])

    RISK_LABEL = {"yüksek": "🔴 Yüksek", "orta": "🟡 Orta", "düşük": "🟢 Düşük"}

    lines = []
    if dual_model:
        lines.append(
            "Görüntü analizi (yapay zeka görsel modeli) ve semptom analizi (metin tabanlı klinik değerlendirme) "
            "birlikte değerlendirilmiştir.\n"
        )
    lines.append("Verdiğiniz bilgilere göre olası dermatolojik durumlar:\n")

    for i, cond in enumerate(conditions, 1):
        name = cond.get("name", "Bilinmiyor")
        confidence = cond.get("confidence", 0.0)
        reason = cond.get("reason", "")
        risk = get_risk_level(name)
        risk_label = RISK_LABEL.get(risk, risk)
        pct = int(confidence * 100)

        lines.append(f"{i}. **{name}**")
        lines.append(f"   Risk: {risk_label}  |  Uyum: %{pct}")
        if reason:
            lines.append(f"   _{reason}_")
        lines.append("")

    if next_questions:
        lines.append("Tanıyı daha da netleştirmek için:")
        for q in next_questions:
            lines.append(f"• {q}")
        lines.append("")

    lines.append(
        "Bu bilgiler yalnızca genel bilgi amaçlıdır. "
        "Kesin tanı için dermatoloji uzmanına başvurunuz."
    )
    lines.append(DISCLAIMER)

    return "\n".join(lines)


def get_diagnosis_result_json(
    state: SymptomState,
    image_path: Optional[str] = None,
    image_b64: Optional[str] = None,
    media_type: str = "image/jpeg",
) -> dict:
    """
    Stateless version for the /diagnose API endpoint.
    Uses dual-model pipeline when an image is provided:
      - DermLIP visual predictions + Claude text reasoning → synthesis.
    Falls back to text-only when no image or DermLIP unavailable.
    Returns raw structured JSON: {conditions, next_questions}.
    """
    group_ids = claude_client.predict_groups(state.to_text(), get_all_groups())
    if group_ids:
        candidate_diseases = get_diseases_by_group_ids(group_ids)
        if len(candidate_diseases) < 3:
            candidate_diseases = get_all_diseases_for_claude()
    else:
        candidate_diseases = get_all_diseases_for_claude()

    has_image = bool(image_b64 or image_path)

    # Dual-model path
    if has_image and _HAS_DERMLIP and _dermlip.is_loaded():
        visual_predictions: list[dict] = []
        try:
            if image_b64:
                visual_predictions = _dermlip.predict_from_b64(image_b64, top_k=5)
            elif image_path:
                visual_predictions = _dermlip.predict_from_path(image_path, top_k=5)
        except Exception as e:
            print(f"[DermLIP] Visual prediction failed: {e}")

        # Claude text reasoning (without image)
        raw_text = claude_client.reason_over_diseases(
            symptoms_text=state.to_text(),
            all_diseases=candidate_diseases,
        )
        text_result = _parse_claude_json(raw_text)

        if visual_predictions:
            raw_synth = claude_client.synthesize_predictions(
                symptoms_text=state.to_text(),
                visual_predictions=visual_predictions,
                text_conditions=text_result.get("conditions", []),
                next_questions=text_result.get("next_questions", []),
            )
            return _parse_claude_json(raw_synth)

        return text_result

    # Text-only path — image is never sent to Claude
    raw_result = claude_client.reason_over_diseases(
        symptoms_text=state.to_text(),
        all_diseases=candidate_diseases,
    )
    return _parse_claude_json(raw_result)
