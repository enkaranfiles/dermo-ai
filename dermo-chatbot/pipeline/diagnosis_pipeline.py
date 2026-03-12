"""End-to-end diagnosis pipeline: symptom → group prediction → filter → Claude reasoning → result."""

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
    # 1. Add user message to history (image embedded in content when provided)
    manager.add_user_message(user_input, image_b64=image_b64, media_type=media_type)

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
        reply = _run_diagnosis(
            manager,
            image_path=image_path,
            image_b64=manager.image_b64,
            media_type=manager.image_media_type,
        )
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


def _run_diagnosis(
    manager: ConversationManager,
    image_path: Optional[str] = None,
    image_b64: Optional[str] = None,
    media_type: str = "image/jpeg",
) -> str:
    """
    Two-stage diagnosis:
      Stage 1 — Claude predicts the most likely disease groups.
      Stage 2 — Claude reasons over diseases filtered to those groups.
    Falls back to all diseases if group prediction returns nothing.
    """
    state: SymptomState = manager.symptom_state

    # Stage 1: group prediction (reuse if already predicted in follow-up phase)
    if not manager.predicted_group_ids:
        group_ids = claude_client.predict_groups(state.to_text(), get_all_groups())
        manager.predicted_group_ids = group_ids

    if manager.predicted_group_ids:
        candidate_diseases = get_diseases_by_group_ids(manager.predicted_group_ids)
        # Safety net: if filtering yields < 3 diseases, fall back to all
        if len(candidate_diseases) < 3:
            candidate_diseases = get_all_diseases_for_claude()
    else:
        candidate_diseases = get_all_diseases_for_claude()

    # Stage 2: reasoning over candidates
    raw_result = claude_client.reason_over_diseases(
        symptoms_text=state.to_text(),
        all_diseases=candidate_diseases,
        image_b64=image_b64,
        image_path=image_path,
        media_type=media_type,
    )

    result = _parse_claude_json(raw_result)
    conditions = result.get("conditions", [])
    next_questions = result.get("next_questions", [])

    RISK_LABEL = {"yüksek": "🔴 Yüksek", "orta": "🟡 Orta", "düşük": "🟢 Düşük"}

    lines = ["Verdiğiniz bilgilere göre olası dermatolojik durumlar:\n"]

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
    Uses two-stage pipeline: group prediction → filtered reasoning.
    Returns raw structured JSON: {conditions, next_questions}.
    """
    group_ids = claude_client.predict_groups(state.to_text(), get_all_groups())
    if group_ids:
        candidate_diseases = get_diseases_by_group_ids(group_ids)
        if len(candidate_diseases) < 3:
            candidate_diseases = get_all_diseases_for_claude()
    else:
        candidate_diseases = get_all_diseases_for_claude()

    raw_result = claude_client.reason_over_diseases(
        symptoms_text=state.to_text(),
        all_diseases=candidate_diseases,
        image_b64=image_b64,
        image_path=image_path,
        media_type=media_type,
    )
    return _parse_claude_json(raw_result)
