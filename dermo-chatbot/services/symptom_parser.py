"""Symptom slot-filling and state management."""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
# Note: keyword-based extraction was removed.
# Slot filling is done entirely via Claude (see claude_client.extract_slots_from_message).


@dataclass
class SymptomState:
    """Holds the structured symptom information collected so far."""
    location: Optional[str] = None
    symptoms: list[str] = field(default_factory=list)
    duration: Optional[str] = None
    appearance: Optional[str] = None
    pain: Optional[str] = None
    itching: Optional[str] = None
    growth: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        """Human-readable summary of collected symptoms."""
        parts = []
        if self.location:
            parts.append(f"Konum: {self.location}")
        if self.symptoms:
            parts.append(f"Semptomlar: {', '.join(self.symptoms)}")
        if self.duration:
            parts.append(f"Süre: {self.duration}")
        if self.appearance:
            parts.append(f"Görünüm: {self.appearance}")
        if self.pain:
            parts.append(f"Ağrı: {self.pain}")
        if self.itching:
            parts.append(f"Kaşıntı: {self.itching}")
        if self.growth:
            parts.append(f"Büyüme: {self.growth}")
        return "\n".join(parts) if parts else "Henüz semptom bilgisi toplanmadı."

    def is_sufficient(self) -> bool:
        """Require location + at least 3 symptom signals for accurate reasoning."""
        symptom_signals = len(self.symptoms)
        return bool(self.location) and symptom_signals >= 3

    def missing_slots(self) -> list[str]:
        """Return slot names that are still insufficient."""
        missing = []
        if not self.location:
            missing.append("location")
        if len(self.symptoms) < 3:
            missing.append("symptoms")
        if not self.duration:
            missing.append("duration")
        if not self.appearance:
            missing.append("appearance")
        return missing


def _extract_json(text: str) -> dict:
    """Try to parse JSON from a Claude response, even if wrapped in markdown."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def update_symptom_state(state: SymptomState, claude_json_response: str) -> SymptomState:
    """
    Merge Claude's structured extraction result into the current SymptomState.
    Claude returns a JSON string; we parse it and update non-null fields.
    """
    extracted = _extract_json(claude_json_response)
    if not extracted:
        return state

    if extracted.get("location"):
        state.location = extracted["location"]
    if extracted.get("symptoms"):
        for s in extracted["symptoms"]:
            if s and s not in state.symptoms:
                state.symptoms.append(s)
    if extracted.get("duration"):
        state.duration = extracted["duration"]
    if extracted.get("appearance"):
        state.appearance = extracted["appearance"]
    if extracted.get("pain"):
        state.pain = extracted["pain"]
    if extracted.get("itching"):
        state.itching = extracted["itching"]
    if extracted.get("growth"):
        state.growth = extracted["growth"]

    return state


