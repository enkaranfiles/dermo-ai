# Dermo-AI

Collects symptoms through conversation, then reasons over a curated disease database to surface the most likely conditions.

> **Not a medical diagnosis tool.** For informational purposes only.

## Algorithm

```
User message
    │
    ├─ Slot filling (current message)      extract_slots_from_message()
    ├─ Slot filling (full conversation)    extract_symptom_state()
    │       ↓
    │   SymptomState { location, symptoms, duration,
    │                  appearance, pain, itching, growth }
    │
    ├─ Sufficient? (location + ≥3 symptoms)
    │       │ NO  → conversational follow-up
    │       │       (early group prediction to ask targeted questions)
    │       │
    │       ↓ YES
    │
    ├─ Stage 1 — Group prediction          predict_groups()
    │       Picks 2–3 disease groups out of 22
    │       Score: 0.7 × symptom_overlap + 0.3 × location_match
    │       Fallback: all diseases if result < 3
    │       ↓
    ├─ Stage 2 — Disease reasoning         reason_over_diseases()
    │       Ranks top 3 conditions from filtered list
    │       Returns: name, confidence (0–1), reason, next_questions
    │       Image (if provided) is passed here as base64
    │       ↓
    └─ Formatted reply with risk levels (🔴 🟡 🟢) + follow-up questions
```

