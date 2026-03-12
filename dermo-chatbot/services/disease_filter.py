"""Disease knowledge base loader. Group-based two-stage filtering for clinical reasoning."""

import json
from pathlib import Path

_DISEASES: list[dict] | None = None


def _load_diseases() -> list[dict]:
    global _DISEASES
    if _DISEASES is None:
        data_path = Path(__file__).parent.parent / "data" / "diseases.json"
        with open(data_path, encoding="utf-8") as f:
            _DISEASES = json.load(f)
    return _DISEASES


def get_all_groups() -> list[dict]:
    """Return unique disease groups for group-prediction step."""
    seen = set()
    groups = []
    for d in _load_diseases():
        gid = d.get("group_id")
        if gid not in seen:
            seen.add(gid)
            groups.append({"group_id": gid, "group": d.get("group", "")})
    return sorted(groups, key=lambda g: g["group_id"])


def get_diseases_by_group_ids(group_ids: list[int]) -> list[dict]:
    """Return compact disease dicts filtered to the given group IDs."""
    return [
        {
            "name": d["name"],
            "group": d.get("group", ""),
            "risk_level": d["risk_level"],
            "typical_symptoms": d["symptoms"],
            "typical_locations": d["locations"],
            "description": d["description"],
        }
        for d in _load_diseases()
        if d.get("group_id") in group_ids
    ]


def get_all_diseases_for_claude() -> list[dict]:
    """Return all diseases in compact form (fallback when group prediction fails)."""
    return [
        {
            "name": d["name"],
            "group": d.get("group", ""),
            "risk_level": d["risk_level"],
            "typical_symptoms": d["symptoms"],
            "typical_locations": d["locations"],
            "description": d["description"],
        }
        for d in _load_diseases()
    ]


def get_risk_level(disease_name: str) -> str:
    """Look up risk_level for a disease by name (for formatting the response)."""
    for d in _load_diseases():
        if d["name"].lower() == disease_name.lower():
            return d["risk_level"]
    return "bilinmiyor"
