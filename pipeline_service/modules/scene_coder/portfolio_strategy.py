from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_STRATEGY_PATH = _ROOT / "portfolio_strategy.json"


def _load_strategy() -> dict[str, Any]:
    try:
        payload = json.loads(_STRATEGY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def system_prompt_suffix() -> str:
    strategy = _load_strategy()
    system = str(strategy.get("system_prompt") or "").strip()
    if not system:
        return ""
    return "\n\n" + system + "\n"


def candidate_user_suffix(candidate_id: int) -> str:
    strategy = _load_strategy()
    roles = strategy.get("candidate_roles")
    if not isinstance(roles, list) or not roles:
        return ""
    role = roles[candidate_id % len(roles)]
    return str(role).strip()
