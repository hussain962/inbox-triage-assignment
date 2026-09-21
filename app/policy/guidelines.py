from __future__ import annotations

import json
from pathlib import Path

from app.paths import DATA


def action_slug(button: str) -> str:
    return button.strip().lower().replace(" ", "-")


class GuidelineCatalog:
    """Written procedures from `guidelines.json`.

    Intent names in the data and subflow titles in the procedure are not the same
    string. At train time we also materialise the per-intent permitted action
    lists the dataset already resolved, so the service can check membership
    without inventing a fragile title matcher.
    """

    def __init__(
        self,
        by_subflow: dict[str, frozenset[str]],
        by_intent: dict[str, list[str]] | None = None,
    ) -> None:
        self._by_subflow = by_subflow
        self._by_intent = by_intent or {}

    @classmethod
    def load(cls, path: Path | None = None) -> GuidelineCatalog:
        source = path or DATA / "guidelines.json"
        raw = json.loads(source.read_text(encoding="utf-8"))
        by_subflow: dict[str, frozenset[str]] = {}
        for flow in raw.values():
            for name, subflow in (flow.get("subflows") or {}).items():
                actions = {
                    action_slug(step["button"])
                    for step in subflow.get("actions") or []
                    if step.get("button")
                }
                by_subflow[name] = frozenset(actions)

        permitted_path = (
            Path(__file__).resolve().parent.parent
            / "inference"
            / "artifacts"
            / "intent_permitted.json"
        )
        by_intent: dict[str, list[str]] = {}
        if permitted_path.exists():
            by_intent = json.loads(permitted_path.read_text(encoding="utf-8"))
        return cls(by_subflow, by_intent)

    def permitted(self, subflow: str) -> frozenset[str] | None:
        return self._by_subflow.get(subflow)

    def permitted_for_intent(self, intent: str) -> list[str]:
        return list(self._by_intent.get(intent) or [])

    def constrain_action(self, intent: str, action: str) -> tuple[str, bool]:
        """Keep the predicted action when allowed; otherwise fall back.

        Returns (action, remapped). Gold itself violates the written procedure
        ~9% of the time, so we only remap when the intent has a known list and
        the prediction is outside it.
        """
        allowed = self.permitted_for_intent(intent)
        if not allowed:
            return action, False
        if action in allowed:
            return action, False
        return allowed[0], True
