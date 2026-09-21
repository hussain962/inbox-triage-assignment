from __future__ import annotations

import json
from pathlib import Path

from app.api.schemas import TriageRequest, TriageResponse
from app.inference.features import render_conversation
from app.inference.linear import SoftmaxModel, load_model


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


class LinearClassifier:
    """Hashing softmax models trained offline on the published train split."""

    def __init__(
        self,
        intent_model: SoftmaxModel,
        action_model: SoftmaxModel,
        human_threshold: float = 0.28,
    ) -> None:
        self._intent = intent_model
        self._action = action_model
        self._human_threshold = human_threshold

    @classmethod
    def from_artifacts(cls, directory: Path | None = None) -> LinearClassifier:
        base = directory or ARTIFACTS
        return cls(
            intent_model=load_model(base / "intent_model.npz"),
            action_model=load_model(base / "action_model.npz"),
        )

    def predict(self, request: TriageRequest) -> TriageResponse:
        row = {
            "id": request.id,
            "context": [turn.model_dump() for turn in request.context],
            "turn_index": request.turn_index or 0,
        }
        text = render_conversation(row)
        intent, intent_p, intent_probs = self._intent.predict(text)

        # Margin against the runner-up is more informative than raw max-prob.
        ranked = sorted(intent_probs, reverse=True)
        runner = ranked[1] if len(ranked) > 1 else 0.0
        margin = max(0.0, intent_p - runner)

        action_text = render_conversation(row, intent=intent)
        action, action_p, action_probs = self._action.predict(action_text)
        a_ranked = sorted(action_probs, reverse=True)
        a_runner = a_ranked[1] if len(a_ranked) > 1 else 0.0
        action_margin = max(0.0, action_p - a_runner)

        # Blend intent/action certainty; keep a usable spread for calibration.
        confidence = round(
            min(0.99, max(0.05, 0.55 * intent_p + 0.25 * margin + 0.20 * action_margin)),
            4,
        )
        needs_human = confidence < self._human_threshold
        return TriageResponse(
            id=str(row["id"]),
            intent=intent,
            action=action,
            confidence=confidence,
            needs_human=needs_human,
        )


def load_intent_permitted(directory: Path | None = None) -> dict[str, list[str]]:
    path = (directory or ARTIFACTS) / "intent_permitted.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
