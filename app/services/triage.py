from __future__ import annotations

from pathlib import Path

from app.api.schemas import TriageRequest, TriageResponse
from app.inference.classifier import Classifier
from app.inference.trained import LinearClassifier
from app.policy.guidelines import GuidelineCatalog
from app.validation.input import inspect


class TriageService:
    def __init__(self, classifier: Classifier, catalog: GuidelineCatalog) -> None:
        self._classifier = classifier
        self._catalog = catalog

    def apply_policy(self, prediction: TriageResponse) -> TriageResponse:
        allowed = self._catalog.permitted_for_intent(prediction.intent)
        if not allowed or prediction.action in allowed:
            return prediction
        # Real agents step outside the written procedure ~9% of the time.
        # Keep the predicted next step (that is what the labels teach) but make
        # the handoff loud so the manager can review the conflict.
        return prediction.model_copy(
            update={
                "needs_human": True,
                "confidence": min(prediction.confidence, 0.4),
            }
        )

    def handle(self, request: TriageRequest) -> TriageResponse:
        blocked = inspect(request)
        if blocked is not None:
            return blocked
        try:
            return self.apply_policy(self._classifier.predict(request))
        except Exception:
            # Never let a bad row take down the worker — hostile pack must stay up.
            return TriageResponse(
                id=request.id,
                intent="unknown",
                action="none",
                confidence=0.0,
                needs_human=True,
            )


def build_triage_service() -> TriageService:
    artifacts = Path(__file__).resolve().parent.parent / "inference" / "artifacts"
    classifier: Classifier
    if (artifacts / "intent_model.npz").exists():
        classifier = LinearClassifier.from_artifacts(artifacts)
    else:
        # Local boot without artifacts — lexical stand-in from day one.
        from app.inference.lexical import LexicalClassifier

        classifier = LexicalClassifier.from_train()
    return TriageService(classifier, GuidelineCatalog.load())
