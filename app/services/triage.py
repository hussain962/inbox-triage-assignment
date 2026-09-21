from __future__ import annotations

from app.api.schemas import TriageRequest, TriageResponse
from app.inference.classifier import Classifier
from app.inference.lexical import LexicalClassifier
from app.policy.guidelines import GuidelineCatalog
from app.validation.input import inspect


class TriageService:
    def __init__(self, classifier: Classifier, catalog: GuidelineCatalog) -> None:
        self._classifier = classifier
        self._catalog = catalog

    def apply_policy(self, prediction: TriageResponse) -> TriageResponse:
        """Hook: the catalog is loaded. The starter does not rewrite the baseline."""
        self._catalog.permitted(prediction.intent)
        return prediction

    def handle(self, request: TriageRequest) -> TriageResponse:
        blocked = inspect(request)
        if blocked is not None:
            return blocked
        return self.apply_policy(self._classifier.predict(request))


def build_triage_service() -> TriageService:
    return TriageService(LexicalClassifier.from_train(), GuidelineCatalog.load())
