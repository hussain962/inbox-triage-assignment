from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.inference.features import tokens


def _stable_hash(token: str, dim: int) -> int:
    h = 2166136261
    for ch in token:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % dim


def featurize(text: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, values) for a hashed bag-of-ngrams vector."""
    counts: Counter[int] = Counter()
    words = tokens(text)
    for w in words:
        counts[_stable_hash("w:" + w, dim)] += 1
    for i in range(len(words) - 1):
        counts[_stable_hash("b:" + words[i] + "_" + words[i + 1], dim)] += 1
    for i in range(len(words) - 2):
        counts[_stable_hash("t:" + words[i] + "_" + words[i + 1] + "_" + words[i + 2], dim)] += 1

    # Char n-grams only on the last customer turn — enough for typos, cheap to hash.
    last_cust = ""
    for line in text.split("\n"):
        if line.startswith("customer:"):
            last_cust = line[len("customer:") :].strip()
    if last_cust:
        snippet = last_cust[-240:]
        for n in (3, 4):
            for i in range(max(0, len(snippet) - n + 1)):
                counts[_stable_hash(f"c{n}:{snippet[i:i+n]}", dim)] += 1

    if not counts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    indices = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    values = np.fromiter(
        (1.0 + math.log(v) for v in counts.values()),
        dtype=np.float64,
        count=len(counts),
    )
    return indices, values


@dataclass
class SoftmaxModel:
    classes: list[str]
    weights: np.ndarray  # (n_classes, dim)
    dim: int

    def predict_proba(self, text: str) -> np.ndarray:
        idx, val = featurize(text, self.dim)
        if idx.size == 0:
            probs = np.ones(len(self.classes), dtype=np.float64) / len(self.classes)
            return probs
        scores = self.weights[:, idx] @ val
        scores -= scores.max()
        exp = np.exp(scores)
        return exp / exp.sum()

    def predict(self, text: str) -> tuple[str, float, np.ndarray]:
        probs = self.predict_proba(text)
        i = int(probs.argmax())
        return self.classes[i], float(probs[i]), probs


def train_softmax(
    texts: list[str],
    labels: list[str],
    dim: int = 2**17,
    epochs: int = 10,
    lr: float = 0.4,
    l2: float = 1e-7,
    seed: int = 7,
) -> SoftmaxModel:
    classes = sorted(set(labels))
    label_to_index = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    weights = np.zeros((n_classes, dim), dtype=np.float64)
    rng = random.Random(seed)
    order = list(range(len(texts)))

    # Precompute features once — hashing is the expensive part.
    print("  hashing features...", flush=True)
    cached = [featurize(text, dim) for text in texts]

    for epoch in range(epochs):
        rng.shuffle(order)
        step = lr / (1.0 + 0.12 * epoch)
        correct = 0
        for i in order:
            idx, val = cached[i]
            if idx.size == 0:
                continue
            scores = weights[:, idx] @ val
            scores -= scores.max()
            exp = np.exp(scores)
            probs = exp / exp.sum()
            y = label_to_index[labels[i]]
            if int(probs.argmax()) == y:
                correct += 1
            error = probs.copy()
            error[y] -= 1.0
            # Sparse column update: W[:, idx] -= step * outer(error, val)
            weights[:, idx] -= step * np.outer(error, val)
            if l2:
                weights[:, idx] *= 1.0 - step * l2
        acc = correct / max(1, len(order))
        print(f"  epoch {epoch + 1}/{epochs} train-acc={acc:.4f}", flush=True)

    return SoftmaxModel(classes=classes, weights=weights, dim=dim)


def save_model(model: SoftmaxModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights=model.weights.astype(np.float32),
        classes=np.array(model.classes, dtype=object),
        dim=np.array([model.dim]),
    )


def load_model(path: Path) -> SoftmaxModel:
    data = np.load(path, allow_pickle=True)
    classes = [str(c) for c in data["classes"].tolist()]
    dim = int(data["dim"][0])
    return SoftmaxModel(
        classes=classes,
        weights=data["weights"].astype(np.float64),
        dim=dim,
    )


def dump_meta(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
