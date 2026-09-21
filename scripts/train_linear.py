#!/usr/bin/env python3
"""Train the hashing softmax intent/action models used by the service."""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.inference.features import render_conversation  # noqa: E402
from app.inference.linear import dump_meta, save_model, train_softmax  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    if path.name.endswith(".gz"):
        text = gzip.open(path, "rt", encoding="utf-8").read()
    else:
        text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def light_noise(text: str, rng: random.Random) -> str:
    """Cheap augmentations that roughly mirror the published noisy slice."""
    if not text or rng.random() < 0.35:
        return text
    kind = rng.choice(["typo", "drop", "swap", "hinglish"])
    chars = list(text)
    if kind == "typo" and chars:
        i = rng.randrange(len(chars))
        if chars[i].isalpha():
            chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
        return "".join(chars)
    if kind == "drop" and len(chars) > 8:
        i = rng.randrange(len(chars))
        del chars[i]
        return "".join(chars)
    if kind == "swap" and len(chars) > 8:
        i = rng.randrange(len(chars) - 1)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    # Mild code-mix spice on the last customer-like fragment.
    return text + " " + rng.choice(["please", "thoda", "jaldi", "yaar", "ok"])


def augment_row(row: dict, rng: random.Random) -> dict:
    clone = json.loads(json.dumps(row))
    context = clone.get("context") or []
    for turn in context:
        if turn.get("speaker") == "customer":
            turn["text"] = light_noise(turn.get("text") or "", rng)
    return clone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "data" / "train.jsonl.gz"))
    parser.add_argument(
        "--out-dir", default=str(ROOT / "app" / "inference" / "artifacts")
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--dim", type=int, default=2**17)
    parser.add_argument("--augment", type=int, default=1, help="extra noisy copies per row")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.train))
    rng = random.Random(11)
    if args.augment > 0:
        extra = []
        for row in rows:
            for _ in range(args.augment):
                if rng.random() < 0.45:
                    extra.append(augment_row(row, rng))
        print(f"augmented with {len(extra)} noisy copies")
        rows = rows + extra

    x_intent = [render_conversation(row) for row in rows]
    y_intent = [row["intent"] for row in rows]
    print(f"training intent on {len(rows)} rows, dim={args.dim}")
    intent_model = train_softmax(
        x_intent, y_intent, dim=args.dim, epochs=args.epochs, lr=0.45, seed=7
    )

    x_action = [render_conversation(row, intent=row["intent"]) for row in rows]
    y_action = [row["action"] for row in rows]
    print(f"training action on {len(rows)} rows")
    action_model = train_softmax(
        x_action, y_action, dim=args.dim, epochs=max(8, args.epochs - 2), lr=0.4, seed=13
    )

    out = Path(args.out_dir)
    save_model(intent_model, out / "intent_model.npz")
    save_model(action_model, out / "action_model.npz")

    # Intent -> historically permitted actions, for the policy layer.
    permitted: dict[str, Counter] = {}
    for row in load_jsonl(Path(args.train)):
        intent = row["intent"]
        permitted.setdefault(intent, Counter())
        for action in row.get("permitted_actions") or []:
            permitted[intent][action] += 1
    permitted_map = {
        intent: [action for action, _ in counts.most_common()]
        for intent, counts in permitted.items()
    }
    (out / "intent_permitted.json").write_text(
        json.dumps(permitted_map) + "\n", encoding="utf-8"
    )
    dump_meta(
        out / "meta.json",
        {
            "train_rows": len(rows),
            "dim": args.dim,
            "epochs": args.epochs,
            "augment": args.augment,
            "n_intents": len(set(y_intent)),
            "n_actions": len(set(y_action)),
            "model": "hashing_softmax_sgd",
        },
    )
    print(f"wrote artifacts to {out}")


if __name__ == "__main__":
    main()
