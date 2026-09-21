#!/usr/bin/env python3
"""Score published splits offline against the trained classifier (no HTTP)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.api.schemas import TriageRequest, Turn  # noqa: E402
from app.services.triage import build_triage_service  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def to_request(row: dict) -> TriageRequest:
    return TriageRequest(
        id=row["id"],
        context=[Turn(speaker=t["speaker"], text=t["text"]) for t in (row.get("context") or [])],
        turn_index=row.get("turn_index"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(ROOT / "data" / "dev.jsonl"))
    parser.add_argument("--eval-noisy", default=str(ROOT / "data" / "dev_noisy.jsonl"))
    parser.add_argument("--eval-hostile", default=str(ROOT / "data" / "hostile.jsonl"))
    parser.add_argument("--pred-out", default=str(ROOT / "results" / "predictions.jsonl"))
    args = parser.parse_args()

    service = build_triage_service()
    rows: list[dict] = []
    for path in (args.eval, args.eval_noisy, args.eval_hostile):
        if path:
            rows.extend(load_jsonl(Path(path)))

    out = Path(args.pred_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            pred = service.handle(to_request(row))
            fh.write(pred.model_dump_json() + "\n")
            if i % 250 == 0:
                print(f"  scored {i}/{len(rows)}")
    print(f"wrote {len(rows)} predictions to {out}")


if __name__ == "__main__":
    main()
