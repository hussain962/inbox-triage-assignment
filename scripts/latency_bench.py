#!/usr/bin/env python3
"""Measure triage latency at a stated concurrency against the live service."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def project(row: dict) -> dict:
    payload = {"id": row["id"], "context": row.get("context") or []}
    if row.get("turn_index") is not None:
        payload["turn_index"] = row["turn_index"]
    return payload


def post_once(base: str, payload: dict) -> tuple[str, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/triage",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()
    return payload["id"], time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--eval", default="data/dev.jsonl")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default="results/latency_bench.json")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.eval))[: args.sample]
    payloads = [project(r) for r in rows]
    durations: list[tuple[str, float]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(post_once, args.base, p) for p in payloads]
        for fut in as_completed(futures):
            durations.append(fut.result())

    seconds = [d for _, d in durations]
    seconds_sorted = sorted(seconds)
    p95 = seconds_sorted[max(0, int(round(0.95 * len(seconds_sorted))) - 1)]
    out = {
        "sample_count": len(seconds),
        "concurrency": args.concurrency,
        "base": args.base,
        "latency_mean_s": round(statistics.mean(seconds), 4),
        "latency_p50_s": round(statistics.median(seconds), 4),
        "latency_p95_s": round(p95, 4),
        "cost_per_message_usd": 0.0,
        "request_ids": [rid for rid, _ in durations],
        "durations_s": [round(d, 4) for _, d in durations],
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"n={out['sample_count']} concurrency={args.concurrency} "
        f"p50={out['latency_p50_s']}s p95={out['latency_p95_s']}s -> {path}"
    )


if __name__ == "__main__":
    main()
