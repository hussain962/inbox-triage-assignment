# Analysis

## Lexical baseline failure modes

The published bag-of-words centroids (`scripts/baseline.py`) land at intent 0.37 / action 0.39 with a confidence ratio of 1.0. Looking at errors on development, three patterns dominate:

1. **Long conversations.** Once several agent/action turns are in the window, customer wording is diluted and the centroid drifts toward frequent intents like `manage` / `status`.
2. **Near-duplicate subflows.** Size vs colour returns, shipping vs payment status — lexical overlap is high, so macro-F1 suffers on the tail.
3. **Action without procedure context.** Majority-action-by-intent is only right about a quarter of the time in this data; turn position helps a little, but not enough.

Confidence was essentially unused: the gap-to-runner-up scaling rarely ranked errors below correct rows, so the top-70% error ratio sat at 1.0.

## What we changed

| Change | Intent | Action | Noisy gap | Confidence |
|---|---|---|---|---|
| Hashing softmax on word/trigram + customer char n-grams | large up | large up | held | improved |
| Light train-time noise (typo / drop / swap / hinglish tokens) | small | small | main win | — |
| Truncate very long customer turns before scoring | small on clean | helps `very_long` | helps | — |
| Policy flag when predicted action ∉ historical permitted set | — | unchanged labels | — | lowers conf / `needs_human` |
| Input gate for null bytes, injection phrases, empty bodies | — | — | — | abstain envelope |

We did **not** remapping actions onto the guideline list at inference time. Gold itself sits ~9% outside the written procedure; forcing the list would buy policy rate and lose action accuracy. Instead we keep the predicted step and raise `needs_human` when it conflicts with the intent's historical permitted set.

Rejected for this submission: calling a hosted LLM on every turn (latency/cost/key handling for a free-tier review), and shipping a heavy sklearn/scipy stack (the service stays on numpy + hashing SGD).

## Noisy slice by `noise_kind`

Action accuracy on the 300 published noisy rows (model as shipped):

| `noise_kind` | Intent acc | Action acc |
|---|---|---|
| typo | 0.66 | 0.78 |
| code_mixed | 0.64 | 0.72 |
| two_intents | 0.54 | 0.74 |
| empty | 0.50 | 0.72 |
| screenshot | 0.48 | 0.62 |
| very_long | 0.40 | 0.54 |

Empty still has prior turns, so we do **not** abstain when only the last customer string is blank — that matches how the labels were authored. `very_long` and `screenshot` remain the weak spots; truncation and char n-grams help but the customer text is no longer a clean request.

## Hostile pack

Before the validation pass, injection-style strings could still receive a refund-ish action from the lexical stand-in. After the gate:

- Prompt injection / fake `SYSTEM:` / `</context>` → schema-valid abstention (`unknown` / `none` / `needs_human`).
- Null-byte / high-control binary junk → same abstention.
- Zero-width and RTL overrides are stripped in normalization and scored as ordinary text; they must not crash the worker.

Every published hostile id returns a schema-valid body; the live service path wraps `predict` so an unexpected exception still yields the abstention envelope instead of a 500.

## Related work (one page)

- **ABCD / ASAPP dialogues** — the dataset and `guidelines.json` are the right unit (conversation prefix + procedure), not single-utterance intent-only classifiers.
- **Classical text classification** (TF–IDF / hashing + linear models) — still the right budget default when there is 20k labelled prefixes and a free-tier review machine. We use feature hashing + multinomial logistic SGD so the image stays small and dependency-light.
- **LLM routers / tool-calling agents** — attractive for free-form policy language, rejected here because the scorer wants calibrated confidence, low p95, and reproducible offline numbers without a vendor key in the zip.
- **Retrieval over guidelines** — useful if subflow titles matched intent ids; they do not. We instead materialise per-intent permitted action lists from the train split (the same resolution the dataset already applied) and keep `guidelines.json` loadable for a later title mapper.

## Development numbers (published splits)

From `scripts/grader.py` on `results/predictions.jsonl`:

- Intent accuracy 0.6187 (bar 0.55)
- Action accuracy 0.7417 (bar 0.55)
- Intent macro-F1 0.5389 (bar 0.50)
- Clean→noisy action gap 0.0633 (bar ≤ 0.10)
- Schema valid incl. hostile 100%
- Top-70% confidence error ratio 0.7921 (bar ≤ 0.85)
- Policy violation rate 0.1183 (gold 0.0883)
