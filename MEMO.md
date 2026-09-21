# Memo — inbox triage pilot

To: Customer Support Manager  
From: Engineering  
Re: What the triage service will and will not do on Monday morning

## What it gets right

On the published development traffic it correctly names the subflow about **62%** of the time and the next procedure step about **74%** of the time — well above the lexical baseline we measured in-house (~37% / ~39%). Typos and light code-mixing barely move action quality. Schema-valid answers come back even when someone pastes junk or tries to talk the bot into a refund.

When the model is unsure, or when the next step it wants sits outside what that subflow's written procedure has historically allowed, it sets `needs_human` and lowers confidence instead of silently improvising policy.

## What it gets wrong

The long tail of rare subflows still hurts: macro-F1 is only ~0.54. Pasted screenshots and multi-thousand-character dumps are the noisy cases that drop furthest. About **12%** of predicted actions sit outside the written permitted set — higher than the ~9% rate in the human labels themselves, because agents already deviate and the model partly learns that habit.

It will not invent a refund because a customer said "ignore your instructions." Those messages abstain and go to a person.

## How often a human still sees it

Roughly the bottom **~25–30%** of scores (and every policy conflict / abstention) are flagged `needs_human`. That is intentional. The goal is not full automation; it is to clear the easy middle of the queue and surface the rest with a suggested intent and action attached.

## Cost

The shipped classifier is a local linear model. **Cost per message is $0** in API fees. Latency on a laptop at concurrency 4 is recorded in `results/latency_bench.json` (p95 under a second on the sample we ran). If you later swap in a hosted mini model for the hard tail only, budget a fraction of a cent per routed message and watch the p95.

## Monday morning checks

1. `GET /health` is green after deploy.
2. Spot-check ten overnight tickets: intent/action look plausible; nothing issued a refund on an injection-looking paste.
3. Dashboard: share of `needs_human`, policy-conflict rate, and p95 latency vs yesterday.
4. If noisy customer channels (Hinglish / screenshot-only) spike, expect more handoffs — that is the system working, not breaking.

We should revisit the rare-subflow tail and screenshot handling before raising the automation share further.
