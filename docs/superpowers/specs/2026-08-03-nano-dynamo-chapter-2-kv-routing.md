# nano-dynamo Chapter 2: KV-Aware Routing — Design

**Status:** Draft
**Date:** 2026-08-03
**Builds on:** Chapter 1 (`2026-07-25-nano-dynamo-design.md`)

## Purpose

Chapter 1 routes requests to workers round-robin — blindly. Chapter 2 makes
the Frontend route each request to the worker most likely to already have the
request's prefix cached, so that worker can reuse its KV cache instead of
recomputing. This is the single most impactful thing Dynamo's router does, and
it's the reason a "router" is worth having at all.

The chapter's arc is deliberately narrow: **we change one function and get
cache-aware routing.** Everything else from Chapter 1 — the Registry, worker
registration and heartbeats, `/generate`, streaming, the 503 path — stays
byte-for-byte the same. Keeping the rest fixed is the lesson.

## Background: what KV-aware routing is

Every worker holds a KV cache: the attention keys/values for tokens it has
already processed. When a new request shares a *prefix* with something a worker
recently handled (same system prompt, same conversation history), routing it to
that worker lets it reuse the cached blocks instead of recomputing them — a
prefix cache hit, and a large latency win. So the Frontend should stop picking
workers blindly and start picking the worker whose cache best matches the
incoming request.

## Non-Goals

- Real tokenization. Chapter 2 approximates "tokens" with whitespace-split
  words (or fixed-size character chunks). Real tokenizers are a Dynamo detail,
  not this lesson.
- Worker-reported KV events / eviction modeling. The router predicts cache
  state from its own routing history (see Approach A below). Worker-reported
  truth is named as a stretch, not built.
- Multi-frontend correctness. The core chapter assumes a single Frontend
  replica; the limitation is called out explicitly as motivation for the
  stretch.

## Core concept: block-hash chains

Split each prompt into fixed-size blocks (default: 16 words). Hash the blocks as
a *chain*, folding each block's hash into the next:

```
h_0 = H(block_0)
h_1 = H(h_0 + block_1)
h_2 = H(h_1 + block_2)
...
```

The chaining is what makes a hash meaningful: `h_k` matches between two requests
only if blocks `0..k` all matched. So a shared `h_k` is proof of a shared prefix
through block `k`. This is exactly how Dynamo identifies reusable KV blocks;
nano-dynamo keeps the idea and drops the tokenizer.

## Where cache state lives (the key decision)

**Approach A — Router-side approximation (chosen as the teaching stand-in).** The
Frontend keeps its own model of what each worker has probably cached, built from
its own routing history: when it routes a request to worker W, it records that
request's block hashes against W. No new communication channel — workers report
nothing. Simple, stays HTTP-only, and teaches the scoring idea directly.

This is a deliberate simplification, **not** how production Dynamo works. Real
Dynamo's steady-state cache picture comes from worker-reported KV events
(Approach B below), not from guessing based on routing history. It does track
in-flight *active sequences* predictively for requests that haven't emitted
events yet, but the source of truth is real events, not the router's own
history. Approach A is chosen here purely because it keeps Chapter 2's diff tiny
and the architecture HTTP-only — see "How real Dynamo differs" for the honest
gap.

**Approach B — Worker-reported KV events (how real Dynamo actually does it; the
natural Chapter 2.5).** Workers announce which blocks they hold and evict, and
the router tracks the truth in a shared radix tree. This is production Dynamo's
primary mechanism, not a stretch in the "exotic" sense — it's simply more
machinery than one teaching chapter should introduce at once (a worker→router
event stream plus eviction modeling), so it's deferred rather than skipped.

**Decision: A for the core chapter (labeled as the stand-in it is), B as
Chapter 2.5 where fidelity to real Dynamo is the explicit goal.**

Honest caveat to state in the chapter: Approach A assumes a single Frontend,
because each Frontend only knows its own routing history. This is the same
single-replica tension Chapter 1 flagged with the embedded-registry option, and
it's precisely what motivates Approach B (or shared router state) once you scale
out.

## Components

**`KVRouter`** (new, `nano_dynamo/frontend/router.py`) — pure, synchronous logic,
no HTTP and no async. Holds:
- `prefix_owners: dict[str, set[str]]` — block-hash → set of worker_ids believed
  to hold it.
- `inflight: dict[str, int]` — worker_id → current in-flight request count.

Methods:
- `block_hashes(prompt: str) -> list[str]` — the chained hash list `h_0..h_n`.
- `select(workers: list[ModelCard], prompt: str) -> ModelCard` — the algorithm
  below.
- `record(worker_id: str, prompt: str) -> None` — register this prompt's hashes
  against the chosen worker.
- `acquire(worker_id)` / `release(worker_id)` — bump/drop the in-flight count
  around a request.

**`Frontend`** (modified, `nano_dynamo/frontend/main.py`) — swap the round-robin
line for `router.select(...)`, wrap the stream in `acquire`/`release`, and call
`router.record(...)` after selection. Small diff; nothing else changes.

**`MockEngine`** (optional but high-value, `nano_dynamo/worker/engine.py`) — make
the payoff *visible*. Accept a cache-hit signal (e.g. number of overlapping
leading blocks) and shrink or skip the simulated "prefill" delay on a hit, so a
reader running two prefix-sharing requests actually sees the second return
faster. Without this the routing is correct but invisible. If included, it needs
a matching tweak to `/generate` to pass the overlap through.

## The selection algorithm

Given the live `workers` and a `prompt`:

1. Compute the request's block-hash chain `h_0..h_n`.
2. Walk it from longest prefix to shortest (`h_n` down to `h_0`). The first
   `h_k` held by one or more *live* workers gives the best cache-hit candidates
   (they share a prefix through block `k`).
3. Among candidates, tie-break by load: pick the worker with the smallest
   `inflight` count, so one "hot" worker with the best cache doesn't get buried.
4. If no block matches any live worker (a cold prompt), fall back to
   least-loaded (ties broken round-robin).
5. Return the chosen worker; the caller then `record`s the hashes against it.

The load tie-break is itself a lesson: pure cache-affinity would pile every
similar request onto one worker. Real Dynamo scores roughly `overlap −
load_penalty`; nano-dynamo uses the simpler "best overlap, break ties by load."

A subtlety to handle: `prefix_owners` can name workers that have since been
reaped. `select` must intersect candidates against the currently-live `workers`
list (Registry remains the source of truth for liveness), and stale entries can
be pruned lazily when encountered.

## Data flow (Chapter 2)

1. Client sends `POST /v1/chat/completions` to the Frontend.
2. Frontend asks the Registry for live workers for the model (unchanged from
   Chapter 1).
3. Frontend calls `router.select(workers, prompt)` instead of round-robin.
4. Frontend `acquire`s the chosen worker, `record`s the prompt's hashes, and
   forwards to that worker's `/generate` (optionally passing the overlap count).
5. Tokens stream back exactly as in Chapter 1; on completion the Frontend
   `release`s the worker.

## Error handling

Unchanged from Chapter 1, with one addition: if the selected worker fails
mid-stream, `release` must still run (so its in-flight count doesn't leak), and
the clean in-band error chunk behavior from Chapter 1 is preserved. Use a
`try/finally` around the stream for the `release`.

## Testing

`KVRouter` is pure logic, so most tests are cheap and precise (no async, no
sockets):
- Shared-prefix routing: after routing request A to a worker, a prefix-sharing
  request B selects the same worker.
- Cold prompt: no overlap → falls back to least-loaded.
- Load tie-break: two workers both hold the prefix → the less-loaded one wins.
- `record` updates `prefix_owners` so a subsequent identical prompt hits.
- Reaped worker: a hash owned only by a now-absent worker is ignored, and
  selection falls back correctly.

Then one end-to-end Frontend test on the Chapter 1 in-process ASGITransport
harness: send request A, send a prefix-sharing request B, assert B reached the
same worker as A.

## What stays unchanged

Registry, worker registration/heartbeat/self-heal, `/generate`'s streaming, the
503-on-no-workers path, the RegistryClient, and all of Chapter 1's tests. The
chapter should actively resist folding in worker KV events, real tokenization,
or eviction — each of those is its own later lesson.

## How real Dynamo differs

This chapter keeps the skeleton of Dynamo's KV router but simplifies several
things. Verified against `lib/kv-router/src/` in the `ai-dynamo/dynamo` repo:

- **Cache state source.** nano-dynamo predicts cache from routing history
  (Approach A). Real Dynamo's primary mechanism is worker-reported KV events:
  workers emit `Stored`/`Removed` events (over ZMQ, including translated vLLM
  `BlockStored`/`BlockRemoved`), and an indexer applies them to a global radix
  tree. `indexer/kv_indexer.rs`, `zmq_wire/mod.rs`.
- **Block hashing.** The chained-hash idea matches, and this is the part
  nano-dynamo is most faithful to. Real Dynamo hashes fixed 16-token blocks with
  a sequential XXH3 chain (each block folds in the prefix before it), and has a
  keyed variant for multi-tenant isolation. nano-dynamo hashes words instead of
  real token IDs. `tracking_hash.rs`.
- **Scoring.** nano-dynamo does lexicographic "best overlap, then least load."
  Real Dynamo subtracts the cached overlap from input length to estimate the
  *remaining prefill work* (`new_tokens = ISL − overlap`) and schedules with WSPT
  / Smith's rule (`score = (1 + priority) / new_tokens`) — a continuous cost, not
  a discrete tie-break. `scheduling/policy.rs`.
- **Final pick.** nano-dynamo takes a deterministic argmax. Real Dynamo computes
  per-worker logits and does **softmax sampling with a temperature** (temperature
  0 collapses to argmax). `scheduling/selector.rs`.
- **Scope.** Real Dynamo is a full scheduler, not just a picker: admission
  control, prefill/decode disaggregation load, multi-tier overlap
  (GPU/CPU/disk), active in-flight sequence tracking, event pruning, and
  cross-replica sync. nano-dynamo omits all of it on purpose.

The honest one-liner: nano-dynamo teaches *why* you'd route by cache overlap and
*how* prefix hashing makes that possible; it does not reproduce Dynamo's
event-driven, cost-based, sampled scheduler.

## Stretch goals (named, not built)

- **Chapter 2.5 — worker-reported KV events (Approach B):** workers announce held
  and evicted blocks; the router tracks truth instead of predicting it, which
  also fixes multi-frontend correctness.
- **Real block hashing over token IDs** instead of words, once a tokenizer is in
  play.
