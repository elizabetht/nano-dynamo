# nano-dynamo Design

**Status:** Draft
**Date:** 2026-07-25

## Purpose

nano-dynamo is a minimal, from-scratch teaching implementation of Dynamo's orchestration layer, in the spirit of nanoGPT and nano-vllm: strip everything that isn't the core lesson, keep everything that is.

Real Dynamo is a large, production-grade, datacenter-scale inference orchestration system. It does not implement inference itself — it orchestrates pluggable engines (vLLM, SGLang, TRT-LLM) sitting underneath it, handling service discovery, request routing, KV-cache-aware placement, and disaggregated prefill/decode serving. nano-dynamo teaches exactly that layer: discovery, routing, and disaggregation — not the inference engine itself, which is why it can be small even though real Dynamo is not.

**Audience:** a teaching resource for others coming from a Python background, with no assumed Rust knowledge.

## Non-Goals

- Not a performance-competitive inference server. No real ML dependencies in the core chapters.
- Not a 1:1 port of Dynamo's Rust core. Where real Dynamo's implementation detail would obscure the lesson (e.g. writing a custom binary transport protocol), nano-dynamo takes the simpler, explicit path instead and documents the difference.
- Not a general-purpose library. It is a sequence of runnable chapters meant to be read, run, and modified.

## Language & Stack

Pure Python — no Rust. Real Dynamo splits a Rust core from Python bindings for production performance; nano-dynamo's lessons (frontend/worker split, service discovery, model cards, KV-aware routing, disaggregated serving) are distributed-systems concepts, not Rust-performance concepts, and a Rust build step would put a barrier between readers and the ideas being taught.

Stack across all services: **FastAPI + Pydantic + uvicorn**. FastAPI gives readable, function-shaped routes and automatic request validation; Pydantic models double as the natural way to teach "the model card is just a typed schema." This is a deliberate simplification versus real Dynamo (which uses etcd + NATS + a Rust core), called out explicitly rather than left implicit.

## Scope

Delivered as sequential chapters, each additive on top of the last:

- **Chapter 1 (this spec's primary scope):** frontend / worker / discovery — the foundation everything else builds on.
- **Chapter 2:** KV-cache-aware routing (replaces Chapter 1's round-robin worker selection).
- **Chapter 3:** disaggregated serving — separate prefill and decode workers, coordinated through the same Registry.

Chapters 2 and 3 are named here for context but are not designed in this document; this spec covers Chapter 1's architecture in full, since Chapters 2–3 build on it without restructuring it (e.g. `ModelCard.worker_type` already exists but is unused until Chapter 3).

## Architecture

Three independent Python processes, all FastAPI + Pydantic + uvicorn, communicating only over HTTP:

- **Registry** — the shared source of truth for "who's alive and what can they serve," standing in for etcd. Workers `POST /register` on startup with a typed `ModelCard`, then `POST /heartbeat/{worker_id}` on a timer. A background `asyncio` reaper task expires any worker whose heartbeat goes stale past a TTL (mirrors etcd lease expiry). `GET /workers?model=...` is how anyone else finds out who's currently alive.
- **Worker** — runs a mock inference engine (async generator yielding fake tokens with a small per-token delay, to feel like real streaming generation with no ML dependencies). Registers itself with the Registry on startup, heartbeats in a background task, and exposes an internal `POST /generate` endpoint that only the Frontend calls.
- **Frontend** — the client-facing, OpenAI-compatible-ish surface (`POST /v1/chat/completions`). On each request, queries the Registry for live workers serving the requested model, picks one (round-robin in Chapter 1 — this is exactly where Chapter 2's KV-aware routing later plugs in), forwards the request, and streams the worker's tokens back to the caller.

This is a genuinely separate Registry process rather than one embedded in the Frontend. An embedded registry was considered (see Alternatives Considered) but rejected: it silently assumes exactly one Frontend replica, since each replica would have its own private view of live workers. That's not just a simplification but a structurally different model from real Dynamo, and it would break Chapter 3, where prefill workers, decode workers, and the Frontend must all agree on one shared view of who's alive. In real Dynamo, discovery is backed by etcd (and the request-plane by NATS), both genuinely independent infrastructure — the Frontend and every Worker are just clients of it, neither one *is* the discovery mechanism. The Registry process preserves that property.

Nothing here talks to anything except over HTTP — the Registry doesn't know about the Frontend's routing logic, the Worker doesn't know who's calling it, and the Frontend doesn't know how the Worker generates tokens. That separation is the actual lesson.

## Components

**`ModelCard`** (Pydantic model, defined once in a shared `models.py` imported by all three processes — not three copies that could drift) — the typed schema a worker registers with:
- `worker_id`, `model_name`, `endpoint_url`
- `worker_type` (defaults to `"aggregated"`; unused until Chapter 3's Prefill/Decode split)
- `registered_at`, `last_heartbeat` (managed by the Registry, not the worker)

This deliberately mirrors real Dynamo's `ModelDeploymentCard` — a typed capability descriptor a worker publishes, not something the Frontend has to infer.

**Registry service**
- `POST /register` — accepts a `ModelCard`, returns a `worker_id`.
- `POST /heartbeat/{worker_id}` — refreshes `last_heartbeat`; 404s if the worker was already reaped.
- `GET /workers?model=...` — returns live cards, optionally filtered by model name.
- Background `asyncio` task, running every few seconds, drops any worker whose `last_heartbeat` is older than the TTL.

**Worker service**
- On startup: registers with the Registry, then spawns a background heartbeat loop.
- Exposes `POST /generate`, called only by the Frontend — an `Engine.generate()` async generator that yields fake tokens one at a time with a configurable delay, so the streaming behavior is real even though the "inference" is fake.
- `Engine` is a small interface (`async def generate(prompt: str) -> AsyncIterator[str]`), with `MockEngine` as the only implementation used in Chapters 1–3. See Appendix for the optional real-engine extension point.

**Frontend service**
- Exposes `POST /v1/chat/completions`.
- Per request: calls the Registry's `GET /workers` for the current live set for the requested model, round-robins to one, forwards the request to that worker's `/generate`, and streams the response straight through to the caller via FastAPI's `StreamingResponse`.

## Data Flow (Chapter 1)

1. Worker process starts → `POST /register` to Registry with its `ModelCard` → Registry returns a `worker_id` → Worker starts a background heartbeat loop (`POST /heartbeat/{worker_id}` every few seconds).
2. Client sends `POST /v1/chat/completions` to the Frontend.
3. Frontend calls Registry's `GET /workers?model=...` to get the current live set.
4. Frontend picks one (round-robin), forwards the request to that worker's `/generate`.
5. Worker's `MockEngine` streams fake tokens back over HTTP; Frontend re-streams them to the original caller via `StreamingResponse` — end to end, a real streaming response, just fake content.
6. If a worker stops heartbeating (crashed, killed, network partition), the Registry's reaper drops it after the TTL elapses; the next `GET /workers` simply won't include it — no special-case error handling needed in the Frontend for "stale worker." Expiry-based liveness beats trying to detect death directly, which is itself a small lesson.

## Error Handling

- **No live workers for the requested model** → Frontend returns `503` with a clear message, not a stack trace. This directly avoids a real bug class hit in production Dynamo (a frontend silently staying "Ready" with no models registered) — nano-dynamo's Frontend is designed to *not* reproduce that failure mode, called out explicitly in comments/README as "the bug class Chapter 1 avoids by design."
- **Worker dies mid-stream** (heartbeat still valid, but the `/generate` call itself fails or times out) → Frontend catches it and returns a clean error to the client rather than hanging or crashing. This does not wait for TTL reap — it's a distinct failure mode from "stopped heartbeating," handled where it's detected.
- **Registry restart** → all workers re-register on their next heartbeat failure (heartbeat gets a `404`, worker re-registers from scratch). No persistence — state is intentionally ephemeral, same spirit as etcd leases.

## Testing

A pytest suite using FastAPI's `TestClient` / `httpx.AsyncClient` to drive all three services in-process (no real subprocess orchestration needed for tests, even though the README's "try it yourself" instructions use three real terminals):

- **Registry:** register → appears in `/workers`; heartbeat refreshes it; TTL expiry actually removes it.
- **Worker:** registers on startup; `/generate` streams the expected number of fake tokens.
- **Frontend:** end-to-end request→response with one worker; `503` when zero workers; round-robin distributes across two-plus workers; a worker that stops responding mid-stream doesn't hang the client.

## Project Structure

```
nano-dynamo/
├── README.md                  # what this is, how to run it, the etcd/NATS simplification called out explicitly
├── pyproject.toml             # fastapi, uvicorn, pydantic, pytest, httpx
├── nano_dynamo/
│   ├── models.py              # ModelCard + shared request/response schemas
│   ├── registry/
│   │   └── main.py            # Registry FastAPI app
│   ├── worker/
│   │   ├── main.py            # Worker FastAPI app
│   │   └── engine.py          # Engine interface + MockEngine
│   └── frontend/
│       └── main.py            # Frontend FastAPI app
├── tests/
│   ├── test_registry.py
│   ├── test_worker.py
│   └── test_frontend.py
└── docs/
    └── appendix-bring-your-own-engine.md   # VLLMEngine sketch, explicitly optional
```

Each of the three services is runnable standalone (e.g. `uvicorn nano_dynamo.registry.main:app`), matching the README's "three real terminals" instructions. Chapters 2/3 add files alongside this (e.g. a `router.py` in `frontend/` for KV-aware routing, `prefill`/`decode` worker variants) rather than restructuring it.

This is a new standalone repository, separate from `ai-dynamo/dynamo`.

## Appendix: Bring Your Own Engine (stretch goal, not a core chapter)

The Worker's `Engine` interface — `async def generate(prompt: str) -> AsyncIterator[str]` — is deliberately the same shape as real Dynamo's backend contract: real Dynamo's Python backends (vLLM/SGLang) call into the Rust core via `register_model`/`fetch_model`, publish a model card, and stream tokens back, structurally the same contract nano-dynamo's `Engine` mirrors.

`MockEngine` is the only implementation used in Chapters 1–3, keeping the core chapters free of ML dependencies (mirroring real Dynamo's own `dynamo.mocker` backend, used for testing the orchestration layer without a real engine underneath). A real engine — e.g. `VLLMEngine` — is documented here as an optional, explicitly-marked extension:

```python
class VLLMEngine:
    def __init__(self, model_name: str):
        from vllm import LLM
        self.llm = LLM(model=model_name)

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        # wraps vLLM's generation call, yielding tokens as they're produced
        ...
```

Swapping `MockEngine` for `VLLMEngine` in the Worker process requires no change to the Frontend, Registry, or routing logic — that boundary is the point of the exercise. Not part of the core chapters; readers who want real inference can implement this themselves using the documented interface.

## Alternatives Considered

**Two processes, registry embedded in the Frontend.** Lower onboarding friction (one fewer terminal), and was the initial recommendation. Rejected because it silently assumes exactly one Frontend replica — there's no way to add a second Frontend sharing the same view of live workers, since each would have its own private registry state. This is a structurally different model from real Dynamo (where discovery is independent infrastructure) and would break Chapter 3's requirement that prefill workers, decode workers, and the Frontend all share one view of who's alive.

**Two processes, stdlib only (`http.server`), no pip installs.** Simplest possible setup, but stdlib's raw HTTP handling is clunky enough to obscure the lesson in boilerplate, and further diverges from real Dynamo (discovery isn't meaningfully separate from anything).

**Real inference engine (vLLM/SGLang) in the core Worker.** Considered and rejected for the core chapters: real vLLM needs a GPU in most configurations and would break the "clone it, `pip install`, run it anywhere" promise that makes a teaching repo actually get read. Preserved as an optional appendix instead (see above), consistent with real Dynamo shipping `dynamo.mocker` for the same reason.
