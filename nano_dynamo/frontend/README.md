# Frontend

The Frontend is the only client-facing service — it exposes an
OpenAI-compatible-ish `POST /v1/chat/completions` endpoint. On each request it
finds a live Worker for the requested model, forwards the request, and streams
the Worker's tokens straight back to the caller. It never generates anything
itself and never stores worker state; it's a router in front of the Registry
and the Workers.

## What a request does

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    workers = await settings.registry_client.list_workers(request.model)
    if not workers:
        raise HTTPException(status_code=503, detail=...)
    prompt = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
    worker = router.select(workers, prompt)
    ...
    return StreamingResponse(token_stream(), media_type="text/plain")
```

1. **Look up live workers** for the requested model, via the Registry (the
   Frontend has no worker list of its own — it asks every time, so a worker
   that was just reaped simply won't appear).
2. **503 if there are none** — see "Why the 503 matters" below.
3. **Flatten the chat messages** into one plain-text prompt.
4. **Pick a worker by cache affinity** (see below).
5. **Stream the worker's tokens back** through a `StreamingResponse`.

## KV-aware routing

Chapter 1 picked a worker round-robin. Chapter 2 replaces that one line with a
`KVRouter` (`router.py`) that tries to send each prompt to the worker most
likely to already hold its prefix in KV cache — a prefix already in cache
doesn't need recomputing, which is where the speedup comes from.

**Block-hash chains.** `block_hashes` splits a prompt into fixed-size blocks of
16 whitespace-separated words and chain-hashes them, folding each hash into the
next:

```python
h_0 = blake2b(""  + "\0" + block_0)     # covers block 0
h_1 = blake2b(h_0 + "\0" + block_1)     # covers blocks 0..1
```

Because `h_k` depends on every word before it, two prompts sharing `h_k` are
guaranteed to share their first `k+1` blocks word for word. Prefix overlap
becomes a hash comparison. The hash is `hashlib.blake2b`, never Python's
built-in `hash()` — that one is randomized per process, so cached prefixes
would stop matching after a restart.

**Predicting cache state (Approach A).** Workers report nothing. After routing,
`record(worker_id, prompt)` files every one of the prompt's block hashes under
that worker in `prefix_owners`, so the router predicts what each worker holds
purely from its own routing history.

**Selection.** `select` walks the prompt's hashes *backwards* — deepest prefix
first — and stops at the first one owned by a live worker, then breaks ties:

1. **Longest cached prefix** wins outright, even against an idler worker.
2. **Least in-flight load** among those, tracked by `acquire`/`release` around
   the stream. Without this, every request sharing a system prompt would pile
   onto one worker.
3. **Round-robin** among what's left — Chapter 1's behavior, now the bottom
   tier. With an empty cache map and idle workers, Chapter 2 reduces exactly
   to Chapter 1.

The Registry stays the source of truth for liveness: each level intersects with
the live worker list, so a reaped worker holding a deep prefix can never shadow
a live one holding a shallower prefix.

**Honest limitations.** This is a teaching implementation, and it predicts
rather than knows:

- **Affinity outranks load absolutely.** The tiers are strict: load only breaks
  ties among workers with *equal* cache overlap. If one worker owns a hot
  prefix, every request sharing it goes there no matter how deep its queue is,
  while an idle worker with a cold cache sits unused. Real Dynamo scores
  overlap and load on a single scale (`--router-kv-overlap-score-credit`), so
  enough load can outweigh a cache hit.
- **Block granularity is coarse.** Overlap counts only in whole 16-word blocks,
  so two prompts sharing 20 words but diverging at word 17 share just one
  block — the extra 4 words buy nothing.
- **Nothing ever evicts.** `prefix_owners` only grows. The router never learns
  that a worker's KV cache actually evicted those blocks, so a "hit" may be
  stale, and memory grows with unique prompt prefixes. `select` compensates
  only for *worker* death, by intersecting with the live list on every read.
- **Words are not tokens.** Real prefix caching works on tokenizer output; this
  approximates with `str.split()`, so block boundaries don't line up with a
  real engine's.
- **One Frontend assumed.** `prefix_owners` and `inflight` are in-process
  dicts. Run two Frontend replicas and each sees only its own traffic, so
  affinity degrades toward round-robin.

Real Dynamo avoids the guessing: workers publish KV cache events, so the router
tracks what's *actually* cached — including evictions — across any number of
frontends.

## Why the 503 matters

If no workers are registered for the requested model, the Frontend returns a
`503` with a clear message instead of accepting the request and hanging or
erroring obscurely. This is a direct callback to a real bug class in
production Dynamo — a frontend silently staying "Ready" with zero models
behind it. Because this check happens *before* any streaming starts, a real
HTTP status code is still available to return.

## Mid-stream worker failure

Once streaming has begun, the `200` status is already committed — there's no
way to retroactively turn it into an error code if a worker dies three tokens
in. So the Frontend catches any failure during streaming and appends a final
in-band error chunk instead:

```python
async def token_stream():
    try:
        async with worker_client.stream("POST", "/generate", json=...) as response:
            response.raise_for_status()
            async for chunk in response.aiter_text():
                yield chunk
    except Exception as exc:
        yield f"\n[error: worker unavailable: {exc}]"
    finally:
        router.release(worker.worker_id)
```

The client gets whatever partial output already streamed, followed by a clean
error marker — rather than a hung connection or a stack trace. The `except`
is deliberately broad because the failure can be a network error *or* an
exception propagating out of the worker's own generator.

The `finally` matters just as much: the worker's in-flight count was
incremented before the stream started, and a crash must still decrement it.
Otherwise a phantom count would make that worker look permanently busy and the
load tier would route around it forever.

## Dependency injection: `FrontendSettings`

```python
@dataclass
class FrontendSettings:
    registry_client: RegistryClientProtocol
    worker_client_factory: Callable[[str], httpx.AsyncClient]
```

The Frontend never constructs its own clients. Both dependencies are passed
in:

- `registry_client` — real `RegistryClient` in production, `FakeRegistryClient`
  in tests.
- `worker_client_factory` — given a worker's `endpoint_url`, return an HTTP
  client for it. In production it's
  `lambda url: httpx.AsyncClient(base_url=url)`; in tests it returns clients
  wired to in-process worker apps via `httpx.ASGITransport` (or, for the
  mid-stream-failure test, a `FlakyWorkerClient` that truly streams then
  raises — something `ASGITransport` can't reproduce, because it buffers the
  whole response before returning it).

This is what lets the Frontend be tested end-to-end with zero real sockets,
and it's the same seam a real deployment plugs genuine network clients into.

## Running it

```bash
python -m nano_dynamo.frontend.main   # defaults to port 8080, REGISTRY_URL=http://127.0.0.1:8000
```

See the top-level `README.md` for the full three-terminal walkthrough and the
`curl` example.
