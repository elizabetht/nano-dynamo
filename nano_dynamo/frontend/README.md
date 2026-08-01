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
    worker = workers[next(counter) % len(workers)]
    ...
    return StreamingResponse(token_stream(), media_type="text/plain")
```

1. **Look up live workers** for the requested model, via the Registry (the
   Frontend has no worker list of its own — it asks every time, so a worker
   that was just reaped simply won't appear).
2. **503 if there are none** — see "Why the 503 matters" below.
3. **Pick one, round-robin** (see below).
4. **Flatten the chat messages** into one plain-text prompt. Chapter 1 doesn't
   need anything richer — `MockEngine` ignores the prompt entirely.
5. **Stream the worker's tokens back** through a `StreamingResponse`.

## Round-robin routing

```python
counter = count()                       # created once per app, in create_app
...
worker = workers[next(counter) % len(workers)]
```

`itertools.count()` yields `0, 1, 2, ...`. Because the counter is created once
in `create_app` and captured by the route via closure, every request through
one Frontend instance shares it — so requests fan out across the live workers
in turn (`% len(workers)` wraps the index to however many are currently alive).
This is the exact seam where Chapter 2's KV-cache-aware routing will later
replace the round-robin pick — nothing else about the Frontend changes.

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
```

The client gets whatever partial output already streamed, followed by a clean
error marker — rather than a hung connection or a stack trace. The `except`
is deliberately broad because the failure can be a network error *or* an
exception propagating out of the worker's own generator.

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
