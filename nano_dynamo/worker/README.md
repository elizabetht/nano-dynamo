# Worker

A Worker is what actually generates tokens in response to a prompt. In
Chapter 1 there's no real inference happening — every Worker runs a mocked
`Engine` instead of a real model, so the whole system can be run, read, and
tested with zero ML dependencies and no GPU.

## `Engine`: the contract, not an implementation

```python
class Engine(Protocol):
    async def generate(self, prompt: str) -> AsyncIterator[str]: ...
```

`Engine` is never instantiated directly — it's a `Protocol`, a structural
type describing what any inference backend must look like: an async
`generate(prompt)` method that yields strings one at a time. The Worker's
code (coming next) is written against this contract, not against any
specific engine, so a real backend can be swapped in later without touching
the Worker's routes, registration logic, or heartbeat loop.

## `MockEngine`: the only implementation used in Chapters 1–3

```python
MockEngine(num_tokens=5, token_delay_seconds=0.05, token_text="token")
```

`generate(prompt)` completely ignores `prompt` and yields
`f"{token_text}_{i}"` for `i` in `range(num_tokens)`, pausing
`token_delay_seconds` between each one with a real `await asyncio.sleep(...)`.
The delay is what makes the *streaming* real — a caller consuming this with
`async for` genuinely receives one token at a time, spaced out, even though
the content is fake.

`token_text` exists purely for testability: it lets two different
`MockEngine`-backed Workers produce distinguishable output (`"from-a_0"` vs.
`"from-b_0"`), which is how later tests can prove a round-robin router is
really alternating between two different workers rather than always hitting
one.

## The Worker service (`main.py`)

`create_app(settings: WorkerSettings) -> FastAPI` builds the actual running
Worker. On startup (via a `lifespan` handler, same pattern as the Registry's
reaper) it calls `settings.registry_client.register(...)` *before* the app
is considered ready — so the app genuinely can't come up without having
successfully told the Registry it exists — then starts a background
heartbeat task. It exposes one route, `POST /generate`, which streams
`settings.engine.generate(request.prompt)` straight through as an HTTP
`StreamingResponse`.

### Self-healing heartbeats

The heartbeat logic is deliberately split into two functions:

```python
async def _send_heartbeat_or_reregister(settings: WorkerSettings, state) -> None:
    try:
        await settings.registry_client.heartbeat(state.worker_id)
    except WorkerNotFoundError:
        state.worker_id = await settings.registry_client.register(
            RegisterRequest(
                model_name=settings.model_name,
                endpoint_url=settings.endpoint_url,
                worker_type=settings.worker_type,
            )
        )


async def _heartbeat_loop(settings: WorkerSettings, state) -> None:
    while True:
        await asyncio.sleep(settings.heartbeat_interval_seconds)
        await _send_heartbeat_or_reregister(settings, state)
```

`_send_heartbeat_or_reregister` is *one heartbeat attempt*. The happy path
just confirms "I'm still alive" to the Registry. But if the Registry
responds `404` — meaning it no longer recognizes this `worker_id` — the
Worker doesn't crash or give up; it just registers itself again from
scratch and overwrites `state.worker_id` with whatever new ID comes back.

This is the "Registry restart → workers self-heal" behavior from the design
spec: the Registry keeps its worker list purely in memory with no
persistence, so if it restarts, every Worker's next heartbeat will get a
`404` (the Registry genuinely has no record of them anymore). Rather than
needing special "did the Registry restart?" detection, a `404` is treated
as one uniform signal to rejoin — whether the real cause is a Registry
restart, or the TTL reaper legitimately expiring a slow Worker, the recovery
is identical.

`_heartbeat_loop` is just "run that, forever, spaced out by
`heartbeat_interval_seconds`" — the part that's actually scheduled as a
background task in `lifespan`. The logic is split this way specifically so
tests can call `_send_heartbeat_or_reregister` directly, once, and assert
on the result, instead of needing to wait through real `asyncio.sleep()`
cycles inside a `while True:` loop to exercise the same behavior.

## Bringing a real engine later

A real engine (e.g. a `VLLMEngine`) would be a new class implementing the
same `Engine` protocol — not a modification to `MockEngine` — swapped in via
`WorkerSettings(engine=VLLMEngine(...))` instead of the default
`MockEngine()`. See the top-level `docs/appendix-bring-your-own-engine.md`
(coming in a later chapter) for a full sketch. This is explicitly a
stretch-goal appendix, not part of the core chapters.
