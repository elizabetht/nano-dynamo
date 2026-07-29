# Worker

A Worker is what actually generates tokens in response to a prompt. In
Chapter 1 there's no real inference happening — every Worker runs a mocked
`Engine` instead of a real model, so the whole system can be run, read, and
tested with zero ML dependencies and no GPU.

*(This file currently covers `engine.py`. Once the Worker service itself —
registration, heartbeating, the `/generate` HTTP endpoint — is built, this
README will be extended to cover that too.)*

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

## Bringing a real engine later

A real engine (e.g. a `VLLMEngine`) would be a new class implementing the
same `Engine` protocol — not a modification to `MockEngine` — swapped in via
`WorkerSettings(engine=VLLMEngine(...))` instead of the default
`MockEngine()`. See the top-level `docs/appendix-bring-your-own-engine.md`
(coming in a later chapter) for a full sketch. This is explicitly a
stretch-goal appendix, not part of the core chapters.
