# nano-dynamo Chapter 2: KV-Aware Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Frontend's round-robin worker selection with cache-aware routing — send each request to the worker most likely to already hold its prompt's prefix in KV cache — while leaving every other Chapter 1 behavior untouched.

**Architecture:** All routing intelligence goes into a new, pure-synchronous `KVRouter` (`nano_dynamo/frontend/router.py`) that hashes each prompt into a chain of block hashes, tracks which worker probably holds which prefix (predicted from its own routing history — "Approach A"), and picks the worker with the longest cached prefix overlap, breaking ties by least in-flight load and then round-robin. The Frontend swaps its one round-robin line for `router.select(...)` and wraps the stream in `acquire`/`record`/`release`.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, httpx, pytest, pytest-asyncio, `hashlib` (stdlib).

## Global Constraints

- Pure Python only — no new third-party dependencies (`hashlib` is stdlib). (spec: Tech Stack; Chapter 1 Global Constraints)
- `KVRouter` is pure synchronous logic — no `async`, no HTTP, no I/O. (spec: Components)
- Approach A only: the router predicts cache state from its own routing history; workers report nothing, no new worker→frontend channel. (spec: Where cache state lives)
- Tokens are approximated by whitespace-split words; no real tokenizer. Default block size 16 words. (spec: Non-Goals; Core concept)
- Use a **stable** hash (`hashlib`), never Python's built-in `hash()` (randomized per process via `PYTHONHASHSEED`). (implementation requirement for determinism across the block-hash chain)
- Everything else from Chapter 1 stays byte-for-byte the same: Registry, worker registration/heartbeat/self-heal, `/generate`, streaming, the 503-on-no-workers path, the `RegistryClient`. All Chapter 1 tests still pass, except `test_round_robins_across_two_workers`, whose assertion is updated in Task 5 (its old strict-alternation-on-identical-prompts behavior is exactly what Chapter 2 changes). (spec: What stays unchanged)
- The Registry remains the source of truth for liveness: `select` only ever returns a worker present in the live `workers` list it is given. (spec: The selection algorithm — subtlety)

---

## File Structure

```
nano_dynamo/
└── frontend/
    ├── main.py        # MODIFIED: build a KVRouter, replace round-robin with router.select,
    │                  #           wrap stream in acquire/record/release (try/finally)
    └── router.py      # NEW: KVRouter — block_hashes, select, record, acquire, release
tests/
├── test_router.py     # NEW: pure-logic KVRouter tests (Tasks 1-4)
└── test_frontend.py   # MODIFIED (Task 5): prefix-affinity, release-on-failure, updated round-robin test
```

Chapter 2 introduces exactly one new source file (`router.py`) and modifies `frontend/main.py`. `models.py`, `registry/`, `registry_client.py`, and the worker are not touched by the core chapter. Making the speedup *visible* is a described optional enhancement at the end (it touches more files and is not required for correct routing).

---

## Task 1: Block-hash chains

**Files:**
- Create: `nano_dynamo/frontend/router.py`
- Create: `tests/test_router.py`

**Interfaces:**
- Produces: `KVRouter(block_size: int = 16)`, `KVRouter.block_hashes(prompt: str) -> list[str]` — the chained list `[h_0, h_1, ..., h_n]` where each `h_k` is a hex digest that depends on words `0..k`. Consumed by every later task.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_router.py
from nano_dynamo.frontend.router import KVRouter


def test_block_hashes_chain_grows_one_per_block():
    router = KVRouter(block_size=2)
    # 5 words, block_size 2 -> blocks: [w0 w1][w2 w3][w4] -> 3 hashes
    hashes = router.block_hashes("a b c d e")
    assert len(hashes) == 3
    assert all(isinstance(h, str) for h in hashes)


def test_block_hashes_empty_prompt_is_empty():
    router = KVRouter(block_size=2)
    assert router.block_hashes("") == []
    assert router.block_hashes("   ") == []


def test_block_hashes_are_a_shared_prefix_chain():
    router = KVRouter(block_size=2)
    a = router.block_hashes("the quick brown fox")     # [the quick][brown fox]
    b = router.block_hashes("the quick brown dog")     # [the quick][brown dog]
    # First block matches (same words) -> same h_0; second differs -> different h_1.
    assert a[0] == b[0]
    assert a[1] != b[1]


def test_block_hashes_diverge_when_first_block_differs():
    router = KVRouter(block_size=2)
    a = router.block_hashes("the quick brown fox")
    b = router.block_hashes("a quick brown fox")
    # First block differs, and the chain folds the prefix in, so every later
    # hash differs too even though later words match.
    assert a[0] != b[0]
    assert a[1] != b[1]


def test_block_hashes_are_deterministic():
    # Stable hash (hashlib), not Python's randomized hash(): same input, same output.
    assert KVRouter().block_hashes("hello world foo") == KVRouter().block_hashes("hello world foo")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nano_dynamo.frontend.router'`

- [ ] **Step 3: Write `nano_dynamo/frontend/router.py`**

```python
# nano_dynamo/frontend/router.py
import hashlib


class KVRouter:
    """Cache-aware worker selection. Pure synchronous logic -- no async, no HTTP.

    Predicts what each worker has cached from the router's own routing history
    (Approach A): the block hashes of every prompt it has sent to a worker are
    recorded against that worker, and a new prompt is routed to the worker with
    the longest matching prefix.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size

    def block_hashes(self, prompt: str) -> list[str]:
        """Split the prompt into fixed-size word blocks and chain-hash them, so
        that hash h_k depends on words 0..k. A shared h_k between two prompts
        therefore proves they share a prefix through block k."""
        words = prompt.split()
        hashes: list[str] = []
        prev = ""
        for start in range(0, len(words), self.block_size):
            block = " ".join(words[start : start + self.block_size])
            prev = hashlib.blake2b((prev + "\x00" + block).encode()).hexdigest()
            hashes.append(prev)
        return hashes
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_router.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add nano_dynamo/frontend/router.py tests/test_router.py
git commit -m "feat: add KVRouter block-hash chains

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Task 2: In-flight load tracking (`acquire` / `release`)

**Files:**
- Modify: `nano_dynamo/frontend/router.py`
- Modify: `tests/test_router.py`

**Interfaces:**
- Consumes: `KVRouter` (Task 1).
- Produces: `KVRouter.inflight: dict[str, int]` (worker_id → current in-flight count, absent means 0), `KVRouter.acquire(worker_id: str) -> None` (increment), `KVRouter.release(worker_id: str) -> None` (decrement, never below 0). Consumed by Task 3 (`select` tie-break) and Task 5 (Frontend wraps the stream).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_router.py  (append)
def test_acquire_and_release_track_inflight():
    router = KVRouter()
    assert router.inflight.get("w1", 0) == 0
    router.acquire("w1")
    router.acquire("w1")
    assert router.inflight["w1"] == 2
    router.release("w1")
    assert router.inflight["w1"] == 1


def test_release_never_goes_negative():
    router = KVRouter()
    router.release("w1")  # release with nothing acquired
    assert router.inflight.get("w1", 0) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_router.py -v`
Expected: FAIL with `AttributeError: 'KVRouter' object has no attribute 'inflight'`

- [ ] **Step 3: Add in-flight tracking to `KVRouter`**

Replace `__init__` and add the two methods:

```python
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.inflight: dict[str, int] = {}
```

```python
    def acquire(self, worker_id: str) -> None:
        self.inflight[worker_id] = self.inflight.get(worker_id, 0) + 1

    def release(self, worker_id: str) -> None:
        self.inflight[worker_id] = max(0, self.inflight.get(worker_id, 0) - 1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_router.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add nano_dynamo/frontend/router.py tests/test_router.py
git commit -m "feat: track in-flight load per worker in KVRouter

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Task 3: `select` — cache affinity, least-load, then round-robin

**Files:**
- Modify: `nano_dynamo/frontend/router.py`
- Modify: `tests/test_router.py`

**Interfaces:**
- Consumes: `KVRouter`, `block_hashes` (Task 1), `inflight`/`acquire` (Task 2); `ModelCard` from `nano_dynamo.models` (Chapter 1) — fields used: `worker_id`, `endpoint_url`, `model_name`.
- Produces: `KVRouter.select(workers: list[ModelCard], prompt: str) -> ModelCard`. Assumes `workers` is non-empty (the Frontend already 503s on empty). Uses an internal `prefix_owners: dict[str, set[str]]` (block-hash → worker_ids) that starts empty and is filled by `record` (Task 4), and a round-robin cursor `_rr` for breaking equal-load ties. Consumed by Task 5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_router.py  (append)
from nano_dynamo.models import ModelCard


def _card(worker_id: str) -> ModelCard:
    return ModelCard(worker_id=worker_id, model_name="demo", endpoint_url=f"http://{worker_id}")


def test_select_cold_prompt_picks_least_loaded():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    router.acquire("w1")  # w1 busier
    chosen = router.select(workers, "brand new prompt")
    assert chosen.worker_id == "w2"


def test_select_prefers_worker_holding_the_prefix():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    # Pretend w1 already served this exact prompt (Task 4 does this for real).
    for h in router.block_hashes("shared prefix here"):
        router.prefix_owners.setdefault(h, set()).add("w1")
    chosen = router.select(workers, "shared prefix here")
    assert chosen.worker_id == "w1"


def test_select_ties_broken_by_least_load():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    # Both workers hold the prefix; w1 is busier, so w2 wins.
    for h in router.block_hashes("shared prefix here"):
        router.prefix_owners.setdefault(h, set()).update({"w1", "w2"})
    router.acquire("w1")
    chosen = router.select(workers, "shared prefix here")
    assert chosen.worker_id == "w2"


def test_select_longer_prefix_overlap_wins():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    hashes = router.block_hashes("a b c d e f")  # 3 blocks
    # w1 shares only block 0; w2 shares blocks 0 and 1 -> w2 has the longer prefix.
    router.prefix_owners.setdefault(hashes[0], set()).update({"w1", "w2"})
    router.prefix_owners.setdefault(hashes[1], set()).add("w2")
    chosen = router.select(workers, "a b c d e f")
    assert chosen.worker_id == "w2"


def test_select_ignores_reaped_workers_in_prefix_owners():
    router = KVRouter(block_size=2)
    workers = [_card("w2")]  # only w2 is live now
    # prefix_owners still names w1 (since reaped), but it isn't in `workers`.
    for h in router.block_hashes("shared prefix here"):
        router.prefix_owners.setdefault(h, set()).add("w1")
    chosen = router.select(workers, "shared prefix here")
    assert chosen.worker_id == "w2"  # falls back to the live worker


def test_select_round_robins_among_equally_loaded_cold_candidates():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    # Distinct cold prompts, both workers idle -> spread round-robin, not piled
    # onto the first-listed worker. (select does not acquire, so load stays 0.)
    a = router.select(workers, "alpha beta")
    b = router.select(workers, "gamma delta")
    assert {a.worker_id, b.worker_id} == {"w1", "w2"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_router.py -v`
Expected: FAIL — `prefix_owners` attribute missing / `select` not defined.

- [ ] **Step 3: Implement `prefix_owners`, `_rr`, and `select`**

Replace `__init__` to add the two new fields:

```python
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.inflight: dict[str, int] = {}
        self.prefix_owners: dict[str, set[str]] = {}
        self._rr = 0  # round-robin cursor for breaking equal-load ties
```

Add the method:

```python
    def select(self, workers, prompt: str):
        """Pick the live worker with the longest cached-prefix overlap, breaking
        ties by least in-flight load and then round-robin. `workers` is assumed
        non-empty; the Frontend 503s before calling this."""
        live_ids = {w.worker_id for w in workers}
        hashes = self.block_hashes(prompt)

        # Longest prefix first: the last hash any live worker holds gives the
        # best cache-hit candidates.
        candidates: set[str] = set()
        for h in reversed(hashes):
            owners = self.prefix_owners.get(h, set()) & live_ids
            if owners:
                candidates = owners
                break

        # Cold prompt (or only reaped owners): every live worker is a candidate.
        if not candidates:
            candidates = live_ids

        # Least in-flight load, preserving input order for stable round-robin.
        ranked = [w for w in workers if w.worker_id in candidates]
        min_load = min(self.inflight.get(w.worker_id, 0) for w in ranked)
        least_loaded = [w for w in ranked if self.inflight.get(w.worker_id, 0) == min_load]

        chosen = least_loaded[self._rr % len(least_loaded)]
        self._rr += 1
        return chosen
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_router.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add nano_dynamo/frontend/router.py tests/test_router.py
git commit -m "feat: add KVRouter.select with cache affinity, load and round-robin

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Task 4: `record` — learn cache state from routing history

**Files:**
- Modify: `nano_dynamo/frontend/router.py`
- Modify: `tests/test_router.py`

**Interfaces:**
- Consumes: `KVRouter`, `block_hashes`, `prefix_owners`, `select` (Tasks 1-3).
- Produces: `KVRouter.record(worker_id: str, prompt: str) -> None` — adds every block hash of `prompt` to `prefix_owners` under `worker_id`. This is the "Approach A" learning step. Consumed by Task 5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_router.py  (append)
def test_record_then_same_prompt_routes_to_same_worker():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    first = router.select(workers, "keep this together")
    router.record(first.worker_id, "keep this together")
    # A second identical prompt should now prefer the worker that served it.
    second = router.select(workers, "keep this together")
    assert second.worker_id == first.worker_id


def test_record_makes_prefix_sharing_prompt_route_together():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    router.record("w1", "the quick brown fox")
    # Shares the first two blocks ("the quick", "brown ...") -> should hit w1,
    # even though w2 is equally idle.
    chosen = router.select(workers, "the quick brown dog")
    assert chosen.worker_id == "w1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_router.py -v`
Expected: FAIL with `AttributeError: 'KVRouter' object has no attribute 'record'`

- [ ] **Step 3: Implement `record`**

```python
    def record(self, worker_id: str, prompt: str) -> None:
        """Remember that `worker_id` has now processed `prompt`, so future
        prefix-sharing prompts route back to it (Approach A prediction)."""
        for h in self.block_hashes(prompt):
            self.prefix_owners.setdefault(h, set()).add(worker_id)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_router.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add nano_dynamo/frontend/router.py tests/test_router.py
git commit -m "feat: add KVRouter.record to learn cache state from routing

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Task 5: Wire `KVRouter` into the Frontend

**Files:**
- Modify: `nano_dynamo/frontend/main.py`
- Modify: `tests/test_frontend.py`

**Interfaces:**
- Consumes: `KVRouter` with `select`, `record`, `acquire`, `release` (Tasks 1-4); the existing `FrontendSettings`, `create_app`, `worker_client_factory`, and the `token_stream` structure (Chapter 1); `FlakyWorkerClient`, `FakeRegistryClient`, `MockEngine`, `_worker_app`, `_factory_for`, `_card`, `_chat_request` (already imported/defined at the top of `tests/test_frontend.py` from Chapter 1).
- Produces: `create_app` now builds a `KVRouter` internally (exposed as `app.state.kv_router` for tests) and routes through it instead of the round-robin `counter`. Prefix-sharing requests route to the same worker; a worker's in-flight count is incremented before its stream and decremented in a `finally` so it never leaks on failure.

- [ ] **Step 1: Write the failing tests, and update the Chapter 1 round-robin test**

Append two new tests:

```python
# tests/test_frontend.py  (append)
def test_prefix_sharing_requests_route_to_the_same_worker():
    registry_client = FakeRegistryClient()
    registry_client.set_workers(
        "demo", [_card("worker-a", "http://worker-a"), _card("worker-b", "http://worker-b")]
    )
    worker_a = _worker_app("http://worker-a", MockEngine(num_tokens=1, token_delay_seconds=0, token_text="from-a"))
    worker_b = _worker_app("http://worker-b", MockEngine(num_tokens=1, token_delay_seconds=0, token_text="from-b"))
    app = create_app(
        FrontendSettings(
            registry_client=registry_client,
            worker_client_factory=_factory_for(
                {"http://worker-a": worker_a, "http://worker-b": worker_b}
            ),
        )
    )

    def send(text: str) -> str:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "demo", "messages": [{"role": "user", "content": text}]},
            ) as response:
                return "".join(response.iter_text())

    first = send("the quick brown fox jumps")
    second = send("the quick brown fox runs")  # shares the leading blocks
    assert first == second  # same worker's distinctive token_text


def test_worker_failure_mid_stream_still_releases_inflight():
    registry_client = FakeRegistryClient()
    registry_client.set_workers("demo", [_card("worker-a", "http://worker-a")])
    app = create_app(
        FrontendSettings(
            registry_client=registry_client,
            worker_client_factory=lambda endpoint_url: FlakyWorkerClient(),
        )
    )
    with TestClient(app) as client:
        with client.stream("POST", "/v1/chat/completions", json=_chat_request()) as response:
            body = "".join(response.iter_text())
    assert "[error: worker unavailable" in body
    # release() must have run in the finally, so nothing leaks.
    assert all(count == 0 for count in app.state.kv_router.inflight.values())
```

Then change the existing `test_round_robins_across_two_workers`: it currently sends the *same* prompt twice and asserts the two workers alternate. With cache-aware routing an identical prompt intentionally routes to the *same* worker, so replace its two identical sends with two prompts that share **no** prefix (different first word), which stay cold and spread round-robin. Replace the request bodies so the two calls use distinct content, e.g.:

```python
    # was: both requests used _chat_request() (identical "hi")
    # now: two prompts with no shared prefix, so routing stays load/round-robin based
    with TestClient(app) as client:
        bodies = []
        for content in ("alpha one two", "zeta three four"):
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "demo", "messages": [{"role": "user", "content": content}]},
            ) as response:
                bodies.append("".join(response.iter_text()))

    assert set(bodies) == {"from-a_0", "from-b_0"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_frontend.py -v`
Expected: FAIL — `app.state.kv_router` does not exist, and prefix-sharing requests do not yet route together.

- [ ] **Step 3: Route through `KVRouter` in `create_app`**

In `nano_dynamo/frontend/main.py`: add the import, drop `from itertools import count`, and replace the `create_app` body:

```python
from nano_dynamo.frontend.router import KVRouter
```

```python
def create_app(settings: FrontendSettings) -> FastAPI:
    app = FastAPI()
    router = KVRouter()
    app.state.kv_router = router  # exposed for tests/introspection

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        workers = await settings.registry_client.list_workers(request.model)
        if not workers:
            raise HTTPException(
                status_code=503,
                detail=f"No live workers registered for model '{request.model}'",
            )
        prompt = "\n".join(f"{message.role}: {message.content}" for message in request.messages)
        worker = router.select(workers, prompt)
        router.record(worker.worker_id, prompt)
        router.acquire(worker.worker_id)
        worker_client = settings.worker_client_factory(worker.endpoint_url)

        async def token_stream():
            # release in finally so a mid-stream failure never leaks the count.
            try:
                async with worker_client.stream(
                    "POST", "/generate", json=GenerateRequest(prompt=prompt).model_dump()
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_text():
                        yield chunk
            except Exception as exc:
                yield f"\n[error: worker unavailable: {exc}]"
            finally:
                router.release(worker.worker_id)

        return StreamingResponse(token_stream(), media_type="text/plain")

    return app
```

- [ ] **Step 4: Run the frontend tests**

Run: `pytest tests/test_frontend.py -v`
Expected: all pass — the 503, single-worker-stream, updated round-robin, mid-stream-error, and two new tests.

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all pass — Chapter 1 tests, the 15 router tests, and the frontend tests.

- [ ] **Step 6: Commit**

```bash
git add nano_dynamo/frontend/main.py tests/test_frontend.py
git commit -m "feat: route through KVRouter for cache-aware worker selection

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Task 6: Documentation

**Files:**
- Modify: `nano_dynamo/frontend/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing programmatically — documents Tasks 1-5.

- [ ] **Step 1: Update the Frontend README's routing section**

In `nano_dynamo/frontend/README.md`, replace the "Round-robin routing" section with a "KV-aware routing" section explaining: block-hash chains, that the router predicts cache state from its own routing history (Approach A), longest-prefix-overlap selection with least-load-then-round-robin tie-break, and that this is the seam Chapter 1's round-robin used to occupy. Add the honest caveat that Approach A assumes a single Frontend and real Dynamo uses worker-reported KV events. Keep it to ~25 lines, matching the file's tone.

- [ ] **Step 2: Update the top-level README**

In `README.md`: change the services table and any prose that calls the Frontend "round-robin" to "KV-cache-aware" (keeping historical references like "Chapter 1's round-robin" intact). In "What's next", remove the Chapter 2 line, leaving Chapter 3 (disaggregated prefill/decode). Update the "Pin the version" note: `0.1.x` is round-robin, `0.2.x` is KV-aware routing.

- [ ] **Step 3: Verify docs match code**

Run: `grep -rn "round-robin" README.md nano_dynamo/frontend/README.md`
Expected: only historical references (e.g. "Chapter 1's round-robin", "0.1.x is round-robin"), no claim that the *current* Frontend selects round-robin.

- [ ] **Step 4: Commit**

```bash
git add README.md nano_dynamo/frontend/README.md
git commit -m "docs: describe KV-aware routing (Chapter 2)

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Optional enhancement (not a core task): make the cache hit visible

The core chapter gives *correct* routing but not a *visible* payoff — a reader
can't feel the speedup because `MockEngine` takes the same time either way. Making
it visible is a genuine vertical slice touching five files, so it's described
here rather than shipped as a core task; do it only if you want the demo.

The slice:
1. `nano_dynamo/models.py` — add `cache_hit_blocks: int = 0` to `GenerateRequest` (backward compatible; default 0).
2. `nano_dynamo/frontend/router.py` — add `cached_prefix_len(worker_id: str, prompt: str) -> int`, counting the leading block hashes the worker already holds (call it *before* `record`, or the count is always full).
3. `nano_dynamo/frontend/main.py` — after `select`, compute `overlap = router.cached_prefix_len(worker.worker_id, prompt)` and send `GenerateRequest(prompt=prompt, cache_hit_blocks=overlap)`.
4. `nano_dynamo/worker/engine.py` — the `Engine` protocol's `generate` gains an optional `cache_hit_blocks: int = 0`; `MockEngine` skips a new `prefill_delay_seconds` when it's nonzero; **`VLLMEngine.generate` must also accept and ignore `cache_hit_blocks`** so the shared protocol call site works.
5. `nano_dynamo/worker/main.py` — `/generate` passes `request.cache_hit_blocks` into `app.state.engine.generate(...)`.

Because it changes the `Engine` protocol signature and `GenerateRequest`, treat it
as its own mini-plan (TDD each file) rather than folding it into Chapter 2. The
routing correctness of Chapter 2 does not depend on it.

---

## Note on releasing

This plan does not cut a release. When Chapter 2 is complete and you want it on
PyPI, tag `v0.2.0` (setuptools-scm derives the version from the tag) and push the
tag — the publish workflow ships it. `0.2.0` is the correct version: KV-aware
routing is the Chapter 2 behavior change the README reserved `0.2.0` for.
