# nano-dynamo Chapter 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build nano-dynamo Chapter 1 — a Registry, Worker, and Frontend, each a standalone FastAPI process, that together demonstrate service discovery, heartbeat-based liveness, and request routing with real streaming, using a mocked inference engine.

**Architecture:** Three independent FastAPI apps (Registry, Worker, Frontend) that only ever talk to each other over HTTP through a single shared `RegistryClient` abstraction. Each app is built by a `create_app(settings)` factory that takes its dependencies (registry client, engine, HTTP client factory) as explicit arguments, so every test can substitute fakes without real sockets — production wiring (real `httpx.AsyncClient`s, real `uvicorn` processes) only happens in the actual `if __name__ == "__main__"` entry points, never inside the app factories themselves.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, uvicorn, httpx, pytest, pytest-asyncio.

## Global Constraints

- Pure Python only — no Rust, no compiled extensions. (spec: Language & Stack)
- All three services use FastAPI + Pydantic + uvicorn. (spec: Language & Stack)
- Zero ML dependencies in Chapter 1 — the only `Engine` implementation is `MockEngine`; real engines are documented only as an optional, separate appendix. (spec: Appendix, Alternatives Considered)
- Registry, Worker, and Frontend are three independent processes communicating only over HTTP — the Registry is never embedded inside the Frontend. (spec: Architecture)
- `ModelCard` (and other shared schemas) are defined exactly once, in `nano_dynamo/models.py`, imported by all three services. (spec: Components)
- Registry state is in-memory only — no persistence, no database. (spec: Error Handling)
- This lives in its own standalone repo (`nano-dynamo`), separate from `ai-dynamo/dynamo`. (spec: Project Structure — already satisfied by this repo)

---

## File Structure

```
nano-dynamo/
├── pyproject.toml
├── README.md
├── nano_dynamo/
│   ├── __init__.py
│   ├── models.py                 # ModelCard + all shared request/response schemas
│   ├── registry_client.py        # RegistryClientProtocol, RegistryClient, WorkerNotFoundError
│   ├── registry/
│   │   ├── __init__.py
│   │   └── main.py                # Registry create_app/RegistrySettings
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── engine.py               # Engine protocol + MockEngine
│   │   └── main.py                 # Worker create_app/WorkerSettings
│   └── frontend/
│       ├── __init__.py
│       └── main.py                 # Frontend create_app/FrontendSettings
├── tests/
│   ├── fakes.py                    # FakeRegistryClient, FlakyEngine test doubles
│   ├── test_models.py
│   ├── test_registry.py
│   ├── test_registry_client.py
│   ├── test_worker_engine.py
│   ├── test_worker.py
│   └── test_frontend.py
└── docs/
    ├── superpowers/specs/2026-07-25-nano-dynamo-design.md   (already exists)
    ├── superpowers/plans/2026-07-27-nano-dynamo-chapter-1.md (this file)
    └── appendix-bring-your-own-engine.md
```

**Why a `RegistryClientProtocol` instead of each service calling `httpx` directly:** the Registry is the one dependency both Worker and Frontend share. Defining it as a `Protocol` with two implementations (`RegistryClient` over real HTTP, `FakeRegistryClient` in-memory for tests) means Worker and Frontend unit tests never need a real running Registry process — each can be tested completely in isolation, which is the "can be tested independently" property called out in the design's isolation principles.

---

## Task 1: Project Scaffolding + Shared Models

**Files:**
- Create: `pyproject.toml`
- Create: `nano_dynamo/__init__.py`
- Create: `nano_dynamo/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Produces: `ModelCard(worker_id, model_name, endpoint_url, worker_type="aggregated", registered_at, last_heartbeat)`, `RegisterRequest(model_name, endpoint_url, worker_type="aggregated")`, `RegisterResponse(worker_id)`, `GenerateRequest(prompt)`, `ChatMessage(role, content)`, `ChatCompletionRequest(model, messages)` — all Pydantic `BaseModel`s, all consumed by every later task.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "nano-dynamo"
version = "0.1.0"
description = "A minimal, from-scratch teaching implementation of Dynamo's orchestration layer."
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.32",
    "pydantic>=2.9",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.setuptools.packages.find]
include = ["nano_dynamo*"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Create empty package `__init__.py`**

```python
# nano_dynamo/__init__.py
```

- [ ] **Step 3: Install the project in editable mode**

Run: `pip install -e ".[dev]"`
Expected: `Successfully installed nano-dynamo-0.1.0`

- [ ] **Step 4: Write the failing test for the shared models**

```python
# tests/test_models.py
import pytest
from pydantic import ValidationError

from nano_dynamo.models import (
    ChatCompletionRequest,
    ChatMessage,
    GenerateRequest,
    ModelCard,
    RegisterRequest,
    RegisterResponse,
)


def test_model_card_defaults_to_aggregated_worker_type():
    card = ModelCard(worker_id="w1", model_name="demo", endpoint_url="http://worker-a")
    assert card.worker_type == "aggregated"
    assert card.registered_at is not None
    assert card.last_heartbeat is not None


def test_model_card_rejects_unknown_worker_type():
    with pytest.raises(ValidationError):
        ModelCard(
            worker_id="w1",
            model_name="demo",
            endpoint_url="http://worker-a",
            worker_type="not-a-real-type",
        )


def test_register_request_defaults_to_aggregated():
    request = RegisterRequest(model_name="demo", endpoint_url="http://worker-a")
    assert request.worker_type == "aggregated"


def test_register_response_round_trips_worker_id():
    response = RegisterResponse(worker_id="w1")
    assert RegisterResponse.model_validate(response.model_dump()).worker_id == "w1"


def test_chat_completion_request_requires_model_and_messages():
    request = ChatCompletionRequest(
        model="demo", messages=[ChatMessage(role="user", content="hi")]
    )
    assert request.messages[0].content == "hi"
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])


def test_generate_request_holds_prompt():
    assert GenerateRequest(prompt="hello").prompt == "hello"
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nano_dynamo.models'`

- [ ] **Step 6: Write `nano_dynamo/models.py`**

```python
# nano_dynamo/models.py
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

WorkerType = Literal["aggregated", "prefill", "decode"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelCard(BaseModel):
    worker_id: str
    model_name: str
    endpoint_url: str
    worker_type: WorkerType = "aggregated"
    registered_at: datetime = Field(default_factory=_utcnow)
    last_heartbeat: datetime = Field(default_factory=_utcnow)


class RegisterRequest(BaseModel):
    model_name: str
    endpoint_url: str
    worker_type: WorkerType = "aggregated"


class RegisterResponse(BaseModel):
    worker_id: str


class GenerateRequest(BaseModel):
    prompt: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: 6 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml nano_dynamo/__init__.py nano_dynamo/models.py tests/test_models.py
git commit -m "feat: add project scaffolding and shared Pydantic models

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Task 2: Registry Service

**Files:**
- Create: `nano_dynamo/registry/__init__.py`
- Create: `nano_dynamo/registry/main.py`
- Create: `tests/test_registry.py`

**Interfaces:**
- Consumes: `ModelCard`, `RegisterRequest`, `RegisterResponse` from `nano_dynamo.models` (Task 1).
- Produces: `RegistrySettings(heartbeat_ttl_seconds=15.0, reaper_interval_seconds=5.0)`, `create_app(settings: RegistrySettings | None = None) -> FastAPI` exposing `POST /register`, `POST /heartbeat/{worker_id}`, `GET /workers?model=...`. Consumed directly by Task 3's `RegistryClient` tests and by real deployment in the README.

- [ ] **Step 1: Create empty `nano_dynamo/registry/__init__.py`**

```python
# nano_dynamo/registry/__init__.py
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_registry.py
import asyncio

from starlette.testclient import TestClient

from nano_dynamo.registry.main import RegistrySettings, create_app


def test_register_then_appears_in_workers_list():
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/register",
            json={"model_name": "demo", "endpoint_url": "http://worker-a"},
        )
        assert response.status_code == 200
        worker_id = response.json()["worker_id"]

        listed = client.get("/workers", params={"model": "demo"}).json()
        assert [w["worker_id"] for w in listed] == [worker_id]


def test_workers_list_filters_by_model():
    app = create_app()
    with TestClient(app) as client:
        client.post("/register", json={"model_name": "demo", "endpoint_url": "http://worker-a"})
        client.post("/register", json={"model_name": "other", "endpoint_url": "http://worker-b"})

        listed = client.get("/workers", params={"model": "demo"}).json()
        assert [w["model_name"] for w in listed] == ["demo"]


def test_heartbeat_refreshes_known_worker():
    app = create_app()
    with TestClient(app) as client:
        worker_id = client.post(
            "/register", json={"model_name": "demo", "endpoint_url": "http://worker-a"}
        ).json()["worker_id"]

        response = client.post(f"/heartbeat/{worker_id}")
        assert response.status_code == 204


def test_heartbeat_on_unknown_worker_returns_404():
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/heartbeat/does-not-exist")
        assert response.status_code == 404


def test_reaper_expires_workers_past_ttl():
    app = create_app(
        RegistrySettings(heartbeat_ttl_seconds=0.05, reaper_interval_seconds=0.02)
    )
    with TestClient(app) as client:
        client.post("/register", json={"model_name": "demo", "endpoint_url": "http://worker-a"})
        assert len(client.get("/workers", params={"model": "demo"}).json()) == 1

        asyncio.run(asyncio.sleep(0.15))

        assert client.get("/workers", params={"model": "demo"}).json() == []
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nano_dynamo.registry.main'`

- [ ] **Step 4: Write `nano_dynamo/registry/main.py`**

```python
# nano_dynamo/registry/main.py
import asyncio
import contextlib
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from nano_dynamo.models import ModelCard, RegisterRequest, RegisterResponse


@dataclass
class RegistrySettings:
    heartbeat_ttl_seconds: float = 15.0
    reaper_interval_seconds: float = 5.0


def create_app(settings: RegistrySettings | None = None) -> FastAPI:
    settings = settings or RegistrySettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        reaper_task = asyncio.create_task(_reap_loop(app, settings))
        yield
        reaper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper_task

    app = FastAPI(lifespan=lifespan)
    app.state.workers = {}
    app.state.lock = asyncio.Lock()

    @app.post("/register", response_model=RegisterResponse)
    async def register(request: RegisterRequest) -> RegisterResponse:
        worker_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        card = ModelCard(
            worker_id=worker_id,
            model_name=request.model_name,
            endpoint_url=request.endpoint_url,
            worker_type=request.worker_type,
            registered_at=now,
            last_heartbeat=now,
        )
        async with app.state.lock:
            app.state.workers[worker_id] = card
        return RegisterResponse(worker_id=worker_id)

    @app.post("/heartbeat/{worker_id}", status_code=204)
    async def heartbeat(worker_id: str) -> None:
        async with app.state.lock:
            card = app.state.workers.get(worker_id)
            if card is None:
                raise HTTPException(status_code=404, detail="worker not registered")
            card.last_heartbeat = datetime.now(timezone.utc)

    @app.get("/workers", response_model=list[ModelCard])
    async def list_workers(model: str | None = None) -> list[ModelCard]:
        async with app.state.lock:
            cards = list(app.state.workers.values())
        if model is not None:
            cards = [card for card in cards if card.model_name == model]
        return cards

    return app


async def _reap_loop(app: FastAPI, settings: RegistrySettings) -> None:
    while True:
        await asyncio.sleep(settings.reaper_interval_seconds)
        cutoff = datetime.now(timezone.utc).timestamp() - settings.heartbeat_ttl_seconds
        async with app.state.lock:
            stale_ids = [
                worker_id
                for worker_id, card in app.state.workers.items()
                if card.last_heartbeat.timestamp() < cutoff
            ]
            for worker_id in stale_ids:
                del app.state.workers[worker_id]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_registry.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add nano_dynamo/registry tests/test_registry.py
git commit -m "feat: add Registry service with heartbeat TTL expiry

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Task 3: Registry Client (shared by Worker and Frontend)

**Files:**
- Create: `nano_dynamo/registry_client.py`
- Create: `tests/fakes.py`
- Create: `tests/test_registry_client.py`

**Interfaces:**
- Consumes: Registry's `POST /register`, `POST /heartbeat/{worker_id}`, `GET /workers` (Task 2); `ModelCard`, `RegisterRequest`, `RegisterResponse` (Task 1).
- Produces: `RegistryClientProtocol` (methods: `async register(request: RegisterRequest) -> str`, `async heartbeat(worker_id: str) -> None`, `async list_workers(model_name: str) -> list[ModelCard]`), `RegistryClient(http_client: httpx.AsyncClient)` implementing it over real HTTP, `WorkerNotFoundError(Exception)` raised by `heartbeat()` on a 404. `tests/fakes.py` produces `FakeRegistryClient` — an in-memory implementation of the same protocol — consumed by Task 5 and Task 6's tests.

- [ ] **Step 1: Write the failing tests for `RegistryClient`**

```python
# tests/test_registry_client.py
import httpx
import pytest

from nano_dynamo.models import RegisterRequest
from nano_dynamo.registry.main import create_app
from nano_dynamo.registry_client import RegistryClient, WorkerNotFoundError


@pytest.fixture
async def client():
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://registry"
    ) as http_client:
        yield RegistryClient(http_client)


async def test_register_returns_worker_id(client):
    worker_id = await client.register(
        RegisterRequest(model_name="demo", endpoint_url="http://worker-a")
    )
    assert worker_id


async def test_heartbeat_on_registered_worker_succeeds(client):
    worker_id = await client.register(
        RegisterRequest(model_name="demo", endpoint_url="http://worker-a")
    )
    await client.heartbeat(worker_id)  # should not raise


async def test_heartbeat_on_unknown_worker_raises_worker_not_found(client):
    with pytest.raises(WorkerNotFoundError):
        await client.heartbeat("does-not-exist")


async def test_list_workers_filters_by_model(client):
    await client.register(RegisterRequest(model_name="demo", endpoint_url="http://worker-a"))
    await client.register(RegisterRequest(model_name="other", endpoint_url="http://worker-b"))

    cards = await client.list_workers("demo")
    assert [card.model_name for card in cards] == ["demo"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_registry_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nano_dynamo.registry_client'`

- [ ] **Step 3: Write `nano_dynamo/registry_client.py`**

```python
# nano_dynamo/registry_client.py
from typing import Protocol

import httpx

from nano_dynamo.models import ModelCard, RegisterRequest, RegisterResponse


class WorkerNotFoundError(Exception):
    """Raised when a heartbeat targets a worker_id the Registry no longer knows about."""


class RegistryClientProtocol(Protocol):
    async def register(self, request: RegisterRequest) -> str: ...
    async def heartbeat(self, worker_id: str) -> None: ...
    async def list_workers(self, model_name: str) -> list[ModelCard]: ...


class RegistryClient:
    def __init__(self, http_client: httpx.AsyncClient):
        self._http = http_client

    async def register(self, request: RegisterRequest) -> str:
        response = await self._http.post("/register", json=request.model_dump(mode="json"))
        response.raise_for_status()
        return RegisterResponse.model_validate(response.json()).worker_id

    async def heartbeat(self, worker_id: str) -> None:
        response = await self._http.post(f"/heartbeat/{worker_id}")
        if response.status_code == 404:
            raise WorkerNotFoundError(worker_id)
        response.raise_for_status()

    async def list_workers(self, model_name: str) -> list[ModelCard]:
        response = await self._http.get("/workers", params={"model": model_name})
        response.raise_for_status()
        return [ModelCard.model_validate(item) for item in response.json()]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_registry_client.py -v`
Expected: 4 passed

- [ ] **Step 5: Write the `FakeRegistryClient` test double**

This is test infrastructure, not production code under test, so it has no failing-test step of its own — it exists to satisfy `RegistryClientProtocol` for Task 5 and Task 6's tests.

```python
# tests/fakes.py
from nano_dynamo.models import ModelCard, RegisterRequest
from nano_dynamo.registry_client import WorkerNotFoundError


class FakeRegistryClient:
    """In-memory stand-in for RegistryClient, so Worker/Frontend tests
    never need a real Registry process."""

    def __init__(self):
        self.registered: list[RegisterRequest] = []
        self.heartbeats: list[str] = []
        self._next_id = 0
        self._fail_heartbeat_for: set[str] = set()
        self._cards_by_model: dict[str, list[ModelCard]] = {}

    async def register(self, request: RegisterRequest) -> str:
        self._next_id += 1
        worker_id = f"worker-{self._next_id}"
        self.registered.append(request)
        return worker_id

    async def heartbeat(self, worker_id: str) -> None:
        self.heartbeats.append(worker_id)
        if worker_id in self._fail_heartbeat_for:
            raise WorkerNotFoundError(worker_id)

    def fail_next_heartbeat_for(self, worker_id: str) -> None:
        self._fail_heartbeat_for.add(worker_id)

    def set_workers(self, model_name: str, cards: list[ModelCard]) -> None:
        self._cards_by_model[model_name] = cards

    async def list_workers(self, model_name: str) -> list[ModelCard]:
        return self._cards_by_model.get(model_name, [])
```

- [ ] **Step 6: Commit**

```bash
git add nano_dynamo/registry_client.py tests/fakes.py tests/test_registry_client.py
git commit -m "feat: add RegistryClient and FakeRegistryClient test double

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Task 4: Worker Engine

**Files:**
- Create: `nano_dynamo/worker/__init__.py`
- Create: `nano_dynamo/worker/engine.py`
- Create: `tests/test_worker_engine.py`

**Interfaces:**
- Produces: `Engine` (Protocol: `async def generate(self, prompt: str) -> AsyncIterator[str]`), `MockEngine(num_tokens=5, token_delay_seconds=0.05, token_text="token")` — consumed by Task 5's Worker service, and by Task 6's Frontend tests (via Worker apps built with different `token_text` values to distinguish workers).

- [ ] **Step 1: Create empty `nano_dynamo/worker/__init__.py`**

```python
# nano_dynamo/worker/__init__.py
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_worker_engine.py
from nano_dynamo.worker.engine import MockEngine


async def test_mock_engine_yields_requested_number_of_tokens():
    engine = MockEngine(num_tokens=3, token_delay_seconds=0)
    tokens = [token async for token in engine.generate("hello")]
    assert tokens == ["token_0", "token_1", "token_2"]


async def test_mock_engine_uses_custom_token_text():
    engine = MockEngine(num_tokens=2, token_delay_seconds=0, token_text="from-a")
    tokens = [token async for token in engine.generate("hello")]
    assert tokens == ["from-a_0", "from-a_1"]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_worker_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nano_dynamo.worker.engine'`

- [ ] **Step 4: Write `nano_dynamo/worker/engine.py`**

```python
# nano_dynamo/worker/engine.py
import asyncio
from collections.abc import AsyncIterator
from typing import Protocol


class Engine(Protocol):
    async def generate(self, prompt: str) -> AsyncIterator[str]: ...


class MockEngine:
    """Fakes token-by-token generation with a real delay between tokens,
    so the streaming behavior is genuine even though the content isn't."""

    def __init__(
        self,
        num_tokens: int = 5,
        token_delay_seconds: float = 0.05,
        token_text: str = "token",
    ):
        self.num_tokens = num_tokens
        self.token_delay_seconds = token_delay_seconds
        self.token_text = token_text

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        for i in range(self.num_tokens):
            await asyncio.sleep(self.token_delay_seconds)
            yield f"{self.token_text}_{i}"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_worker_engine.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add nano_dynamo/worker/__init__.py nano_dynamo/worker/engine.py tests/test_worker_engine.py
git commit -m "feat: add Engine protocol and MockEngine

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Task 5: Worker Service

**Files:**
- Create: `nano_dynamo/worker/main.py`
- Create: `tests/test_worker.py`

**Interfaces:**
- Consumes: `RegistryClientProtocol`, `WorkerNotFoundError` (Task 3); `Engine`, `MockEngine` (Task 4); `RegisterRequest`, `GenerateRequest` (Task 1); `FakeRegistryClient` (Task 3's `tests/fakes.py`).
- Produces: `WorkerSettings(model_name, endpoint_url, registry_client, worker_type="aggregated", heartbeat_interval_seconds=5.0, engine=MockEngine())`, `create_app(settings: WorkerSettings) -> FastAPI` exposing `POST /generate`. Consumed by Task 6's Frontend tests.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_worker.py
import pytest
from starlette.testclient import TestClient

from nano_dynamo.models import RegisterRequest
from nano_dynamo.registry_client import WorkerNotFoundError
from nano_dynamo.worker.engine import MockEngine
from nano_dynamo.worker.main import WorkerSettings, _send_heartbeat_or_reregister, create_app
from tests.fakes import FakeRegistryClient


def _settings(registry_client: FakeRegistryClient) -> WorkerSettings:
    return WorkerSettings(
        model_name="demo",
        endpoint_url="http://worker-a",
        registry_client=registry_client,
        engine=MockEngine(num_tokens=3, token_delay_seconds=0),
    )


def test_registers_with_registry_on_startup():
    registry_client = FakeRegistryClient()
    app = create_app(_settings(registry_client))
    with TestClient(app):
        pass
    assert registry_client.registered == [
        RegisterRequest(model_name="demo", endpoint_url="http://worker-a")
    ]


async def test_send_heartbeat_or_reregister_refreshes_existing_worker():
    registry_client = FakeRegistryClient()
    state = type("State", (), {})()
    state.worker_id = "worker-1"

    await _send_heartbeat_or_reregister(_settings(registry_client), state)

    assert registry_client.heartbeats == ["worker-1"]
    assert state.worker_id == "worker-1"


async def test_send_heartbeat_or_reregister_recovers_from_worker_not_found():
    registry_client = FakeRegistryClient()
    # Consume "worker-1" up front so the id issued by the recovery-path
    # register() call below is distinguishable from the starting id.
    await registry_client.register(
        RegisterRequest(model_name="demo", endpoint_url="http://worker-a")
    )
    registry_client.fail_next_heartbeat_for("worker-1")
    state = type("State", (), {})()
    state.worker_id = "worker-1"

    await _send_heartbeat_or_reregister(_settings(registry_client), state)

    assert registry_client.heartbeats == ["worker-1"]
    assert state.worker_id == "worker-2"  # re-registered after the 404


def test_generate_streams_tokens_from_engine():
    registry_client = FakeRegistryClient()
    app = create_app(_settings(registry_client))
    with TestClient(app) as client:
        with client.stream("POST", "/generate", json={"prompt": "hi"}) as response:
            body = "".join(response.iter_text())
    assert body == "token_0token_1token_2"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nano_dynamo.worker.main'`

- [ ] **Step 3: Write `nano_dynamo/worker/main.py`**

```python
# nano_dynamo/worker/main.py
import asyncio
import contextlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from nano_dynamo.models import GenerateRequest, RegisterRequest
from nano_dynamo.registry_client import RegistryClientProtocol, WorkerNotFoundError
from nano_dynamo.worker.engine import Engine, MockEngine


@dataclass
class WorkerSettings:
    model_name: str
    endpoint_url: str
    registry_client: RegistryClientProtocol
    worker_type: str = "aggregated"
    heartbeat_interval_seconds: float = 5.0
    engine: Engine = field(default_factory=MockEngine)


def create_app(settings: WorkerSettings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.worker_id = await settings.registry_client.register(
            RegisterRequest(
                model_name=settings.model_name,
                endpoint_url=settings.endpoint_url,
                worker_type=settings.worker_type,
            )
        )
        heartbeat_task = asyncio.create_task(_heartbeat_loop(settings, app.state))
        yield
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task

    app = FastAPI(lifespan=lifespan)

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        async def token_stream():
            async for token in settings.engine.generate(request.prompt):
                yield token

        return StreamingResponse(token_stream(), media_type="text/plain")

    return app


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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_worker.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add nano_dynamo/worker/main.py tests/test_worker.py
git commit -m "feat: add Worker service with registration and self-healing heartbeat

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Task 6: Frontend Service

**Files:**
- Create: `nano_dynamo/frontend/__init__.py`
- Create: `nano_dynamo/frontend/main.py`
- Modify: `tests/fakes.py` — add `FlakyEngine`
- Create: `tests/test_frontend.py`

**Interfaces:**
- Consumes: `RegistryClientProtocol` (Task 3); `ChatCompletionRequest`, `GenerateRequest`, `ModelCard` (Task 1); `WorkerSettings`, `create_app` from `nano_dynamo.worker.main` (Task 5, used only in tests to build real in-memory Worker apps); `FakeRegistryClient` (Task 3).
- Produces: `FrontendSettings(registry_client, worker_client_factory)`, `create_app(settings: FrontendSettings) -> FastAPI` exposing `POST /v1/chat/completions`. Terminal component — nothing later in this plan consumes it.

- [ ] **Step 1: Create empty `nano_dynamo/frontend/__init__.py`**

```python
# nano_dynamo/frontend/__init__.py
```

- [ ] **Step 2: Add `FlakyEngine` to `tests/fakes.py`**

Append to the existing `tests/fakes.py` from Task 3:

```python
class FlakyEngine:
    """Yields one token, then raises — simulates a worker crashing mid-stream."""

    async def generate(self, prompt: str):
        yield "partial"
        raise RuntimeError("worker crashed mid-stream")
```

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_frontend.py
import httpx
from starlette.testclient import TestClient

from nano_dynamo.frontend.main import FrontendSettings, create_app
from nano_dynamo.models import ModelCard
from nano_dynamo.worker.engine import MockEngine
from nano_dynamo.worker.main import WorkerSettings
from nano_dynamo.worker.main import create_app as create_worker_app
from tests.fakes import FakeRegistryClient, FlakyEngine


def _card(worker_id: str, endpoint_url: str) -> ModelCard:
    return ModelCard(worker_id=worker_id, model_name="demo", endpoint_url=endpoint_url)


def _worker_app(endpoint_url: str, engine) -> object:
    return create_worker_app(
        WorkerSettings(
            model_name="demo",
            endpoint_url=endpoint_url,
            registry_client=FakeRegistryClient(),
            engine=engine,
        )
    )


def _factory_for(apps_by_url: dict[str, object]):
    def factory(endpoint_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=apps_by_url[endpoint_url]),
            base_url=endpoint_url,
        )

    return factory


def _chat_request(model: str = "demo") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def test_returns_503_when_no_workers_registered():
    registry_client = FakeRegistryClient()
    app = create_app(FrontendSettings(registry_client=registry_client, worker_client_factory=_factory_for({})))

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_chat_request())

    assert response.status_code == 503


def test_streams_tokens_from_single_worker():
    registry_client = FakeRegistryClient()
    registry_client.set_workers("demo", [_card("worker-a", "http://worker-a")])
    worker_app = _worker_app("http://worker-a", MockEngine(num_tokens=3, token_delay_seconds=0))
    app = create_app(
        FrontendSettings(
            registry_client=registry_client,
            worker_client_factory=_factory_for({"http://worker-a": worker_app}),
        )
    )

    with TestClient(app) as client:
        with client.stream("POST", "/v1/chat/completions", json=_chat_request()) as response:
            body = "".join(response.iter_text())

    assert body == "token_0token_1token_2"


def test_round_robins_across_two_workers():
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

    with TestClient(app) as client:
        bodies = []
        for _ in range(2):
            with client.stream("POST", "/v1/chat/completions", json=_chat_request()) as response:
                bodies.append("".join(response.iter_text()))

    assert set(bodies) == {"from-a_0", "from-b_0"}


def test_worker_failure_mid_stream_returns_clean_error_chunk():
    registry_client = FakeRegistryClient()
    registry_client.set_workers("demo", [_card("worker-a", "http://worker-a")])
    worker_app = _worker_app("http://worker-a", FlakyEngine())
    app = create_app(
        FrontendSettings(
            registry_client=registry_client,
            worker_client_factory=_factory_for({"http://worker-a": worker_app}),
        )
    )

    with TestClient(app) as client:
        with client.stream("POST", "/v1/chat/completions", json=_chat_request()) as response:
            body = "".join(response.iter_text())

    assert body.startswith("partial")
    assert "[error: worker unavailable" in body
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `pytest tests/test_frontend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nano_dynamo.frontend.main'`

- [ ] **Step 5: Write `nano_dynamo/frontend/main.py`**

```python
# nano_dynamo/frontend/main.py
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from nano_dynamo.models import ChatCompletionRequest, GenerateRequest
from nano_dynamo.registry_client import RegistryClientProtocol


@dataclass
class FrontendSettings:
    registry_client: RegistryClientProtocol
    worker_client_factory: Callable[[str], httpx.AsyncClient]


def create_app(settings: FrontendSettings) -> FastAPI:
    app = FastAPI()
    counter = count()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        workers = await settings.registry_client.list_workers(request.model)
        if not workers:
            raise HTTPException(
                status_code=503,
                detail=f"No live workers registered for model '{request.model}'",
            )
        worker = workers[next(counter) % len(workers)]
        prompt = "\n".join(f"{message.role}: {message.content}" for message in request.messages)
        worker_client = settings.worker_client_factory(worker.endpoint_url)

        async def token_stream():
            # By the time a worker fails, the 200 response has already
            # started streaming, so a status code is no longer an option --
            # surface the failure as a final in-band chunk instead. Caught
            # broadly because a crash inside the worker's own generator
            # (not just a network error) can also propagate here.
            try:
                async with worker_client.stream(
                    "POST", "/generate", json=GenerateRequest(prompt=prompt).model_dump()
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_text():
                        yield chunk
            except Exception as exc:
                yield f"\n[error: worker unavailable: {exc}]"

        return StreamingResponse(token_stream(), media_type="text/plain")

    return app
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_frontend.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add nano_dynamo/frontend tests/fakes.py tests/test_frontend.py
git commit -m "feat: add Frontend service with round-robin routing and streaming

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Task 7: README and Bring-Your-Own-Engine Appendix

**Files:**
- Create: `README.md`
- Create: `docs/appendix-bring-your-own-engine.md`

**Interfaces:**
- Consumes: nothing programmatically — this is documentation describing how to run Tasks 1–6's finished services together. No later task depends on it.

- [ ] **Step 1: Write `README.md`**

```markdown
# nano-dynamo

A minimal, from-scratch teaching implementation of [Dynamo](https://github.com/ai-dynamo/dynamo)'s
orchestration layer — service discovery, heartbeat-based liveness, and request
routing — in pure Python, modeled after nanoGPT and nano-vllm.

Real Dynamo orchestrates real inference engines (vLLM, SGLang, TRT-LLM) at
datacenter scale using a Rust core, etcd for discovery, and NATS for
transport. nano-dynamo strips all of that away and keeps only the
orchestration ideas: three independent services, a mocked inference engine,
and plain HTTP.

**One deliberate simplification worth naming up front:** real Dynamo's
discovery is backed by etcd, genuinely independent infrastructure that every
Frontend and Worker is just a client of. nano-dynamo's Registry plays the
same role but is a single in-memory Python process instead of etcd — good
enough to teach the pattern, but not durable and not clustered.

## Chapter 1: Frontend, Worker, Registry

Three processes, three terminals:

```bash
pip install -e ".[dev]"

# terminal 1
uvicorn nano_dynamo.registry.main:create_app --factory --port 8000

# terminal 2
python -m nano_dynamo.worker.main

# terminal 3
uvicorn nano_dynamo.frontend.main:create_app --factory --port 8080
```

Then, in a fourth terminal:

```bash
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "demo", "messages": [{"role": "user", "content": "hi"}]}'
```

You should see fake tokens (`token_0`, `token_1`, ...) stream back.

If you stop the Worker process and immediately retry the `curl`, you'll get a
`503` once the Registry's heartbeat TTL expires the dead worker — this is the
bug class Chapter 1 is designed to avoid by construction: the Frontend never
silently reports itself as ready with zero usable workers behind it.

## Running the tests

```bash
pytest -v
```

## What's next

- **Chapter 2** adds KV-cache-aware routing, replacing Chapter 1's
  round-robin worker selection.
- **Chapter 3** adds disaggregated prefill/decode serving.
- See [`docs/appendix-bring-your-own-engine.md`](docs/appendix-bring-your-own-engine.md)
  for an optional, non-core extension: swapping the mocked engine for a real
  one.
```

- [ ] **Step 2: Write `docs/appendix-bring-your-own-engine.md`**

```markdown
# Appendix: Bring Your Own Engine

Not part of the core chapters — an optional extension for readers who want
to see real tokens instead of fake ones.

The Worker's `Engine` protocol (`nano_dynamo/worker/engine.py`) is
deliberately the same shape as real Dynamo's backend contract: real
Dynamo's Python backends (vLLM, SGLang) register a model card and stream
tokens back over a contract that is structurally the same as this one.

```python
class Engine(Protocol):
    async def generate(self, prompt: str) -> AsyncIterator[str]: ...
```

A real engine only has to implement that one method. Sketch of what a
vLLM-backed implementation would look like:

```python
from collections.abc import AsyncIterator

from vllm import LLM, SamplingParams


class VLLMEngine:
    def __init__(self, model_name: str):
        self._llm = LLM(model=model_name)

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        # vLLM's native API is sync/batched; a real implementation would
        # run it in a thread and bridge results back through an
        # asyncio.Queue so this stays an async generator. Omitted here
        # since it's genuinely orthogonal to nano-dynamo's lesson: nothing
        # in the Registry, Worker HTTP layer, or Frontend needs to change
        # to support this.
        raise NotImplementedError
```

Swap it in by constructing `WorkerSettings(..., engine=VLLMEngine("some-model"))`
instead of the default `MockEngine()` — nothing in the Registry, the
`RegistryClient`, or the Frontend needs to change. That boundary staying
untouched is the actual point of the exercise: pluggable backends underneath
a stable orchestration layer is a real Dynamo concept, not a nano-dynamo
invention.
```

- [ ] **Step 3: Manually verify the README's instructions work end to end**

Run the three commands from `README.md` Chapter 1 section in three terminals, then run the `curl` command from a fourth.
Expected: the `curl` output streams `token_0token_1token_2token_3token_4` (default `MockEngine(num_tokens=5)`), then the connection closes cleanly.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/appendix-bring-your-own-engine.md
git commit -m "docs: add README and bring-your-own-engine appendix

Signed-off-by: elizabetht <email2eliza@gmail.com>"
```

---

## Note on Task 7's `python -m nano_dynamo.worker.main`

Tasks 1–6 build `create_app(settings)` factories, not runnable entry points — deliberately, since every test needs to inject fakes instead of real settings. Task 7's README references a real `if __name__ == "__main__":` entry point for the Worker (reading `endpoint_url`/`registry_url` from environment variables or CLI args, constructing a real `RegistryClient` over real `httpx.AsyncClient`, and calling `uvicorn.run(...)`), and equivalent factory-style `--factory` invocations for Registry and Frontend. Writing those thin entry points is part of Task 7's Step 1 deliverable alongside the README — they're the "production wiring" referenced in this plan's Architecture section, and have no independent test of their own (they're a few lines of argument parsing and object construction, exercised by Task 7 Step 3's manual verification instead of a unit test).
