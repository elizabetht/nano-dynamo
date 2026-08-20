import httpx
from starlette.testclient import TestClient

from nano_dynamo.frontend.main import FrontendSettings, create_app
from nano_dynamo.models import ModelCard
from nano_dynamo.worker.engine import MockEngine
from nano_dynamo.worker.main import WorkerSettings
from nano_dynamo.worker.main import create_app as create_worker_app
from tests.fakes import FakeRegistryClient, FlakyWorkerClient


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

    # Two prompts with no shared prefix, so both stay cache-cold and routing
    # falls through to the load / round-robin tiers. (Chapter 1 sent the *same*
    # prompt twice here; Chapter 2 deliberately routes that back to one worker.)
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


def test_worker_failure_mid_stream_returns_clean_error_chunk():
    registry_client = FakeRegistryClient()
    registry_client.set_workers("demo", [_card("worker-a", "http://worker-a")])
    # A worker that truly streams "partial" then fails mid-stream. We inject it
    # at the client-factory seam rather than going through ASGITransport, which
    # buffers the whole response and so can't reproduce partial-then-failure.
    app = create_app(
        FrontendSettings(
            registry_client=registry_client,
            worker_client_factory=lambda endpoint_url: FlakyWorkerClient(),
        )
    )

    with TestClient(app) as client:
        with client.stream("POST", "/v1/chat/completions", json=_chat_request()) as response:
            body = "".join(response.iter_text())

    assert body.startswith("partial")
    assert "[error: worker unavailable" in body


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

    # The router hashes "user: <content>" in 16-word blocks, so the two prompts
    # must share their first 16 words -- "user:" plus 15 of content -- to share
    # a block hash. Divergence inside a block buys nothing.
    shared = "the quick brown fox jumps over the lazy dog while the cat naps quietly today"
    first = send(f"{shared} alpha")
    second = send(f"{shared} beta")  # shares the leading block
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
