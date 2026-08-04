from starlette.testclient import TestClient

from nano_dynamo.models import RegisterRequest
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


def test_engine_factory_is_built_lazily_at_startup_and_used():
    registry_client = FakeRegistryClient()
    built = []

    def factory():
        engine = MockEngine(num_tokens=2, token_delay_seconds=0, token_text="lazy")
        built.append(engine)
        return engine

    settings = WorkerSettings(
        model_name="demo",
        endpoint_url="http://worker-a",
        registry_client=registry_client,
        engine_factory=factory,
    )
    app = create_app(settings)
    assert built == []  # constructing the app must not build the engine

    with TestClient(app) as client:
        assert len(built) == 1  # built once, on startup
        with client.stream("POST", "/generate", json={"prompt": "hi"}) as response:
            body = "".join(response.iter_text())
    assert body == "lazy_0lazy_1"
