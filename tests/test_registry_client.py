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
