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
