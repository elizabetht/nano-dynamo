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


if __name__ == "__main__":
    import os

    import httpx
    import uvicorn

    from nano_dynamo.registry_client import RegistryClient

    registry_url = os.environ.get("REGISTRY_URL", "http://127.0.0.1:8000")
    host = os.environ.get("WORKER_HOST", "127.0.0.1")
    port = int(os.environ.get("WORKER_PORT", "8001"))
    # endpoint_url is what the Frontend will use to reach this worker, so it
    # must be an address others can dial -- not necessarily the bind host.
    endpoint_url = os.environ.get("WORKER_ENDPOINT_URL", f"http://{host}:{port}")

    settings = WorkerSettings(
        model_name=os.environ.get("WORKER_MODEL_NAME", "demo"),
        endpoint_url=endpoint_url,
        registry_client=RegistryClient(httpx.AsyncClient(base_url=registry_url)),
    )
    uvicorn.run(create_app(settings), host=host, port=port)
