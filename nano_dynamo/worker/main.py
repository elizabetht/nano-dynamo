import asyncio
import contextlib
from collections.abc import Callable
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
    # A ready-made engine (the mock default). Used when engine_factory is None.
    engine: Engine = field(default_factory=MockEngine)
    # An optional factory built once at startup instead of `engine`. Needed for
    # engines like vLLM that must be constructed inside the running event loop,
    # on the GPU host, and are too heavy to build eagerly at import time.
    engine_factory: Callable[[], Engine] | None = None


def create_app(settings: WorkerSettings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # If a factory is set (e.g. vLLM), build it at startup and override the
        # synchronous default below -- so a slow-loading engine only registers
        # once it can actually serve. The synchronous default in create_app
        # keeps the mock path working under httpx.ASGITransport, which does not
        # run lifespan events (same reason the Registry inits state eagerly).
        if settings.engine_factory:
            app.state.engine = settings.engine_factory()
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
    # Synchronous default so the engine is available even without lifespan
    # (httpx.ASGITransport in tests); lifespan may override it with the factory.
    app.state.engine = settings.engine

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        async def token_stream():
            async for token in app.state.engine.generate(request.prompt):
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
    model_name = os.environ.get("WORKER_MODEL_NAME", "demo")

    # WORKER_ENGINE=mock (default, no GPU) or WORKER_ENGINE=vllm (real inference).
    # vLLM is passed as a factory so it's built at startup on the GPU host, not
    # eagerly here -- see WorkerSettings.engine_factory.
    engine_factory = None
    if os.environ.get("WORKER_ENGINE", "mock") == "vllm":
        from nano_dynamo.worker.vllm_engine import VLLMEngine

        engine_factory = lambda: VLLMEngine(model_name)  # noqa: E731

    settings = WorkerSettings(
        model_name=model_name,
        endpoint_url=endpoint_url,
        registry_client=RegistryClient(httpx.AsyncClient(base_url=registry_url)),
        engine_factory=engine_factory,
    )
    uvicorn.run(create_app(settings), host=host, port=port)
