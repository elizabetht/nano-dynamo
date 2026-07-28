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
