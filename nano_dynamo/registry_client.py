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
