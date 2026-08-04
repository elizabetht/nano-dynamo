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
        self._heartbeat_error: Exception | None = None

    async def register(self, request: RegisterRequest) -> str:
        self._next_id += 1
        worker_id = f"worker-{self._next_id}"
        self.registered.append(request)
        return worker_id

    async def heartbeat(self, worker_id: str) -> None:
        self.heartbeats.append(worker_id)
        if self._heartbeat_error is not None:
            raise self._heartbeat_error
        if worker_id in self._fail_heartbeat_for:
            raise WorkerNotFoundError(worker_id)

    def fail_next_heartbeat_for(self, worker_id: str) -> None:
        self._fail_heartbeat_for.add(worker_id)

    def raise_on_heartbeat(self, error: Exception) -> None:
        """Make every heartbeat raise `error` (simulates the Registry being
        unreachable) until cleared with clear_heartbeat_error()."""
        self._heartbeat_error = error

    def clear_heartbeat_error(self) -> None:
        self._heartbeat_error = None

    def set_workers(self, model_name: str, cards: list[ModelCard]) -> None:
        self._cards_by_model[model_name] = cards

    async def list_workers(self, model_name: str) -> list[ModelCard]:
        return self._cards_by_model.get(model_name, [])


class FlakyWorkerClient:
    """A fake worker HTTP client that truly streams one chunk, then raises
    mid-stream. httpx.ASGITransport can't reproduce this — it buffers the whole
    ASGI response before returning it, so partial-output-then-failure is
    unrepresentable through it. This fakes at the HTTP-client layer instead,
    which is where a mid-stream worker failure actually surfaces to the
    Frontend.

    It doubles as its own async context manager and response object, matching
    how the Frontend uses `worker_client.stream(...)`:

        async with worker_client.stream(...) as response:
            response.raise_for_status()
            async for chunk in response.aiter_text():
                ...
    """

    def stream(self, method: str, url: str, **kwargs) -> "FlakyWorkerClient":
        return self

    async def __aenter__(self) -> "FlakyWorkerClient":
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    async def aiter_text(self):
        yield "partial"
        raise RuntimeError("worker crashed mid-stream")
