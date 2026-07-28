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
