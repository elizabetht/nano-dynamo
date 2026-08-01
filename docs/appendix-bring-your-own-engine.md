# Appendix: Bring Your Own Engine

Not part of the core chapters — an optional extension for readers who want
to see real tokens instead of fake ones.

The Worker's `Engine` protocol (`nano_dynamo/worker/engine.py`) is
deliberately the same shape as real Dynamo's backend contract: real
Dynamo's Python backends (vLLM, SGLang) register a model card and stream
tokens back over a contract that is structurally the same as this one.

```python
class Engine(Protocol):
    async def generate(self, prompt: str) -> AsyncIterator[str]: ...
```

A real engine only has to implement that one method. Sketch of what a
vLLM-backed implementation would look like:

```python
from collections.abc import AsyncIterator

from vllm import LLM


class VLLMEngine:
    def __init__(self, model_name: str):
        self._llm = LLM(model=model_name)

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        # vLLM's native API is sync/batched; a real implementation would run
        # it in a thread and bridge results back through an asyncio.Queue so
        # this stays an async generator. Omitted here since it's genuinely
        # orthogonal to nano-dynamo's lesson: nothing in the Registry, the
        # Worker's HTTP layer, or the Frontend needs to change to support it.
        raise NotImplementedError
```

Swap it in by constructing `WorkerSettings(..., engine=VLLMEngine("some-model"))`
instead of the default `MockEngine()` — nothing in the Registry, the
`RegistryClient`, or the Frontend needs to change. That boundary staying
untouched is the actual point of the exercise: pluggable backends underneath
a stable orchestration layer is a real Dynamo concept, not a nano-dynamo
invention.

Concretely, in `nano_dynamo/worker/main.py`'s entry point you'd replace the
default:

```python
settings = WorkerSettings(
    model_name=os.environ.get("WORKER_MODEL_NAME", "demo"),
    endpoint_url=endpoint_url,
    registry_client=RegistryClient(httpx.AsyncClient(base_url=registry_url)),
    engine=VLLMEngine(os.environ.get("WORKER_MODEL_NAME", "demo")),  # <-- the only change
)
```

Everything else — registration, heartbeats, the `/generate` route, the
Frontend's routing and streaming — is entirely unaware of which engine is
underneath.
