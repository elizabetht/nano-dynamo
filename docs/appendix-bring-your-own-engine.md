# Appendix: Bring Your Own Engine

Optional — for running real inference instead of the mock. This is no longer
just a sketch: `nano_dynamo/worker/vllm_engine.py` implements a real
vLLM-backed engine.

The Worker's `Engine` protocol (`nano_dynamo/worker/engine.py`) is
deliberately the same shape as real Dynamo's backend contract: real Dynamo's
Python backends embed vLLM's `AsyncLLM` in-process and stream tokens back
(`components/src/dynamo/vllm/worker_factory.py`), which is structurally the
same as this.

```python
class Engine(Protocol):
    async def generate(self, prompt: str) -> AsyncIterator[str]: ...
```

A real engine only has to implement that one method. `VLLMEngine` does it by
embedding vLLM's `AsyncLLM`:

```python
class VLLMEngine:
    def __init__(self, model_name: str, max_tokens: int = 256, temperature: float = 0.7):
        from vllm import AsyncEngineArgs, SamplingParams
        from vllm.v1.engine.async_llm import AsyncLLM
        self._llm = AsyncLLM.from_engine_args(AsyncEngineArgs(model=model_name))
        self._sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        request_id = uuid.uuid4().hex
        emitted = 0
        async for output in self._llm.generate(prompt, self._sampling_params, request_id):
            delta = _new_suffix(output.outputs[0].text, emitted)  # cumulative -> delta
            if delta:
                yield delta
                emitted += len(delta)
```

## Running it

You need a GPU and `pip install vllm` in the worker's environment. Then select
the engine with an environment variable — the mock stays the default, so no GPU
is needed to run the rest of the repo:

```bash
WORKER_ENGINE=vllm WORKER_MODEL_NAME=Qwen/Qwen3-0.6B python -m nano_dynamo.worker.main
```

### Fitting alongside other GPU workloads

By default vLLM grabs ~90% of the GPU (`gpu_memory_utilization=0.9`), which for a
small model is overkill and will fail if the card is already busy. Two optional
knobs let a small model coexist with other workloads:

```bash
WORKER_ENGINE=vllm \
WORKER_MODEL_NAME=Qwen/Qwen3-0.6B \
WORKER_GPU_MEMORY_UTILIZATION=0.15 \
WORKER_MAX_MODEL_LEN=2048 \
python -m nano_dynamo.worker.main
```

- `WORKER_GPU_MEMORY_UTILIZATION` — fraction of GPU memory vLLM may use
  (0.0–1.0). Lower it to leave room for other processes.
- `WORKER_MAX_MODEL_LEN` — cap the context length, which shrinks the KV-cache
  reservation. Unset means vLLM's default.

Both are optional; leaving either unset keeps vLLM's own default.

The Registry, Frontend, `RegistryClient`, routing, and streaming are all
completely unaware of which engine is underneath. That boundary staying
untouched is the point: pluggable backends beneath a stable orchestration layer
is a real Dynamo concept, not a nano-dynamo invention.

## Running across machines

Only the vLLM worker needs a GPU. Where everything else runs is up to you — the
three services just talk over HTTP, so placement is a networking question, not a
hardware one:

- **Registry** — a tiny CPU-only process (an in-memory dict behind FastAPI). Run
  it on any host the workers and Frontend can reach. Pick somewhere stable so its
  address doesn't move; it needs no GPU.
- **Frontend** — also CPU-only (it just looks up workers and streams bytes
  through). Any host reachable by clients and able to reach the workers.
- **Worker (`WORKER_ENGINE=vllm`)** — must run on a GPU host. Run one per GPU
  node.

The only rules are reachability: every worker and the Frontend must reach
`REGISTRY_URL`; the Frontend must reach each worker's advertised
`WORKER_ENDPOINT_URL`; clients must reach the Frontend.

Example: Registry and Frontend on a CPU host at `10.0.0.1`, a vLLM worker on a
GPU host at `10.0.0.2`.

```bash
# on the CPU host (10.0.0.1)
python -m nano_dynamo.registry.main    # REGISTRY_PORT=8000
python -m nano_dynamo.frontend.main    # FRONTEND_PORT=8080, REGISTRY_URL defaults to localhost:8000

# on the GPU host (10.0.0.2)
REGISTRY_URL=http://10.0.0.1:8000 \
WORKER_ENGINE=vllm \
WORKER_MODEL_NAME=Qwen/Qwen3-0.6B \
WORKER_HOST=0.0.0.0 \
WORKER_ENDPOINT_URL=http://10.0.0.2:8001 \
python -m nano_dynamo.worker.main
```

Two cross-machine gotchas:

- **`WORKER_ENDPOINT_URL` must be the node's real, dialable address** (here
  `http://10.0.0.2:8001`), not `127.0.0.1` — it's what the worker advertises to
  the Registry, and the Frontend dials exactly that. A default of `127.0.0.1`
  would tell the Frontend to look for the worker on its *own* machine.
- **`WORKER_HOST=0.0.0.0`** makes the worker bind on all interfaces so it's
  reachable from other hosts, rather than only on loopback.

Because nano's Registry keeps its state in memory with no replication (a Chapter
1 simplification), restarting it wipes the worker list — but every worker
re-registers on its next heartbeat, so the system heals on its own regardless of
where the Registry runs.

## Two things to know

- **The engine is built at startup, not eagerly.** `AsyncLLM` grabs the GPU and
  spins up background tasks, so it's passed as a factory
  (`WorkerSettings(engine_factory=...)`) and constructed inside the running
  event loop, once, when the worker starts — see `WorkerSettings.engine_factory`.
  The worker only registers with the Registry *after* the model is loaded, so it
  never advertises itself before it can serve.
- **No chat template yet.** The Frontend flattens chat messages into a plain
  string, so an instruct model won't get its chat template applied. Applying
  `tokenizer.apply_chat_template` is a good small follow-up; it isn't needed to
  see real tokens stream.
- **vLLM's engine API is version-sensitive.** The import path
  (`vllm.v1.engine.async_llm.AsyncLLM`) and constructor match what current
  Dynamo uses; if your installed vLLM differs, adjust `vllm_engine.py`.
