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

The Registry, Frontend, `RegistryClient`, routing, and streaming are all
completely unaware of which engine is underneath. That boundary staying
untouched is the point: pluggable backends beneath a stable orchestration layer
is a real Dynamo concept, not a nano-dynamo invention.

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
