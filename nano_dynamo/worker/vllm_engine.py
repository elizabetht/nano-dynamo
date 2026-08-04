"""Optional real-inference engine backed by vLLM.

This is the "Bring Your Own Engine" appendix made real. It implements the same
`Engine` protocol as `MockEngine`, so nothing in the Registry, the Frontend, or
the Worker's HTTP layer changes -- you swap it in via
`WorkerSettings(engine_factory=lambda: VLLMEngine(model_name))`.

`vllm` is imported lazily inside `__init__` so importing this module is harmless
on a machine without vLLM (or without a GPU); it only touches vLLM when you
actually construct the engine.

Note: vLLM's engine API is version-sensitive. This targets the same surface
Dynamo's own backend uses (`vllm.v1.engine.async_llm.AsyncLLM`); if your
installed vLLM differs, adjust the import and constructor accordingly.
"""

import uuid
from collections.abc import AsyncIterator


def _new_suffix(text: str, already_emitted: int) -> str:
    """vLLM yields cumulative text; the Engine protocol wants deltas. Given the
    latest cumulative `text` and how many characters we've already emitted,
    return only the newly-added suffix (empty if nothing new)."""
    return text[already_emitted:] if len(text) > already_emitted else ""


def _engine_args_kwargs(
    model_name: str,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
) -> dict:
    """Build the AsyncEngineArgs kwargs, omitting any knob that wasn't set so
    vLLM keeps its own defaults rather than being handed None."""
    kwargs: dict = {"model": model_name}
    if gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    return kwargs


class VLLMEngine:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
    ):
        from vllm import AsyncEngineArgs, SamplingParams
        from vllm.v1.engine.async_llm import AsyncLLM

        engine_args = AsyncEngineArgs(
            **_engine_args_kwargs(model_name, gpu_memory_utilization, max_model_len)
        )
        self._llm = AsyncLLM.from_engine_args(engine_args)
        self._sampling_params = SamplingParams(
            max_tokens=max_tokens, temperature=temperature
        )

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        request_id = uuid.uuid4().hex
        emitted = 0
        async for output in self._llm.generate(prompt, self._sampling_params, request_id):
            delta = _new_suffix(output.outputs[0].text, emitted)
            if delta:
                yield delta
                emitted += len(delta)
