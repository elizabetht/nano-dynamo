from collections.abc import Callable
from dataclasses import dataclass
from itertools import count

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from nano_dynamo.models import ChatCompletionRequest, GenerateRequest
from nano_dynamo.registry_client import RegistryClientProtocol


@dataclass
class FrontendSettings:
    registry_client: RegistryClientProtocol
    worker_client_factory: Callable[[str], httpx.AsyncClient]


def create_app(settings: FrontendSettings) -> FastAPI:
    app = FastAPI()
    counter = count()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        workers = await settings.registry_client.list_workers(request.model)
        if not workers:
            raise HTTPException(
                status_code=503,
                detail=f"No live workers registered for model '{request.model}'",
            )
        worker = workers[next(counter) % len(workers)]
        prompt = "\n".join(f"{message.role}: {message.content}" for message in request.messages)
        worker_client = settings.worker_client_factory(worker.endpoint_url)

        async def token_stream():
            # By the time a worker fails, the 200 response has already
            # started streaming, so a status code is no longer an option --
            # surface the failure as a final in-band chunk instead. Caught
            # broadly because a crash inside the worker's own generator
            # (not just a network error) can also propagate here.
            try:
                async with worker_client.stream(
                    "POST", "/generate", json=GenerateRequest(prompt=prompt).model_dump()
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_text():
                        yield chunk
            except Exception as exc:
                yield f"\n[error: worker unavailable: {exc}]"

        return StreamingResponse(token_stream(), media_type="text/plain")

    return app
