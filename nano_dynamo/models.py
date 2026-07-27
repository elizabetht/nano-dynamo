from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

WorkerType = Literal["aggregated", "prefill", "decode"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelCard(BaseModel):
    worker_id: str
    model_name: str
    endpoint_url: str
    worker_type: WorkerType = "aggregated"
    registered_at: datetime = Field(default_factory=_utcnow)
    last_heartbeat: datetime = Field(default_factory=_utcnow)


class RegisterRequest(BaseModel):
    model_name: str
    endpoint_url: str
    worker_type: WorkerType = "aggregated"


class RegisterResponse(BaseModel):
    worker_id: str


class GenerateRequest(BaseModel):
    prompt: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
