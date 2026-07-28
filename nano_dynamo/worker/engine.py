import asyncio
from collections.abc import AsyncIterator
from typing import Protocol


class Engine(Protocol):
    async def generate(self, prompt: str) -> AsyncIterator[str]: ...


class MockEngine:
    """Fakes token-by-token generation with a real delay between tokens,
    so the streaming behavior is genuine even though the content isn't."""

    def __init__(
        self,
        num_tokens: int = 5,
        token_delay_seconds: float = 0.05,
        token_text: str = "token",
    ):
        self.num_tokens = num_tokens
        self.token_delay_seconds = token_delay_seconds
        self.token_text = token_text

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        for i in range(self.num_tokens):
            await asyncio.sleep(self.token_delay_seconds)
            yield f"{self.token_text}_{i}"
