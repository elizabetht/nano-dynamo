from nano_dynamo.worker.engine import MockEngine


async def test_mock_engine_yields_requested_number_of_tokens():
    engine = MockEngine(num_tokens=3, token_delay_seconds=0)
    tokens = [token async for token in engine.generate("hello")]
    assert tokens == ["token_0", "token_1", "token_2"]


async def test_mock_engine_uses_custom_token_text():
    engine = MockEngine(num_tokens=2, token_delay_seconds=0, token_text="from-a")
    tokens = [token async for token in engine.generate("hello")]
    assert tokens == ["from-a_0", "from-a_1"]
