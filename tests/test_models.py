import pytest
from pydantic import ValidationError

from nano_dynamo.models import (
    ChatCompletionRequest,
    ChatMessage,
    GenerateRequest,
    ModelCard,
    RegisterRequest,
    RegisterResponse,
)


def test_model_card_defaults_to_aggregated_worker_type():
    card = ModelCard(worker_id="w1", model_name="demo", endpoint_url="http://worker-a")
    assert card.worker_type == "aggregated"
    assert card.registered_at is not None
    assert card.last_heartbeat is not None


def test_model_card_rejects_unknown_worker_type():
    with pytest.raises(ValidationError):
        ModelCard(
            worker_id="w1",
            model_name="demo",
            endpoint_url="http://worker-a",
            worker_type="not-a-real-type",
        )


def test_register_request_defaults_to_aggregated():
    request = RegisterRequest(model_name="demo", endpoint_url="http://worker-a")
    assert request.worker_type == "aggregated"


def test_register_response_round_trips_worker_id():
    response = RegisterResponse(worker_id="w1")
    assert RegisterResponse.model_validate(response.model_dump()).worker_id == "w1"


def test_chat_completion_request_requires_model_and_messages():
    request = ChatCompletionRequest(
        model="demo", messages=[ChatMessage(role="user", content="hi")]
    )
    assert request.messages[0].content == "hi"
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])


def test_generate_request_holds_prompt():
    assert GenerateRequest(prompt="hello").prompt == "hello"
