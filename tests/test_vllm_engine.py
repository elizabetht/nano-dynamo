"""Tests for the pure delta-extraction logic in VLLMEngine.

The vLLM-backed streaming itself needs a GPU and a vLLM install, so it isn't
exercised here. What we *can* test in isolation is the one bit of real logic:
turning vLLM's cumulative text into the per-step deltas the Engine protocol
expects. Importing the module must not require vLLM (the import is lazy).
"""

from nano_dynamo.worker.vllm_engine import _new_suffix


def test_new_suffix_returns_only_the_added_text():
    assert _new_suffix("hello", 0) == "hello"
    assert _new_suffix("hello world", len("hello")) == " world"


def test_new_suffix_empty_when_nothing_new():
    assert _new_suffix("hello", len("hello")) == ""


def test_deltas_reconstruct_the_full_cumulative_text():
    cumulative = ["The", "The quick", "The quick brown"]
    emitted = 0
    pieces = []
    for text in cumulative:
        delta = _new_suffix(text, emitted)
        if delta:
            pieces.append(delta)
            emitted += len(delta)
    assert "".join(pieces) == "The quick brown"
