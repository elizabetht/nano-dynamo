from nano_dynamo.frontend.router import KVRouter


def test_block_hashes_chain_grows_one_per_block():
    router = KVRouter(block_size=2)
    # 5 words, block_size 2 -> blocks: [w0 w1][w2 w3][w4] -> 3 hashes
    hashes = router.block_hashes("a b c d e")
    assert len(hashes) == 3
    assert all(isinstance(h, str) for h in hashes)


def test_block_hashes_empty_prompt_is_empty():
    router = KVRouter(block_size=2)
    assert router.block_hashes("") == []
    assert router.block_hashes("   ") == []


def test_block_hashes_are_a_shared_prefix_chain():
    router = KVRouter(block_size=2)
    a = router.block_hashes("the quick brown fox")     # [the quick][brown fox]
    b = router.block_hashes("the quick brown dog")     # [the quick][brown dog]
    # First block matches (same words) -> same h_0; second differs -> different h_1.
    assert a[0] == b[0]
    assert a[1] != b[1]


def test_block_hashes_diverge_when_first_block_differs():
    router = KVRouter(block_size=2)
    a = router.block_hashes("the quick brown fox")
    b = router.block_hashes("a quick brown fox")
    # First block differs, and the chain folds the prefix in, so every later
    # hash differs too even though later words match.
    assert a[0] != b[0]
    assert a[1] != b[1]


def test_block_hashes_are_deterministic():
    # Stable hash (hashlib), not Python's randomized hash(): same input, same output.
    assert KVRouter().block_hashes("hello world foo") == KVRouter().block_hashes("hello world foo")
