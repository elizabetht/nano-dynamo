from nano_dynamo.frontend.router import KVRouter
from nano_dynamo.models import ModelCard


def _card(worker_id: str) -> ModelCard:
    return ModelCard(worker_id=worker_id, model_name="demo", endpoint_url=f"http://{worker_id}")


def test_block_hashes_chain_grows_one_per_block():
    router = KVRouter(block_size=2)
    # 5 words, block_size 2 -> blocks: [w0 w1][w2 w3][w4] -> 3 hashes
    hashes = router.block_hashes("a b c d e")
    for i, h in enumerate(hashes):
        print(f"hashes_{i} = {h[:16]}") 
    assert len(hashes) == 3
    assert all(isinstance(h, str) for h in hashes)


def test_block_hashes_empty_prompt_is_empty():
    router = KVRouter(block_size=2)
    assert router.block_hashes("") == []
    assert router.block_hashes("   ") == []


def test_block_hashes_are_a_shared_prefix_chain():
    router = KVRouter(block_size=2)
    a = router.block_hashes("the quick brown fox")     # [the quick][brown fox]
    b = router.block_hashes("the quick brown dog")  
    for i, h in enumerate(a):
        print(f"a_{i} = {h[:16]}") 
    for i, h in enumerate(b):
        print(f"b_{i} = {h[:16]}")  # [the quick][brown dog]
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


def test_acquire_and_release_track_inflight():
    router = KVRouter()
    assert router.inflight.get("w1", 0) == 0
    router.acquire("w1")
    router.acquire("w1")
    assert router.inflight["w1"] == 2
    router.release("w1")
    assert router.inflight["w1"] == 1


def test_release_never_goes_negative():
    router = KVRouter()
    router.release("w1")  # release with nothing acquired
    assert router.inflight.get("w1", 0) == 0


def test_select_cold_prompt_picks_least_loaded():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    router.acquire("w1")  # w1 busier
    chosen = router.select(workers, "brand new prompt")
    assert chosen.worker_id == "w2"


def test_select_ties_broken_by_least_load():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    # Both workers hold the prefix; w1 is busier, so w2 wins.
    for h in router.block_hashes("shared prefix here"):
        router.prefix_owners.setdefault(h, set()).update({"w1", "w2"})
    router.acquire("w1")
    chosen = router.select(workers, "shared prefix here")
    assert chosen.worker_id == "w2"


def test_select_longer_prefix_overlap_wins():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    hashes = router.block_hashes("a b c d e f")  # 3 blocks
    # w1 shares only block 0; w2 shares blocks 0 and 1 -> w2 has the longer prefix.
    router.prefix_owners.setdefault(hashes[0], set()).update({"w1", "w2"})
    router.prefix_owners.setdefault(hashes[1], set()).add("w2")
    chosen = router.select(workers, "a b c d e f")
    assert chosen.worker_id == "w2"


def test_select_ignores_reaped_workers_in_prefix_owners():
    router = KVRouter(block_size=2)
    workers = [_card("w2")]  # only w2 is live now
    # prefix_owners still names w1 (since reaped), but it isn't in `workers`.
    for h in router.block_hashes("shared prefix here"):
        router.prefix_owners.setdefault(h, set()).add("w1")
    chosen = router.select(workers, "shared prefix here")
    assert chosen.worker_id == "w2"  # falls back to the live worker


def test_select_round_robins_among_equally_loaded_cold_candidates():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    # Distinct cold prompts, both workers idle -> spread round-robin, not piled
    # onto the first-listed worker. (select does not acquire, so load stays 0.)
    a = router.select(workers, "alpha beta")
    b = router.select(workers, "gamma delta")
    assert {a.worker_id, b.worker_id} == {"w1", "w2"}


def test_record_then_same_prompt_routes_to_same_worker():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    first = router.select(workers, "keep this together")
    router.record(first.worker_id, "keep this together")
    # A second identical prompt should now prefer the worker that served it.
    second = router.select(workers, "keep this together")
    assert second.worker_id == first.worker_id


def test_record_makes_prefix_sharing_prompt_route_together():
    router = KVRouter(block_size=2)
    workers = [_card("w1"), _card("w2")]
    router.record("w1", "the quick brown fox")
    # Shares the first two blocks ("the quick", "brown ...") -> should hit w1,
    # even though w2 is equally idle.
    chosen = router.select(workers, "the quick brown dog")
    assert chosen.worker_id == "w1"
