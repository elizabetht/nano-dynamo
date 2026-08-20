import hashlib


class KVRouter:
    """Cache-aware worker selection. Pure synchronous logic -- no async, no HTTP.

    Predicts what each worker has cached from the router's own routing history
    (Approach A): the block hashes of every prompt it has sent to a worker are
    recorded against that worker, and a new prompt is routed to the worker with
    the longest matching prefix.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        # In-flight requests per worker; an absent worker means 0. Used by
        # `select` to break ties *among workers with equal cache overlap* --
        # it does not override affinity. A hot shared prefix owned by one
        # worker keeps going to that worker however deep its queue gets;
        # real Dynamo instead scores overlap against load on one scale
        # (`--router-kv-overlap-score-credit`) so load can win.
        self.inflight: dict[str, int] = {}
        self.prefix_owners: dict[str, set[str]] = {}
        self._rr = 0  # round-robin cursor for breaking equal-load ties

    def block_hashes(self, prompt: str) -> list[str]:
        """Split the prompt into fixed-size word blocks and chain-hash them, so
        that hash h_k depends on words 0..k. A shared h_k between two prompts
        therefore proves they share a prefix through block k."""
        words = prompt.split()
        hashes: list[str] = []
        prev = ""
        for start in range(0, len(words), self.block_size):
            block = " ".join(words[start : start + self.block_size])
            prev = hashlib.blake2b((prev + "\x00" + block).encode()).hexdigest()
            hashes.append(prev)
        return hashes

    def select(self, workers, prompt: str):
        """Pick the live worker with the longest cached-prefix overlap, breaking
        ties by least in-flight load and then round-robin. `workers` is assumed
        non-empty; the Frontend 503s before calling this."""
        live_ids = {w.worker_id for w in workers}
        hashes = self.block_hashes(prompt)

        # Longest prefix first: the last hash any live worker holds gives the
        # best cache-hit candidates.
        candidates: set[str] = set()
        for h in reversed(hashes):
            owners = self.prefix_owners.get(h, set()) & live_ids
            if owners:
                candidates = owners
                break

        # Cold prompt (or only reaped owners): every live worker is a candidate.
        if not candidates:
            candidates = live_ids

        # Least in-flight load, preserving input order for stable round-robin.
        ranked = [w for w in workers if w.worker_id in candidates]
        min_load = min(self.inflight.get(w.worker_id, 0) for w in ranked)
        least_loaded = [w for w in ranked if self.inflight.get(w.worker_id, 0) == min_load]

        chosen = least_loaded[self._rr % len(least_loaded)]
        self._rr += 1
        return chosen

    def record(self, worker_id: str, prompt: str) -> None:
        """Remember that `worker_id` has now processed `prompt`, so future
        prefix-sharing prompts route back to it (Approach A prediction)."""
        for h in self.block_hashes(prompt):
            self.prefix_owners.setdefault(h, set()).add(worker_id)

    def acquire(self, worker_id: str) -> None:
        self.inflight[worker_id] = self.inflight.get(worker_id, 0) + 1

    def release(self, worker_id: str) -> None:
        self.inflight[worker_id] = max(0, self.inflight.get(worker_id, 0) - 1)
