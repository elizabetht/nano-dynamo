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
