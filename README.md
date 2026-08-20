# nano-dynamo

A minimal, from-scratch teaching implementation of
[Dynamo](https://github.com/ai-dynamo/dynamo)'s orchestration layer — service
discovery, heartbeat-based liveness, and request routing — in pure Python,
modeled after [nanoGPT](https://github.com/karpathy/nanoGPT) and
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

Real Dynamo orchestrates real inference engines (vLLM, SGLang, TRT-LLM) at
datacenter scale using a Rust core, etcd for discovery, and NATS for
transport. nano-dynamo strips all of that away and keeps only the
orchestration ideas: three independent services, a mocked or real inference engine,
and plain HTTP.

**One deliberate simplification worth naming up front:** real Dynamo's
discovery is backed by etcd, genuinely independent infrastructure that every
Frontend and Worker is just a client of. nano-dynamo's Registry plays the
same role but is a single in-memory Python process instead of etcd — good
enough to teach the pattern, but not durable and not clustered.

## The three services

| Service   | Role                                                              | Default port |
|-----------|------------------------------------------------------------------|--------------|
| Registry  | Source of truth for "who's alive"; stands in for etcd            | 8000         |
| Worker    | Generates tokens (mock by default; real via vLLM); registers + heartbeats to the Registry | 8001         |
| Frontend  | Client-facing API; finds live workers and routes/streams to them, KV-cache-aware | 8080         |

They only ever talk to each other over HTTP. See each service's own README
(`nano_dynamo/registry/`, `nano_dynamo/worker/`) for the details of how it
works.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Chapter 1: Frontend, Worker, Registry

Three processes, three terminals (each needs `source .venv/bin/activate`
first):

```bash
# terminal 1 — Registry
python -m nano_dynamo.registry.main

# terminal 2 — Worker
python -m nano_dynamo.worker.main

# terminal 3 — Frontend
python -m nano_dynamo.frontend.main
```

Then, in a fourth terminal:

```bash
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "demo", "messages": [{"role": "user", "content": "hi"}]}'
```

You should see fake tokens (`token_0`, `token_1`, ...) stream back one at a
time.

If you stop the Worker process (Ctrl-C in terminal 2) and immediately retry
the `curl`, you'll get a `503` once the Registry's heartbeat TTL expires the
dead worker — this is the bug class Chapter 1 is designed to avoid by
construction: the Frontend never silently reports itself as ready with zero
usable workers behind it.

## Run with real vLLM instead of the mock (optional)

Everything above uses the mock engine, so it runs anywhere with no GPU. To
serve real tokens, run the Worker with `WORKER_ENGINE=vllm` on a GPU host
(`pip install vllm` first). Keep the Registry and Frontend exactly as above —
only the Worker command changes:

```bash
# terminal 2 — Worker, now backed by vLLM
WORKER_ENGINE=vllm \
WORKER_MODEL_NAME=Qwen/Qwen3-0.6B \
WORKER_GPU_MEMORY_UTILIZATION=0.2 \
WORKER_MAX_MODEL_LEN=2048 \
python -m nano_dynamo.worker.main
```

vLLM will download and load the model (a real pause with lots of vLLM logs);
the Worker only registers once loading finishes. Then test it — note the
`"model"` field **must match** `WORKER_MODEL_NAME`, since the Frontend routes by
model name (a mismatch is the most common cause of a `503`):

```bash
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Say hello in one sentence."}]}'
```

`WORKER_GPU_MEMORY_UTILIZATION` (fraction of GPU memory vLLM may claim) lets a
small model share a busy GPU instead of grabbing vLLM's default ~90%; both it
and `WORKER_MAX_MODEL_LEN` are optional. See
[`docs/appendix-bring-your-own-engine.md`](docs/appendix-bring-your-own-engine.md)
for cross-machine wiring and the `Engine` interface.

## Chapter 2: KV-cache-aware routing

Chapter 1's Frontend picked workers round-robin. Chapter 2 replaces that with a
`KVRouter` that sends each prompt to the worker most likely to already hold its
prefix in KV cache — a cached prefix doesn't need recomputing.

It hashes each prompt into a chain of 16-word block hashes (so a shared hash
proves a shared prefix), remembers which worker it sent each prompt to, and
picks by longest cached-prefix overlap, breaking ties by least in-flight load
and then round-robin. Nothing else changes: same endpoint, same streaming, same
`503`, same Registry. With no cache history and idle workers it behaves exactly
like Chapter 1.

Nothing new to run — the walkthrough above already uses it. With two workers up,
send two prompts that share a long opening and they'll land on the same worker;
send two unrelated prompts and they'll spread.

The router predicts cache state from its own routing history rather than being
told by the workers, which is a real simplification with real limits — see
[`nano_dynamo/frontend/README.md`](nano_dynamo/frontend/README.md) for the
algorithm and an honest list of what it gets wrong.

### Verifying it with two real vLLM workers

Routing only becomes observable with two workers serving the same model. The
run below used three machines — a CPU host for the Registry and Frontend, and
two GPU hosts each running a vLLM worker — but two workers on one box with
different `WORKER_PORT`s works the same way.

```bash
# CPU host (192.168.1.75) — Registry, then Frontend
REGISTRY_HOST=0.0.0.0 python -m nano_dynamo.registry.main
FRONTEND_HOST=0.0.0.0 REGISTRY_URL=http://127.0.0.1:8000 python -m nano_dynamo.frontend.main

# each GPU host — same WORKER_MODEL_NAME, its own WORKER_ENDPOINT_URL
WORKER_ENGINE=vllm \
WORKER_MODEL_NAME=Qwen/Qwen3-0.6B \
WORKER_HOST=0.0.0.0 \
WORKER_ENDPOINT_URL=http://192.168.1.76:8001 \
REGISTRY_URL=http://192.168.1.75:8000 \
WORKER_GPU_MEMORY_UTILIZATION=0.2 \
WORKER_MAX_MODEL_LEN=2048 \
python -m nano_dynamo.worker.main
```

Wait until both appear, then send the same prompt twice:

```bash
curl -s "http://192.168.1.75:8000/workers?model_name=Qwen/Qwen3-0.6B"

PROMPT="You are a helpful assistant. Answer using only the following context. The capital of France is Paris and its population is about two million people. What is the capital?"
for i in 1 2; do
  curl -s -N -X POST http://192.168.1.75:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"Qwen/Qwen3-0.6B\", \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}]}"
done
```

Each worker logs a line per request, so count them on each GPU host to see where
the two requests went:

```bash
grep -c "POST /generate" <worker log>
```

Both requests land on **one** worker — the second is a cache hit, so the router
sends it back to the worker that already holds the prefix:

| Frontend version | worker A | worker B |
|------------------|----------|----------|
| `0.2.0` (KV-aware)   | **+2**   | +0       |
| `0.1.1` (round-robin) | +1       | +1       |

Downgrading only the Frontend (`pip install nano-dynamo==0.1.1`, restart it, and
leave the workers untouched) reproduces the bottom row — the same prompt gets
split across both workers, recomputing the prefix on the second one.

Prompts don't have to be identical, only prefix-sharing. Change the last few
words (`What is the population?`) and the request still routes to the same
worker, because the blocks before the change hash the same.

Four things that will otherwise look like bugs:

- **Short prompts prove nothing.** The Frontend hashes `"user: <content>"` in
  16-word blocks, so about 15 shared content words are needed before a single
  block matches. `"hi"` twice shares no block hash at all.
- **A cold prompt's worker is not predictable.** The round-robin cursor advances
  on every selection, including cache hits, so an unrelated prompt won't
  reliably land on "the other" worker. Two unrelated prompts back to back do
  alternate.
- **Restarting the Frontend forgets everything.** `prefix_owners` is in-process
  memory, so the first prompt after a restart is always cold.
- **You won't see a speedup at this size.** A 0.6B model with a 30-word prompt
  prefills in milliseconds, so the saved work is lost in the noise. Routing is
  what's observable here, not latency. vLLM confirms prefix caching is active in
  its startup config line (`enable_prefix_caching=True`), but driving `AsyncLLM`
  in-process doesn't start vLLM's periodic stats logger, so there's no hit-rate
  line in the worker log to read.

## Configuration

Every service reads its config from environment variables, with the defaults
above. The ones you're most likely to touch:

| Variable         | Default                 | Used by            |
|------------------|-------------------------|--------------------|
| `REGISTRY_URL`   | `http://127.0.0.1:8000` | Worker, Frontend   |
| `REGISTRY_PORT`  | `8000`                  | Registry           |
| `WORKER_PORT`    | `8001`                  | Worker             |
| `WORKER_MODEL_NAME` | `demo`               | Worker             |
| `WORKER_ENGINE`  | `mock`                  | Worker (`mock` or `vllm`) |
| `WORKER_GPU_MEMORY_UTILIZATION` | vLLM default (~0.9) | Worker (vLLM) |
| `WORKER_MAX_MODEL_LEN` | vLLM default        | Worker (vLLM)      |
| `FRONTEND_PORT`  | `8080`                  | Frontend           |

The vLLM knobs are covered in [Run with real vLLM](#run-with-real-vllm-instead-of-the-mock-optional)
above.

To run two workers for the same model (and watch the Frontend spread requests
across them), start a second Worker on a different port:

```bash
WORKER_PORT=8002 python -m nano_dynamo.worker.main
```

## Running the tests

```bash
pytest -v
```

## Deploying

Every host that runs a service needs the package installed there — it's one
codebase, and the role is chosen at launch by the command and env vars. Pick
whichever fits how many machines you're spreading across.

### Single box (simplest)

Run all three services on one machine, each in its own terminal, exactly as in
the Chapter 1 walkthrough above. A single GPU box can even run the Registry,
Frontend, and a `WORKER_ENGINE=vllm` worker together.

### Install from PyPI (recommended)

On each host, install the package and launch whichever service that host runs —
no git checkout needed per node:

```bash
pip install nano-dynamo==0.2.0
python -m nano_dynamo.registry.main   # or .frontend.main / .worker.main
```

**Pin the version.** The `0.2.x` line is Chapter 2 — KV-cache-aware routing.
The `0.1.x` line is Chapter 1 — round-robin worker scheduling; pin `==0.1.1` if
you want that older selection behavior. Both run the same services with the same
endpoints, and both support real vLLM inference; only how the Frontend picks a
worker changed. Installing unpinned (`pip install nano-dynamo`) always pulls the
latest.

### Build a wheel locally (alternative)

If you'd rather not go through PyPI, build the artifact and copy it to each host:

```bash
python -m build                       # produces dist/nano_dynamo-<version>-py3-none-any.whl
pip install dist/nano_dynamo-*.whl
python -m nano_dynamo.registry.main   # or .frontend.main / .worker.main
```

### Real vLLM on a GPU

The installs above bring the CPU-only mock path; `pip install vllm` on the GPU
host adds real inference. For the launch command and knobs see
[Run with real vLLM](#run-with-real-vllm-instead-of-the-mock-optional) above,
and [`docs/appendix-bring-your-own-engine.md`](docs/appendix-bring-your-own-engine.md)
for cross-machine wiring.

## What's next

- **Chapter 3** adds disaggregated prefill/decode serving.
