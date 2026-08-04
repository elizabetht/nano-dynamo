# nano-dynamo

A minimal, from-scratch teaching implementation of
[Dynamo](https://github.com/ai-dynamo/dynamo)'s orchestration layer — service
discovery, heartbeat-based liveness, and request routing — in pure Python,
modeled after [nanoGPT](https://github.com/karpathy/nanoGPT) and
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

Real Dynamo orchestrates real inference engines (vLLM, SGLang, TRT-LLM) at
datacenter scale using a Rust core, etcd for discovery, and NATS for
transport. nano-dynamo strips all of that away and keeps only the
orchestration ideas: three independent services, a mocked inference engine,
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
| Worker    | Generates tokens (mocked); registers + heartbeats to the Registry | 8001         |
| Frontend  | Client-facing API; finds live workers and routes/streams to them  | 8080         |

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
| `FRONTEND_PORT`  | `8080`                  | Frontend           |

For real inference instead of the mock, set `WORKER_ENGINE=vllm` (needs a GPU
and `pip install vllm`); see
[`docs/appendix-bring-your-own-engine.md`](docs/appendix-bring-your-own-engine.md).

To run two workers for the same model (and watch the Frontend round-robin
between them), start a second Worker on a different port:

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

### As a wheel (a few machines)

Build once, then `pip install` the artifact on each host — no git checkout
needed per node:

```bash
python -m build                       # produces dist/nano_dynamo-0.1.0-py3-none-any.whl
# copy the wheel to each host, then on each:
pip install nano_dynamo-0.1.0-py3-none-any.whl
python -m nano_dynamo.registry.main   # or .frontend.main / .worker.main
```

### As a container (one image, many roles)

The `Dockerfile` builds a single CPU-only image; you pick the service at
`docker run` time — the same pattern real Dynamo uses (one image per node role).

```bash
docker build -t nano-dynamo .

# each on its host (or all on one, on a shared docker network):
docker run -p 8000:8000 nano-dynamo python -m nano_dynamo.registry.main
docker run -p 8080:8080 -e REGISTRY_URL=http://<registry-host>:8000 \
    nano-dynamo python -m nano_dynamo.frontend.main
docker run -p 8001:8001 -e REGISTRY_URL=http://<registry-host>:8000 \
    -e WORKER_ENDPOINT_URL=http://<worker-host>:8001 \
    nano-dynamo python -m nano_dynamo.worker.main
```

The image sets `*_HOST=0.0.0.0` so services are reachable once their ports are
published. Set `REGISTRY_URL` and `WORKER_ENDPOINT_URL` to addresses the other
containers/hosts can actually dial (see the cross-machine notes in the
appendix).

### Real vLLM on a GPU

The image above is CPU-only and runs the mock engine. For real inference you
need vLLM, which requires a CUDA-capable base image — build a separate image
`FROM` vLLM's official image (or a CUDA base with `pip install vllm` plus this
wheel) and run the worker with `WORKER_ENGINE=vllm`. See
[`docs/appendix-bring-your-own-engine.md`](docs/appendix-bring-your-own-engine.md).

## What's next

- **Chapter 2** adds KV-cache-aware routing, replacing Chapter 1's
  round-robin worker selection.
- **Chapter 3** adds disaggregated prefill/decode serving.
- See [`docs/appendix-bring-your-own-engine.md`](docs/appendix-bring-your-own-engine.md)
  for an optional, non-core extension: swapping the mocked engine for a real
  one.
