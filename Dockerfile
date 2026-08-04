# syntax=docker/dockerfile:1
#
# One image, three roles. Build once; pick the service at `docker run` time:
#
#   docker run -p 8000:8000 nano-dynamo python -m nano_dynamo.registry.main
#   docker run -p 8080:8080 nano-dynamo python -m nano_dynamo.frontend.main
#   docker run -p 8001:8001 nano-dynamo python -m nano_dynamo.worker.main
#
# This image is CPU-only and uses the mock engine. For real vLLM inference on a
# GPU, see docs/appendix-bring-your-own-engine.md -- that needs a CUDA/vLLM base
# image, not this slim one.
FROM python:3.12-slim

WORKDIR /app

# Copy only what the build needs first, so the pip layer caches across code edits.
COPY pyproject.toml ./
COPY nano_dynamo ./nano_dynamo
RUN pip install --no-cache-dir .

# Inside a container the services must bind on all interfaces, not loopback, to
# be reachable once their ports are published. (Harmless here; only affects the
# image.)
ENV REGISTRY_HOST=0.0.0.0 \
    FRONTEND_HOST=0.0.0.0 \
    WORKER_HOST=0.0.0.0

# Registry 8000, Worker 8001, Frontend 8080 -- publish whichever the role needs.
EXPOSE 8000 8001 8080

# Default role is the Registry; override with any of the commands above.
CMD ["python", "-m", "nano_dynamo.registry.main"]
