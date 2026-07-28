# Registry

The Registry is nano-dynamo's stand-in for etcd: the shared source of truth
for "who's alive and what can they serve." Workers `POST /register` on
startup and `POST /heartbeat/{worker_id}` on a timer; anyone else calls
`GET /workers?model=...` to find the current live set.

## The reaper

The Registry never actively checks whether a worker is still alive — it
doesn't ping workers or wait for them to disconnect. Instead, a background
task (`_reap_loop`) wakes up every `reaper_interval_seconds` and deletes any
worker whose `last_heartbeat` is older than `heartbeat_ttl_seconds`.

This is expiry-based liveness, not active death-detection: a dead worker
isn't noticed the instant it dies, only once its heartbeat goes stale past
the TTL. That lag is a deliberate trade — it avoids needing health-check
pings or per-worker timeout/retry logic, at the cost of a worst-case
`heartbeat_ttl_seconds` delay before a dead worker actually disappears from
`/workers`. It's the same trade real etcd makes with lease expiry, which is
exactly what the Registry is standing in for here.
