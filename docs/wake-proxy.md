# waker.py — the sleep-aware wake proxy

One RTX 5090 can't hold every model at once (chat 0.55 + embed 0.10 + rerank 0.10
co-fit; coder needs 0.90 alone). Instead of manually starting/stopping units, every
vLLM server runs **permanently** with `--enable-sleep-mode`, and `waker.py` (:8008)
sits between LiteLLM and all of them:

```
LiteLLM ──▶ waker :8008 ──▶ /chat/…     → :8001  (awake?  forward)
                        ├─▶ /embed-sm/… → :8002  (asleep? wake first, then forward)
                        ├─▶ /rerank/…   → :8004
                        └─▶ /coder/…    → :8003  (no room? sleep idle models first)
Caddy /docling ─▶ waker :8007 ─▶ :8011 docling container (docker start/stop on demand)
```

A sleeping vLLM process keeps its compiled CUDA graphs but releases its GPU memory,
so "swapping models" is seconds, not the 1–2 minute cold boot. Clients never see the
mechanics — just a slow first token after a swap.

Non-vLLM GPU tenants join the same budget via `[[container]]` blocks in models.toml
(currently docling, `memory = 0.16`): instead of sleep/wake the waker docker-stops
and -starts them. The waker binds their stable public port (`listen`, 8007 for
docling — what Caddy routes to) and forwards to the real container (`upstream`,
8011), so clients never notice the lifecycle. Its `sleep_ttl` doubles as leak
mitigation: docling-serve leaks GPU/RAM slowly, and a stopped container is the
one guaranteed reclaim.

## How it decides

- Request for model X, X awake → forward immediately (adds ~ms).
- X asleep → check the VRAM budget (`waker_budget`, default 0.92, sum of awake
  `memory` fractions). If X doesn't fit, put idle models to sleep — least-recently-used
  first, never one with requests in flight, never one used within the last
  `waker_grace` seconds (default 60, anti-thrash). Then wake X and forward.
- Nothing evictable and X doesn't fit → fast `503` with `Retry-After` so LiteLLM's
  retries (and cloud fallbacks, once enabled) take over.
- A model with `sleep_ttl > 0` is parked automatically after that many idle seconds
  (coder: 900). `sleep_ttl = 0` means "stay awake until someone needs the room"
  (chat, embed, rerank).
- Backends without sleep support (unit not yet restarted with the new flags, or
  down) are passed through untouched — the waker degrades to a plain proxy.

## Config

No new config files. The waker reads:
- `infra.toml` — which vLLM units exist + their ports (`./serve.py <slug>`).
- `models.toml` — `memory` per slug, plus `sleep_level` (1 = weights→CPU RAM,
  ~1 s wake, costs RAM; 2 = discard weights, ~seconds wake from NVMe, ~0 RAM) and
  `sleep_ttl`. Box RAM is tight (~9 GB spare of 59), so only the 0.6B models use
  level 1; chat and coder use level 2.
- `[defaults]` in models.toml may set `waker_budget` / `waker_grace`.

vLLM only exposes `/sleep`, `/wake_up`, `/is_sleeping` under `VLLM_SERVER_DEV_MODE=1`
— infra.toml bakes that into every vLLM unit. The endpoints (and the waker itself)
are unauthenticated but loopback-only + ufw-denied, the same trust model as the
vLLM ports.

## Operations

```bash
curl -s localhost:8008/waker/status | jq        # who's awake, in flight, idle
curl -X POST localhost:8008/waker/sleep/chat    # force-park now
curl -X POST localhost:8008/waker/wake/chat     # force-wake now
journalctl -u waker -f                          # logs (sleep/wake decisions)
```

LiteLLM's `api_base` for every vLLM model is `http://localhost:8008/<slug>/v1`
(see llm-config.yml). To bypass the waker entirely (rollback), point the api_bases
back at the direct ports (8001/8002/8004) and `docker restart litellm`.

## Enabling the coder

1. `models.toml`: set `enable = true` on the coder block.
2. `llm-config.yml`: uncomment the `qwen-coder` entry → `docker restart litellm`.
3. `infra register && infra deploy && infra enable vllm-coder`.
4. First boot needs the card: `curl -X POST localhost:8008/waker/sleep/chat` first
   (serve.py's fit check runs at unit start; `overhead_gb = 6.0` is an estimate —
   if it refuses wrongly, launch once with `--fit` and record the real overhead).
   Expect a ~19 GB HF download on the very first start.
5. After that it's hands-off: a `qwen-coder` request sleeps chat and wakes coder;
   15 idle minutes later the reaper parks coder and the next `qwen-chat` request
   wakes chat back up.

## Known tradeoffs

- **coder + embeddings can't co-reside** (0.90 + 0.10 ≈ 1.0 > budget): while coder
  is awake, an embeddings request will evict it only if coder has been idle past the
  grace window; otherwise the request 503s and LiteLLM retries. Heavy dad-rag
  ingestion during a long coder session will be lumpy. If that ever matters, move
  embed/rerank off-GPU (Infinity/TEI on CPU) and the conflict disappears.
- **coder + docling contend the same way** (0.90 + 0.16 > budget): waking coder
  evicts an idle docling (docker stop) and vice versa, with docling's longer
  `grace = 300` protecting active ingestions — async-conversion status polls count
  as activity, so a watched conversion keeps its container alive. Interleaving
  coder work with bulk ingestion means swap latency each way (docling cold start ≈
  container boot + lazy model load on the first page).
- **Async docling conversions need a live poller**: a *sync* conversion holds
  in_flight and is safe at any length; an *async* conversion whose client stops
  polling for >300 s looks idle and can be evicted mid-task by a coder wake.
  Use sync, or keep polling, for conversions you can't afford to re-run.
- First request after a level-2 wake re-reads weights from NVMe (seconds for the
  4B, tens of seconds for the 32B); LiteLLM's `num_retries`/timeout absorb it.
- **Level-2 wake needs an explicit weight reload.** `/wake_up` after a level-2
  sleep re-allocates VRAM but leaves it *uninitialized* — the model serves fluent
  garbage (verified live on vLLM 0.22.1). waker.py therefore always follows a
  level-2 wake with `POST /collective_rpc {"method": "reload_weights"}` before
  routing traffic. If you ever wake a level-2 backend by hand, do the same — and
  the fix for an already-corrupted backend is that same reload call (no unit
  restart needed).
