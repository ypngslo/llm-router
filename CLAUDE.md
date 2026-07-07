# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`llm-router` is a **configuration repo** plus two small host programs. It declares a
self-hosted AI stack: a LiteLLM proxy (the "front door" / router) exposing one
OpenAI-compatible endpoint, a sleep-aware wake proxy (`waker.py`) that time-shares the
GPU between vLLM model servers, and a few sibling containers (audio, document
conversion). There is no build step and no test suite — the artifacts are
`docker-compose.yml`, `llm-config.yml`, `models.toml`, `serve.py`, `waker.py`, and
`infra.toml`. (`serve.sh` is the legacy launcher, superseded by `serve.py` + `models.toml`.)

## Architecture

```
client ──Bearer LITELLM_MASTER_KEY──▶ LiteLLM (container, 127.0.0.1:4000)
                                          │  proxies by model_name
                                          ├─ qwen-chat ─┐
                                          ├─ embeddings ┤   waker.py :8008 (sleep-aware proxy)
                                          ├─ rerank ────┤   wakes the target vLLM model on
                                          ├─ qwen-coder ┘   demand, sleeps idle ones for room
                                          │                   ├─ :8001 chat+VISION (Qwen3.5-4B, 0.55)
                                          │                   ├─ :8002 embed-sm (Qwen3-Emb-0.6B, 0.10)
                                          │                   ├─ :8004 rerank (Qwen3-Rerank-0.6B, 0.10)
                                          │                   └─ :8003 coder 32B AWQ (0.90, disabled)
                                          ├─ transcribe → 127.0.0.1:8005  parakeet STT container (CPU)
                                          └─ tts        → 127.0.0.1:8006  kokoro TTS container (CPU)
docling clients ──X-Api-Key──▶ Caddy /docling ──▶ :8007 waker passthrough ──▶ :8011 docling
                                                  (docker start/stop on demand, GPU ~5 GB)
```

- **LiteLLM** runs as a Docker container with `network_mode: host`, binds 127.0.0.1:4000
  only (per infra, only Caddy listens publicly), and mounts `llm-config.yml` read-only.
  Router hardening (retries, retry_policy, cooldown_time: 0 on local deployments) lives in
  that file, plus a **commented CLOUD BACKUP block**: uncomment + add a key to `.env` to get
  transparent local→cloud failover (same `model_name`, `order: 2`).
- **vLLM servers** are host systemd units. `serve.py <slug>` launches one model defined in
  `models.toml`; before exec'ing vLLM it derives the model's footprint (weights from the HF
  cache, KV/token from config.json), checks live free VRAM, and **refuses to start** if the
  configured `memory` fraction can't deliver `maxlen` tokens of KV (`--fit` auto-sizes
  instead). Ports are NOT in models.toml — infra.toml owns them (`Environment=PORT=<n>` in
  the unit); manual runs need `PORT=<n> ./serve.py <slug>`.
- **waker.py** (:8008) is the sleep-aware proxy LiteLLM actually talks to
  (`api_base: http://localhost:8008/<slug>/v1`). Every vLLM unit runs with
  `--enable-sleep-mode` + `VLLM_SERVER_DEV_MODE=1`; the waker wakes the requested model,
  first sleeping idle ones (LRU, never mid-request, `waker_grace` anti-thrash) when the
  budget (`waker_budget`, sum of awake `memory` fractions) won't hold both. Per-model
  `sleep_level`/`sleep_ttl` live in models.toml. See `docs/wake-proxy.md`; rollback =
  point llm-config api_bases back at the direct ports.
- **qwen-chat is also the vision endpoint** — Qwen3.5-4B is natively multimodal and vLLM
  serves its vision encoder by default; `image_url` content parts work today (verified).
- **Audio containers** (parakeet STT, kokoro TTS) run on CPU — zero VRAM, unaffected by GPU
  swaps — and are fronted by LiteLLM like the vLLM backends.
- **docling-serve** is NOT an OpenAI API: it gets its own Caddy route (`/docling`),
  authenticated with `DOCLING_API_KEY` (X-Api-Key header). GPU build (cu128 pinned —
  Blackwell needs it; CUDA images have no `latest` tag). It is a **managed tenant of
  waker.py** (`[[container]]` in models.toml): the waker binds its public :8007 and
  forwards to the real container on :8011, docker-stopping it when coder needs VRAM
  and starting it on demand; `sleep_ttl` (30 min) auto-stops it, which is also the
  mitigation for its known upstream GPU/RAM leaks.
- **server-infra integration** (`infra.toml`) declares all services to the external `infra`
  CLI (`~/code/system/server-infra`): reserves ports, renders systemd/compose units
  (incl. `environment =` vars like `VLLM_SERVER_DEV_MODE=1` for the coder), wires Caddy.

## Common commands

```bash
./serve.py --list           # model table + live fit prediction (weights, KV tokens)
PORT=8001 ./serve.py chat   # manual foreground run (systemd normally does this)
curl -s localhost:8008/waker/status | jq        # sleep/wake state of every vLLM backend
curl -X POST localhost:8008/waker/sleep/chat    # force-park a model (waker/wake/<slug> reverses)

docker compose up -d        # litellm + parakeet + kokoro + docling
docker compose logs -f litellm
docker restart litellm      # after editing llm-config.yml (mounted ro)

# Smoke tests through the proxy (key from .env):
curl http://127.0.0.1:4000/v1/models -H "Authorization: Bearer $LITELLM_MASTER_KEY"
curl http://127.0.0.1:4000/v1/audio/speech -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts","voice":"af_bella","input":"hello"}' -o out.mp3
curl http://127.0.0.1:4000/v1/audio/transcriptions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -F file=@out.mp3 -F model=transcribe
```

`.env` (gitignored) must define `LITELLM_MASTER_KEY` and `DOCLING_API_KEY`; see `.env.example`.

## GPU / VRAM is the central constraint

One RTX 5090 = **32 GB**. Each `vllm serve` reserves a fraction up front (`memory` in
`models.toml` → `--gpu-memory-utilization`). Models co-run only if enabled fractions sum
to **< ~0.95**; `serve.py`'s launch-time fit check enforces reality (see
`docs/archive/serve-sizing-plan.md`).

- Resident set: chat (0.55) + embed-sm (0.10) + rerank (0.10) = 0.75, plus docling ~5 GB.
- `coder` (32B AWQ, 0.90) runs **solo**: waker.py swaps it in on demand (sleeps chat,
  wakes coder in seconds; reaper parks it after `sleep_ttl`). Activation runbook:
  `docs/wake-proxy.md`.
- `maxlen` caps context → KV-cache VRAM; lower it first when a model won't fit.
  `--kv-cache-dtype fp8` halves KV (commented upgrade note on the chat entry).
- `overhead_gb` per model is vLLM's beyond-weights+KV reserve — measured for chat/embeds,
  an estimate for coder; replace estimates with measured values after real runs.

## Editing rules / gotchas

- **`model` in `llm-config.yml` (hosted_vllm/<slug>) must match `slug` in `models.toml`**
  (vLLM's `--served-model-name`), *not* the HF repo path. The `model_name` (`qwen-chat`,
  `embeddings`, `transcribe`, ...) is the separate public name clients call.
- **Enabling a vLLM model is a 3-place change**: `enable = true` in `models.toml`,
  uncomment its block in `llm-config.yml` (+ `docker restart litellm`), and ensure the
  `infra.toml` service is registered + enabled. Keep ports consistent: chat 8001,
  embed 8002, coder 8003, rerank 8004, parakeet 8005, kokoro 8006, docling 8007,
  waker 8008. New vLLM models are picked up by waker.py automatically (it parses
  infra.toml + models.toml at startup — restart the waker unit after adding one).
- **Qwen3.5 chat tool-calling**: `--enable-auto-tool-choice --tool-call-parser qwen3_xml`
  (XML-style calls — *not* `hermes`, which expects JSON). Qwen3.5 needs a vLLM from `main`.
- **vLLM 0.22** dropped `--task`; embedding/rerank models use `--runner pooling`
  (serve.py sets it for `task = "embed"` / `"score"`). The reranker needs the
  `--hf-overrides` JSON in its params — written without spaces because serve.py splits
  `params` on whitespace.
- `serve.py` uses an **absolute path for uv** on purpose: systemd runs units with a
  stripped env. Relative `project` in `models.toml [defaults]` resolves against the repo
  dir, which is systemd-safe.
- llm-config.yml is mounted read-only into the container — edits need `docker restart litellm`.

## Status / in-flight work

- `coder` is fully wired for on-demand swapping via waker.py but `enable = false`;
  activation steps in `docs/wake-proxy.md` (first boot needs the card free).
- Cloud fallback + budget blocks in `llm-config.yml` are commented until a cloud API key
  lands in `.env`.
- P2 of `docs/archive/serve-sizing-plan.md` (record real runs → learn `overhead_gb`)
  is not yet implemented.
- Optional next steps live as comments: fp8 KV on chat (doubles context), Postgres +
  LiteLLM spend UI, ComfyUI image gen (on-demand only), embed/rerank on CPU via
  Infinity if coder-vs-embeddings contention ever matters.
