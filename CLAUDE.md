# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`llm-router` is a **configuration repo** plus two small host programs declaring a
self-hosted AI stack: a LiteLLM proxy (one OpenAI-compatible endpoint), a sleep-aware
wake proxy (`waker.py`) that time-shares the GPU between vLLM model servers, and a few
sibling containers (audio, document conversion). There is no build step and no test
suite. **Each model is a folder under `models/`** (`model.toml` + `litellm.yml` +
`infra.toml`; shared knobs in `models/defaults.toml`); the root artifacts are
`docker-compose.yml`, `llm-config.yml`, `serve.py`, `waker.py`, and `infra.toml`
(the stack chassis: litellm + waker only).

**The architecture diagram, component tour, request-flow, and design rationale live in
[README.md](README.md) — read it first.** Deep dive on the wake proxy (eviction rules,
sleep levels, known tradeoffs, coder activation runbook): `docs/wake-proxy.md`.
Rollback from the waker = point llm-config api_bases back at the direct ports.

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

`.env` (gitignored) must define every key `llm-config.yml` resolves; see `.env.example`.

## GPU / VRAM is the central constraint

One RTX 5090 = **32 GB**. Each `vllm serve` reserves a fraction up front (`memory` in
its `model.toml` → `--gpu-memory-utilization`). Models co-run only if enabled fractions sum
to **< ~0.95**; `serve.py`'s launch-time fit check enforces reality.

- Resident set: chat (0.62, Qwen3.5-9B online-FP8) + embed-sm (0.10) + rerank (0.10)
  = 0.82. docling (0.16) exceeds the waker budget beside the trio, so its bursts
  briefly evict an idle tenant (LRU) and vice versa.
- `coder` (32B AWQ, 0.90) runs **solo**: waker.py swaps it in on demand (sleeps chat,
  wakes coder in seconds; reaper parks it after `sleep_ttl`).
- `maxlen` caps context → KV-cache VRAM; lower it first when a model won't fit.
- `overhead_gb` per model is vLLM's beyond-weights+KV reserve — measured values in
  `.runstats.json` (written by serve.py's detached recorder) override the configured
  estimate; fit checks and `--list` prefer measured (`*` marker) over the formula,
  which badly underestimates hybrid-attention models.

## Editing rules / gotchas

- **The folder name IS the slug** (vLLM's `--served-model-name`). In a folder's
  `litellm.yml`, `model: hosted_vllm/<slug>` and the api_base path must use the folder
  name, *not* the HF repo path. The `model_name` (`qwen-chat`, `embeddings`,
  `transcribe`, ...) is the separate public name clients call.
- **Enabling a vLLM model**: everything lives in its folder — `enable = true` in
  `models/<slug>/model.toml`, uncomment its line in `llm-config.yml`'s `include:` list
  (+ `docker restart litellm`), and `cd models/<slug> && infra register && infra enable`.
  Keep ports consistent: chat 8001, embed 8002, coder 8003, rerank 8004, parakeet 8005,
  kokoro 8006, docling 8007, waker 8008, docling-app 8011. New models are picked up by
  waker.py automatically (it parses the model folders at startup — restart the waker
  unit after adding one).
- **In a model's `infra.toml`**, `working_dir` resolves against that file's dir
  (`"../.."` = repo root) and a relative `exec_start` resolves against `working_dir`
  (so `"./serve.py <slug>"`, not `"../../serve.py <slug>"`).
- **Qwen3.5 non-negotiable flags** (hybrid GDN family): `--dtype bfloat16` (fp16
  crashes on mixed dtypes), `--max-num-batched-tokens 2096` (GDN cache alignment),
  `--enable-auto-tool-choice --tool-call-parser qwen3_xml` (XML-style calls — *not*
  `hermes`, which expects JSON). Qwen3.5 needs a vLLM from `main`.
- **vLLM 0.22** dropped `--task`; embedding/rerank models use `--runner pooling`
  (serve.py sets it for `task = "embed"` / `"score"`). The reranker needs the
  `--hf-overrides` JSON in its params — written without spaces because serve.py splits
  `params` on whitespace.
- `serve.py` uses an **absolute path for uv** on purpose: systemd runs units with a
  stripped env. Relative `project` in `models/defaults.toml` resolves against the repo
  dir, which is systemd-safe.
- llm-config.yml is mounted read-only into the container — edits need `docker restart litellm`.
- **qwen-chat is also the vision endpoint** — Qwen3.5 is natively multimodal and vLLM
  serves the vision encoder by default; `image_url` content parts work.

## Status / in-flight work

- Chat is Qwen3.5-9B quantized to FP8 at load (`--quantization fp8`), 64K served
  context (measured 422K KV tokens at memory=0.62 — ample headroom). The 4B entry
  is the documented rollback; the 27B was measured and rejected (doesn't fit).
- `coder` is fully wired for on-demand swapping via waker.py but `enable = false`;
  activation steps in `docs/wake-proxy.md` (first boot needs the card free).
- Cloud fallback is live: `qwen-chat` fails over to DeepSeek V4 Pro via OpenRouter
  (`order: 2`), which is also the `context_window_fallbacks` target. `deepseek-flash`
  is the cheap 1M-ctx explicit-pick model. Cloud spend hard-capped ($25/30d).
- Optional next steps live as comments: Postgres + LiteLLM spend UI, ComfyUI image gen
  (on-demand only), embed/rerank on CPU via Infinity if coder-vs-embeddings contention
  ever matters.
