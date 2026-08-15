# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **configuration repo** plus two small host programs declaring a self-hosted AI
stack: a LiteLLM proxy (one OpenAI-compatible endpoint on :4000), a sleep-aware
wake proxy (`waker.py` on :8008) that time-shares one 32 GB RTX 5090 between vLLM
model servers, and sibling containers (audio on CPU, docling on GPU). No build
step, no test suite.

**Architecture, component tour, and design rationale live in [README.md](README.md)
— read it first.** Wake-proxy deep dive (eviction rules, sleep levels, tradeoffs,
coder activation): `docs/wake-proxy.md`.

## The model-folder contract

Every model is a folder under `models/`. **The folder name IS the slug** (vLLM's
`--served-model-name`) — nothing else defines it. A folder holds up to three files:

| File | Read by | Contents |
|---|---|---|
| `model.toml` | serve.py, waker.py | bare keys, no table header: `enable`, `name` (HF repo), `memory`, `maxlen`, `overhead_gb`, `params`, `sleep_level`, `sleep_ttl` |
| `litellm.yml` | LiteLLM (via `llm-config.yml`'s `include:` list) | its public `model_list` entries — `model: hosted_vllm/<slug>`, `api_base: http://localhost:8008/<slug>/v1` |
| `infra.toml` | `infra` CLI | its `[[service]]` block: port, systemd unit / compose service |

Variants: cloud-only model = just `litellm.yml` (deepseek-pro/-flash); managed
GPU container = `container.toml` instead of `model.toml` (docling); bench/rollback
config = just `model.toml` with `enable = false` (chat-27b, chat-4b, embed).
Shared knobs (uv project, bind_host, `waker_budget`/`waker_grace`) live in
`models/defaults.toml`.

The public `model_name` clients call (`qwen-chat`, `embeddings`, `transcribe`, ...)
lives only in `litellm.yml` and is independent of the slug. Root `infra.toml`
holds only the stack chassis (litellm + waker).

## Workflows

**Add / enable a model** (three touch points, two of them in the folder):
1. `models/<slug>/` — write `model.toml` (`enable = true`), `litellm.yml`, `infra.toml`.
2. `llm-config.yml` — add (or uncomment) its line in the `include:` list →
   `docker restart litellm`.
3. `cd models/<slug> && infra register && infra deploy && infra enable`.
4. `infra restart waker` — the waker parses the model folders at startup only.

**Disable**: `infra disable <service>`, comment its include line, restart litellm.
Set `enable = false` in `model.toml` to record intent.

**Swap the chat model** (e.g. rollback 9B → 4B): the folder name is the slug, so
swap the folder names (`models/chat` ↔ `models/chat-4b`), flip the `enable`
flags, and `infra restart vllm-chat`. The systemd unit runs `serve.py chat` —
it serves whatever `models/chat/` contains.

**Retune** (`memory`/`maxlen`/`params`): edit `model.toml`, `infra restart
vllm-<name>`. serve.py refuses to start a config that won't fit (run with
`--fit` to auto-size); a real run records measured KV capacity into
`.runstats.json`, which future fit checks and `--list` prefer over the formula
(`*` marker — the formula badly underestimates hybrid-attention models).

## Common commands

```bash
./src/serve.py --list           # model table + live fit prediction (weights, KV tokens)
PORT=8001 ./src/serve.py chat   # manual foreground run (systemd normally does this)
curl -s localhost:8008/waker/status | jq        # sleep/wake state of every backend
curl -X POST localhost:8008/waker/sleep/chat    # force-park (waker/wake/<slug> reverses)

docker compose up -d        # litellm + parakeet + kokoro + docling
docker restart litellm      # after editing llm-config.yml or any litellm.yml (ro mounts)

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

One RTX 5090 = **32 GB**. Each `vllm serve` reserves `memory` × card up front;
models co-run only if enabled fractions sum under the waker budget (0.92).
Resident trio: chat 0.62 (Qwen3.5-9B online-FP8, 64K ctx) + embed-sm 0.10 +
rerank 0.10 = 0.82. docling (0.16) doesn't fit beside the trio — its bursts
evict an idle tenant (LRU) and vice versa. coder (0.90) runs solo via waker
swaps. Lower `maxlen` first when a model won't fit; `overhead_gb` estimates get
replaced by measured runstats after a real run.

## Gotchas

- **Qwen3.5 non-negotiable flags** (hybrid GDN family): `--dtype bfloat16` (fp16
  crashes on mixed dtypes), `--max-num-batched-tokens 2096` (GDN cache alignment),
  `--enable-auto-tool-choice --tool-call-parser qwen3_xml` (XML-style calls — *not*
  `hermes`, which expects JSON). Qwen3.5 needs a vLLM from `main`. qwen-chat is
  also the vision endpoint (natively multimodal; `image_url` parts work).
- **vLLM 0.22** dropped `--task`; embedding/rerank models use `--runner pooling`
  (serve.py sets it for `task = "embed"` / `"score"`). The reranker's
  `--hf-overrides` JSON is written without spaces because serve.py splits
  `params` on whitespace.
- **In a model's `infra.toml`**: `working_dir` resolves against that file's dir
  (`"../.."` = repo root); a relative `exec_start` resolves against `working_dir`
  — so `"./src/serve.py <slug>"`, NOT `"../../src/serve.py <slug>"`.
- **LiteLLM's `include:` appends** `model_list` across files (verified in source);
  fragment paths resolve against `/app` in the container — the compose file mounts
  `./models:/app/models:ro` alongside the config. Both mounts are read-only:
  every LiteLLM-visible edit needs `docker restart litellm`.
- **Ports** (allocated by infra, pinned in each folder's infra.toml): chat 8001,
  embed 8002, coder 8003, rerank 8004, parakeet 8005, kokoro 8006, docling 8007
  (waker passthrough; real container 8011), waker 8008.
- serve.py uses an **absolute path for uv** (systemd's stripped env); relative
  `project` in `models/defaults.toml` resolves against the repo dir. PORT comes
  from the unit environment, never from config.
- **Level-2 wake serves garbage without a weight reload** — waker.py handles it;
  if you wake a level-2 backend by hand, POST `/collective_rpc
  {"method":"reload_weights"}` (details in docs/wake-proxy.md).

## Status / in-flight work

- Chat (Qwen3.5-9B FP8, 64K ctx, measured 422K KV tokens at memory=0.62) is
  live again via `infra enable vllm-chat` — first boot through the per-folder
  infra config. embed-sm, rerank, and the audio containers (parakeet/kokoro)
  are still parked from the 2026-08 ComfyUI stint: re-enable with
  `infra enable vllm-embed` / `vllm-rerank` and `infra enable parakeet` /
  `kokoro` when needed.
- `coder` is fully wired for on-demand swapping but disabled; activation:
  `docs/wake-proxy.md`. The 27B/35B chat upgrades were measured and rejected
  (notes in `models/chat-27b/model.toml`).
- Cloud fallback is live: `qwen-chat` → DeepSeek V4 Pro via OpenRouter
  (`order: 2`), also the `context_window_fallbacks` target; `deepseek-flash` is
  the cheap 1M-ctx explicit pick. Cloud spend hard-capped ($25/30d).
- Optional next steps live as comments: Postgres + LiteLLM spend UI, ComfyUI
  image gen (on-demand only), embed/rerank on CPU via Infinity if
  coder-vs-embeddings contention ever matters.
