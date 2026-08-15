# llm-router

A self-hosted AI platform on **one consumer GPU**. A single OpenAI-compatible
endpoint serves chat + vision, embeddings, reranking, speech-to-text,
text-to-speech, and document conversion — with the models **time-sharing a
32 GB RTX 5090** through vLLM sleep mode, and transparent cloud failover
(with a hard spend cap) when the local stack can't answer.

> This is my personal infrastructure, published as a working demonstration.
> It runs my agents, my RAG pipeline, and my document ingestion every day, but
> it is not packaged for reuse: every number in the model configs is tuned to
> one specific box (32 GB of VRAM, ~59 GB of RAM, NVMe, Blackwell CUDA). Read
> it as a reference for *how* to build this kind of thing, not as an installable.

There is no build step and no framework — the whole system is two small Python
programs (~1,000 lines total) plus config. **Each model is a folder**: everything
that defines it — its vLLM server config, its public LiteLLM entry, its own
service declaration — lives in one place, and adding or retiring a model is a
folder operation:

```
models/
  chat/            Qwen3.5-9B — the resident chat + vision model
    model.toml       vLLM config: HF repo, VRAM fraction, context, sleep behavior
    litellm.yml      its public API entries (incl. its cloud-failover twin)
    infra.toml       its service: port, systemd unit  (cd here → infra register)
  embed-sm/  rerank/  coder/       same shape
  transcribe/  tts/                CPU containers: litellm.yml + infra.toml only
  docling/                         managed GPU container: container.toml instead
  deepseek-pro/  deepseek-flash/   cloud-only: just a litellm.yml
  chat-27b/  chat-4b/  embed/      bench + rollback configs (disabled)
  defaults.toml                    shared knobs (uv project, bind host, waker budget)
```

| File | Role |
|---|---|
| `serve.py` | Launches one vLLM server per model — and **refuses to start** what won't fit |
| `waker.py` | Sleep-aware reverse proxy that time-shares the GPU between models on demand |
| `llm-config.yml` | LiteLLM router: stack-wide routing/budget settings + the model include list |
| `docker-compose.yml` | LiteLLM + the CPU audio containers + docling |
| `infra.toml` | The stack chassis (LiteLLM + waker) declared to my external `infra` CLI |
| `docs/wake-proxy.md` | Design notes and known tradeoffs of the wake proxy |

## What one card serves

Clients see one endpoint (`/v1` with a Bearer key) and call models by name:

| `model` | Backing | VRAM | Notes |
|---|---|---|---|
| `qwen-chat` | Qwen3.5-9B, FP8 (vLLM, host) | 0.62 | 64K context, native vision (`image_url` works), XML tool calling |
| `embeddings` | Qwen3-Embedding-0.6B (vLLM) | 0.10 | `--runner pooling` |
| `rerank` | Qwen3-Reranker-0.6B (vLLM) | 0.10 | Cohere-style `/rerank`, cross-encoder |
| `transcribe` | Parakeet TDT 0.6B v3 (container) | 0 (CPU) | ~30× realtime on CPU |
| `tts` | Kokoro-82M (container) | 0 (CPU) | voice mixes like `af_bella(2)+af_sky(1)` |
| `qwen-coder` | Qwen2.5-Coder-32B AWQ (vLLM) | 0.90 | needs the card to itself — swapped in on demand (currently disabled) |
| `deepseek-pro` / `deepseek-flash` | OpenRouter | — | 1M-context cloud escape hatches |
| `openai/*`, `anthropic/*` | wildcard passthrough | — | any cloud model through the same endpoint |

Plus `docling-serve` for document conversion — not an OpenAI-shaped API, so it
gets its own authenticated route instead of a LiteLLM entry.

The chat + embeddings + reranker trio is resident (0.82 of the card). Everything
bigger is swapped in and out automatically — that's the interesting part.

## Architecture

```
client ──Bearer key──▶ LiteLLM (container, 127.0.0.1:4000)
                           │  routes by model_name; retries; cloud failover
                           ├─ qwen-chat ─┐
                           ├─ embeddings ┤  waker.py :8008 — sleep-aware proxy.
                           ├─ rerank ────┤  Wakes the target vLLM server, parking
                           ├─ qwen-coder ┘  idle ones first if the card is full.
                           │                  ├─ :8001 chat    Qwen3.5-9B (0.62)
                           │                  ├─ :8002 embed   Qwen3-Emb-0.6B (0.10)
                           │                  ├─ :8004 rerank  Qwen3-Rerank-0.6B (0.10)
                           │                  └─ :8003 coder   Qwen2.5-32B-AWQ (0.90)
                           ├─ transcribe ──▶ :8005 parakeet STT (CPU container)
                           ├─ tts ─────────▶ :8006 kokoro TTS (CPU container)
                           └─ deepseek-*, wildcards ──▶ OpenRouter / cloud APIs

docling clients ──X-Api-Key──▶ Caddy /docling ──▶ :8007 waker passthrough
                                                    └─▶ :8011 docling container
                                                        (docker start/stop on demand)
```

Three layers, each doing one job:

- **LiteLLM** is the front door: one API surface, auth, per-key budgets,
  retries, and failover policy. It never knows the GPU exists.
- **waker.py** is the GPU scheduler: LiteLLM points every local model at
  `http://localhost:8008/<slug>/v1`, and the waker guarantees the slug behind
  that path is awake before forwarding — evicting someone else if it has to.
- **vLLM servers** (one systemd unit per model, launched by `serve.py`) own
  the actual VRAM. Every one runs permanently with `--enable-sleep-mode`.

## The core problem: 32 GB is not enough

Each `vllm serve` process reserves a fixed fraction of the card up front
(`--gpu-memory-utilization`). My full lineup sums to well over 1.0, so
something has to give. The usual answers are bad: run fewer models, restart
units by hand, or buy another GPU.

The answer here is **vLLM sleep mode**. A sleeping vLLM process stays alive
and keeps its compiled CUDA graphs, but releases its GPU memory — so a "model
swap" is seconds, not a 1–2 minute cold boot. `waker.py` turns that primitive
into an automatic scheduler:

1. A request for model X arrives. If X is awake: forward (adds ~ms).
2. If X is asleep, sum the `memory` fractions of everything awake. While
   X won't fit under the budget (`waker_budget`, default 0.92), put idle
   models to sleep — least-recently-used first, never one with requests in
   flight, never one used within the last `waker_grace` seconds (anti-thrash).
3. Wake X, wait for `/health`, forward. Clients see a slow first token, never
   an error.
4. If nothing is evictable, fail *fast* with a `503 + Retry-After` — so
   LiteLLM's retry/failover machinery takes over instead of a hung request.

Details that took real debugging to get right:

- **Two sleep levels.** Level 1 parks weights in CPU RAM (~1 s wake) but the
  box only has ~9 GB spare, so only the 0.6 B models use it. Level 2 discards
  weights entirely (~0 RAM, wake reloads from NVMe page cache in seconds).
- **A level-2 wake serves garbage without an explicit reload.** vLLM's
  `/wake_up` re-allocates the memory but leaves it *uninitialized* — the model
  happily generates fluent nonsense (verified live on vLLM 0.22.1). The waker
  always follows a level-2 wake with `POST /collective_rpc
  {"method": "reload_weights"}` before routing traffic.
- **Non-vLLM tenants join the same budget.** docling-serve is a GPU container
  with no sleep support, so the waker manages it with `docker stop/start`
  instead, binding its stable public port itself and forwarding to the real
  container. Its idle-stop TTL doubles as the mitigation for docling's known
  GPU-memory leaks.
- **State reconciliation, not state assumption.** A background reaper probes
  every backend on an interval, parks anything idle past its `sleep_ttl`, and
  re-syncs the waker's view with reality — so a manually restarted unit or a
  crashed backend can't wedge the scheduler.

## serve.py: refuse to start what won't fit

The second failure mode of a shared card is a unit that boots, OOMs the GPU
(or silently starves its KV cache), and takes a neighbor down with it.
`serve.py` is a launcher that does the arithmetic *before* exec'ing vLLM:

- Derives the model's footprint with no test run: weights from the HF cache's
  actual shard sizes, KV bytes/token from `config.json` (attention layout ×
  dtype, halved for fp8 KV), plus a per-model measured `overhead_gb`.
- Reads live free VRAM from `nvidia-smi`, and **fails fast with a diagnosis**
  if the configured fraction can't deliver `maxlen` tokens of context —
  or auto-sizes the fraction with `--fit`.
- **Prefers measurement over formula.** At every launch it forks a detached
  recorder that waits for the server, scrapes the run's *actual* KV capacity
  from the journal, and stores it in `.runstats.json`. Future fit checks and
  `./serve.py --list` use measured truth — which matters, because the formula
  is wildly wrong for hybrid-attention models (Qwen3.5's GDN layers: the
  formula predicts ~13K tokens where a real run measures ~420K).
- The recorder doubles as a **watchdog**: if the server never becomes healthy,
  it SIGKILLs the wedged launch (after verifying via cgroup that the pid still
  belongs to the right systemd unit) so `Restart=on-failure` actually fires —
  vLLM can hang at boot in ways systemd otherwise reports as `active`.

```
$ ./serve.py --list
SLUG    NAME                                     ON  MEM   TASK      MAXLEN   WEIGHTS   PRED-KV-TOK
chat-27b Qwen/Qwen3.5-27B-GPTQ-Int4              no  0.74  generate  32768    ~16 GB    ?
chat    Qwen/Qwen3.5-9B                          yes 0.62  generate  65536    10.8 GB   422,012*
chat-4b Qwen/Qwen3.5-4B                          no  0.55  generate  12288    8.7 GB    ~13,660
embed-sm Qwen/Qwen3-Embedding-0.6B               yes 0.1   embed     4096     1.1 GB    14,432*
rerank  Qwen/Qwen3-Reranker-0.6B                 yes 0.1   score     4096     1.1 GB    14,576*
coder   Qwen/Qwen2.5-Coder-32B-Instruct-AWQ      no  0.9   generate  16384    ~19 GB    ?
```
(`*` = measured from a real run via `.runstats.json`, `~` = formula estimate —
note the formula's ~13K prediction for the 4B chat model vs six-figure measured
reality for its hybrid-attention sibling.)

## When local isn't enough: transparent cloud failover

Two LiteLLM mechanisms, both invisible to clients:

- **Same name, second priority.** `qwen-chat` has a second deployment entry —
  DeepSeek V4 Pro via OpenRouter with `order: 2`. Only when the local
  deployment's retries are exhausted (connection refused, timeout, 429, 5xx)
  does the request escalate to the cloud. Callers keep calling `qwen-chat`.
- **Context-window fallback.** A request over the local 64K window routes to
  the 1M-context cloud model instead of erroring.

Local calls cost $0, so a global `max_budget: 25` / `budget_duration: 30d`
in LiteLLM is effectively a hard monthly cap on cloud-failover spend. Requests
are traced to Langfuse (a sibling stack) tagged by virtual key, so spend and
latency are sliceable per app.

## Design rules the repo follows

- **Nothing is configured twice.** Ports live only in each model's `infra.toml`
  (baked into systemd units as `PORT=`); model facts live only in its
  `model.toml`. The waker derives its entire world by parsing the model folders
  at startup — adding a model needs zero waker config. `serve.py` reads the
  same `infra.toml` files right back to find its own unit name. The one
  list maintained by hand is `llm-config.yml`'s `include:` — LiteLLM appends
  each folder's `litellm.yml` to its model list, so a model's public identity
  ships next to the model.
- **Fail fast and loud, recover without a human.** The fit check refuses
  doomed launches; the waker 503s instead of queueing forever; the watchdog
  hands wedged processes back to systemd; the reaper reconciles drifted state.
- **Only the front door is reachable.** Everything binds loopback; the only
  public listeners are the HTTPS edge routes (LiteLLM with a master key,
  docling with an API key). The vLLM sleep endpoints — which are
  unauthenticated by design — share that loopback-only trust boundary.

## Operations cheatsheet

```bash
./serve.py --list                                # model table + live fit prediction
curl -s localhost:8008/waker/status | jq         # who's awake, in flight, idle
curl -X POST localhost:8008/waker/sleep/chat     # force-park a model
docker compose up -d                             # litellm + parakeet + kokoro + docling
docker restart litellm                           # after editing llm-config.yml

# Smoke test through the front door:
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen-chat","messages":[{"role":"user","content":"hello"}]}'
```

`.env` (gitignored) holds the keys — see `.env.example` for the full list.

## What's deliberately not here

- **Ports, units, and the HTTPS edge** are owned by a separate
  personal `infra` CLI that reads `infra.toml`; this repo only declares its
  services. Without it, the stack still runs by hand:
  `PORT=8001 ./serve.py chat`, `PORT=8008 ./waker.py`, `docker compose up -d`.
- **Observability** (Langfuse + Postgres for LiteLLM's virtual keys and spend
  logs) is a sibling stack this config points at.
- **Tests.** The programs are deliberately small enough to read, and the
  system's real test is the fit check + smoke curls above running on live
  hardware every day.

## License

MIT — see [LICENSE](LICENSE).
