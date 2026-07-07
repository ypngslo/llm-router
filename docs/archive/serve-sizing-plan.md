# Plan: make `serve.py` size-aware at launch (fail-fast on bad fit, optional auto-size)

Goal: stop hand-guessing whether a model fits. The intelligence lives **in the launch
path** — `serve.py <slug>` is what systemd runs when `infra enable`/deploy starts a unit,
so that is the only moment serve.py exists and can inspect the live GPU. Before it `exec`s
vLLM, it derives the model's footprint, reads what's free on the card right now, and decides
whether the configured budget will actually deliver the requested context. This is exactly
the check that was missing when chat silently started with 6,192 KV tokens against a 32,768
`maxlen`.

**Not a human-facing tool.** Nobody runs `serve.py` by hand to "figure things out." Every
behavior here is automatic, triggered by the unit starting.

## Decisions locked in

- **Default = fail-fast (A).** If the model won't get its `maxlen` worth of KV in the
  currently-free VRAM, the unit **refuses to start** and writes a clear reason to the
  journal. Loud and visible beats silently-starved.
- **`--fit` flag = auto-size (B).** When set, instead of failing, serve.py computes a budget
  that fits the free VRAM and launches with that (overriding the configured fraction).
- **Config schema stays as-is.** `memory` remains the vLLM `--gpu-memory-utilization`
  fraction (the authoritative knob, 1:1 with what vLLM consumes). `maxlen` is the context
  target serve.py validates against. No new "target/concurrency" abstraction — serve.py just
  checks that the chosen fraction delivers `maxlen`. (An optional `concurrency` multiplier can
  come later; default is "hold one full-length sequence," i.e. `maxlen`.)

## Ground truth we verified this session

- **KV bytes/token** = `2 × layers × kv_heads × head_dim × dtype_bytes`, from `config.json`.
  Qwen3-4B arch (36 / 8 / 128 / bf16) = **144 KB/token** (72 KB with fp8 KV).
- **Weights size** = sum of `*.safetensors` in the HF snapshot dir. Exact, no run needed.
- **Live free VRAM** = `nvidia-smi --query-gpu=memory.free` (and `--query-compute-apps` for
  per-PID). Card = 32,607 MiB.
- **Predictable** at launch: weights + KV bytes/token. **Not predictable**: vLLM's
  activation/CUDA-graph/profiling reserve (~8 GB observed on chat) — only a real run reveals
  it, so it's learned (P2) and bootstrapped from a conservative constant until then.
- vLLM exposes `--kv-cache-memory-bytes` to pin KV exactly — what auto-size (B) emits.

---

## P0 — Launch-time fit check, fail-fast (the core; build first)

**What:** At the top of `serve.py <slug>`, before `os.execvp`:
1. Derive weights (safetensors) + KV/token (config.json); halve KV if `params` has fp8.
2. Read live free VRAM from `nvidia-smi`.
3. Predict KV tokens the configured `memory` fraction yields *given current free VRAM*:
   `(min(memory × 32607MiB, free) − weights − overhead) / (KV_bytes/token)`.
4. **If predicted tokens < `maxlen` → exit non-zero with a clear journal line** and do not
   launch. Else `exec` vLLM unchanged.

**Example failure message:**
```
[serve.py] chat: configured memory=0.55 → ~6,200 KV tokens, but maxlen=32768.
  free VRAM 4.2 GB (embed resident). Need ~18 GB bf16 / ~9 GB fp8 for 32k.
  Fix: raise memory, add --kv-cache-dtype fp8, lower maxlen, or shrink embed.  (--fit to auto-size)
```

**Why:** Turns the worst failure mode (boots fine, serves a crippled context) into a loud,
legible refusal. Pure read-only math + one nvidia-smi call; the exec path is otherwise
untouched, so systemd signal handling is unaffected.

**Effort:** small. **Risk:** low — but it *can* block a boot, so the message must be
actionable and the check conservative (don't false-negative a fit that actually works).

## P1 — `--fit` flag: auto-size to free VRAM (opt-in override)

**What:** With `--fit` (or `auto_size = true` in `[defaults]`), don't fail — instead compute
the largest budget that fits current free VRAM and launch with it: either a derived
`--gpu-memory-utilization` or, preferred, a pinned `--kv-cache-memory-bytes` sized to hit
`maxlen` (+ headroom). The configured `memory` becomes a cap, not a fixed value.

**Why:** The system self-arranges — whoever starts second sizes itself to the remainder, no
hand-tuned fractions. This is the "determine it on its own" behavior, kept opt-in because it
overrides config and sits on the boot path.

**Effort:** small–medium (inverts P0's math). **Risk:** medium — it launches with a computed
value, so it must clamp hard and fall back to fail-fast if the computed budget is implausible.

## P2 — Automatic run recording → learn the overhead (`.runstats.json`)

**What:** Record each real run's actual numbers so the P0/P1 estimate of vLLM's overhead
stops being a guess. Keyed by (model, gpu, vllm version): real `num_gpu_blocks`/KV tokens,
reserved VRAM, prefix-caching state, OOM/health, timestamp.

**How (automatic, no manual step), pick one:**
- **Detached one-shot recorder:** serve.py double-forks a tiny background process at launch
  that waits ~60 s, scrapes this model's `/metrics` + `nvidia-smi`, writes `.runstats.json`,
  exits. The main process still `exec`s vLLM — the recorder sits *beside* it, never between
  systemd and vLLM. Fully hands-off.
- **systemd timer (P4):** same scrape on a schedule instead of a forked child.

`.runstats.json` is **gitignored** (gpu- and version-specific). Next launch reads it so the
overhead term in P0/P1 is exact, not bracketed.

**Effort:** medium. **Risk:** low (read-only scrape, isolated from the serving process).

## P3 — Read-only diagnostics (optional, nice-to-have)

**What:** `serve.py --list` prints derived weights / KV-per-token / predicted-fit per enabled
model without launching — handy when debugging why a unit refused to start. Not required for
the automatic behavior; it's just a window into the same math.

**Effort:** small. **Risk:** none.

## P4 — Auto-probe systemd timer (only if P2's detached recorder isn't chosen)

**What:** A `vllm-probe.timer` registered via `infra.toml` that runs the scrape a minute after
units settle, keeping `.runstats.json` fresh. Alternative delivery for P2, not additional value.

**Effort:** small. **Risk:** low.

---

## Explicitly NOT doing

- **No human-facing `plan`/tuning CLI.** Everything is automatic at unit start.
- **No fork-and-babysit of the launch path.** `os.execvp` stays; recording is a detached
  child or a timer, never a wrapper between systemd and vLLM.
- **No new config abstraction.** `memory` stays the vLLM fraction; `maxlen` is the target.
- **No committed/portable stats**, no general autotuner — derived math + one learned overhead
  constant is enough for a 1–3 model box.

## Build order

P0 (fail-fast fit check) → P2 (learn the overhead so P0 is exact) → P1 (`--fit` auto-size).
P3/P4 only if useful. P0 alone already prevents the starvation class of bug.
