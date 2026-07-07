#!/usr/bin/env python3
"""serve.py — launch a single VRAM-capped vLLM server for one model.

Config lives in models.toml (same dir); this file just reads it and execs vllm.

Usage:
  ./serve.py <slug>          start the named model in the foreground
  ./serve.py <slug> --fit    auto-size the budget to currently-free VRAM (override config)
  ./serve.py --list          show the model table + predicted fit

Before launching, serve.py derives the model's footprint (weights from the HF cache,
KV bytes/token from config.json), reads how much VRAM is free *right now*, and checks
whether the configured `memory` fraction will actually deliver `maxlen` tokens of KV.

  default        → fail fast: if it won't fit, refuse to start with a clear reason.
  --fit          → instead of failing, compute a budget that fits and launch with that.

VRAM: each `vllm serve` reserves `memory` (a fraction of the card) up front. Models
co-run only if their reservations fit alongside what's already resident.
"""

import os
import sys
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "models.toml"
# uv lives under ~/.local/bin which is NOT on systemd's stripped PATH, so call it
# by absolute path. Override via the environment.
UV = os.environ.get("UV", "/home/beans/.local/bin/uv")

GiB = 1024**3
MiB = 1024**2
# Bytes vLLM reserves beyond weights+KV (activations, CUDA graphs, profiling). NOT
# derivable from config and highly model-dependent — ~8 GiB on chat (generate, 32k)
# vs ~1.4 GiB on embed (pooling). So it's set PER-MODEL via `overhead_gb` in
# models.toml (bootstrap values measured this session; P2 will replace them with
# recorded run-stats). This default is only a fallback for a model that sets none.
DEFAULT_OVERHEAD_GB = 3.0
# Fail only when the shortfall is gross. The footprint math is approximate (vLLM's
# real reservation can drift from memory×total by the CUDA context), so a model that
# lands within this fraction of its target is treated as fitting — we catch chat's
# 6k-vs-32k starvation, not embed's 4.0k-vs-4.1k rounding.
FIT_TOLERANCE = 0.10
DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float": 4, "half": 2}


def load():
    with CONFIG.open("rb") as f:
        data = tomllib.load(f)
    defaults = data.get("defaults", {})
    models = {m["slug"]: m for m in data.get("model", [])}
    return defaults, models


def enabled(models):
    return [m for m in models.values() if m.get("enable")]


def human(b):
    return f"{b / GiB:.1f} GB" if b else "?"


# ── Footprint (derived from disk, no run needed) ─────────────────────────────────
def hf_snapshot(name):
    """Return the cached HF snapshot dir for `name`, or None if not downloaded yet.

    Mirrors HF's own cache resolution (HF_HUB_CACHE > HF_HOME/hub > ~/.cache/.../hub),
    so it looks wherever the *running user* (root under systemd) actually downloaded.
    """
    hub = (os.environ.get("HF_HUB_CACHE")
           or os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub"))
    base = Path(hub) / ("models--" + name.replace("/", "--")) / "snapshots"
    if not base.is_dir():
        return None
    # Pick whichever snapshot actually has a config.json (newest first).
    for snap in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (snap / "config.json").is_file():
            return snap
    return None


def footprint(m):
    """Derive (weights_bytes, kv_bytes_per_token) from the HF cache + config.json.

    Returns None if the model isn't cached yet (e.g. a cold first run) — callers then
    skip the fit check rather than block a download.
    """
    snap = hf_snapshot(m["name"])
    if snap is None:
        return None
    # Weights: sum the actual on-disk shard sizes (works for bf16 and quantized alike).
    weights = sum(f.stat().st_size for f in snap.glob("*.safetensors"))
    if weights == 0:
        weights = sum(f.stat().st_size for f in snap.glob("*.bin"))  # legacy
    if weights == 0:
        return None

    cfg = json.loads((snap / "config.json").read_text())
    # Multimodal models (e.g. Qwen3.5) nest the LM params under text_config;
    # merge them up so the KV math below sees them.
    cfg = {**cfg, **cfg.get("text_config", {})}
    layers = cfg.get("num_hidden_layers")
    kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")
    head_dim = cfg.get("head_dim")
    if head_dim is None and cfg.get("hidden_size") and cfg.get("num_attention_heads"):
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    if not (layers and kv_heads and head_dim):
        return None

    # KV element size: model dtype, unless the KV cache is pinned to fp8 in params.
    params = m.get("params", "")
    if "fp8" in params and "kv-cache-dtype" in params:
        kv_elem = 1
    else:
        kv_elem = DTYPE_BYTES.get(str(cfg.get("torch_dtype", "")).lower(), 2)
    # 2 = K and V.
    kv_per_token = 2 * layers * kv_heads * head_dim * kv_elem
    return weights, kv_per_token


# ── Live GPU state ───────────────────────────────────────────────────────────────
def gpu_mem():
    """(total_bytes, free_bytes) for GPU 0, or None if nvidia-smi is unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        total, free = (int(x) * MiB for x in out.split(","))
        return total, free
    except Exception:
        return None


# ── Fit math (shared by the check and --fit) ─────────────────────────────────────
def overhead_bytes(defaults, m):
    return float(m.get("overhead_gb", defaults.get("overhead_gb", DEFAULT_OVERHEAD_GB))) * GiB


def predict_tokens(claim_bytes, weights, kv_per_token, overhead):
    """KV tokens deliverable if vLLM gets `claim_bytes` of the card."""
    kv = claim_bytes - weights - overhead
    return max(0, int(kv // kv_per_token))


def required_tokens(m):
    return (m.get("maxlen") or 0) * max(1, m.get("concurrency", 1))


def check_fit(defaults, m):
    """Return (verdict, message). verdict ∈ {'ok','skip','fail'}.

    'skip' = couldn't derive footprint or read the GPU; proceed without blocking.
    """
    fp = footprint(m)
    if fp is None:
        return "skip", f"{m['slug']}: footprint not derivable (not cached yet?) — skipping fit check"
    weights, kv_per_token = fp
    gm = gpu_mem()
    if gm is None:
        return "skip", f"{m['slug']}: nvidia-smi unavailable — skipping fit check"
    total, free = gm

    need_tok = required_tokens(m)
    if not need_tok:
        return "ok", f"{m['slug']}: no maxlen set — nothing to validate"

    overhead = overhead_bytes(defaults, m)
    claim = m["memory"] * total
    detail = (f"{m['slug']}: weights {human(weights)}, KV {kv_per_token/1024:.0f} KB/tok, "
              f"overhead {human(overhead)}; free {human(free)} of {human(total)}")

    # 1) Can vLLM even reserve memory×total given what's already resident?
    if claim > free:
        return "fail", (f"{detail}\n  memory={m['memory']} wants {human(claim)} but only "
                        f"{human(free)} free — vLLM would OOM on load. Lower memory, or free the card.")
    # 2) Does that reservation hold the requested context? (tolerance: only flag a
    #    gross shortfall, not approximation noise near the boundary.)
    got = predict_tokens(claim, weights, kv_per_token, overhead)
    if got < need_tok * (1 - FIT_TOLERANCE):
        return "fail", (f"{detail}\n  memory={m['memory']} → ~{got:,} KV tokens, but need "
                        f"{need_tok:,} (maxlen). Raise memory, add --kv-cache-dtype fp8, "
                        f"lower maxlen, or free the card.  (--fit to auto-size)")
    return "ok", f"{detail}\n  memory={m['memory']} → ~{got:,} KV tokens ≥ {need_tok:,} needed ✓"


def auto_fit(defaults, m):
    """Compute a --gpu-memory-utilization that fits free VRAM and the target context.

    Returns (memory_fraction, message) or (None, failure_message) if it can't fit even
    when claiming all free VRAM.
    """
    fp = footprint(m)
    gm = gpu_mem()
    if fp is None or gm is None:
        return None, f"{m['slug']}: --fit needs footprint + nvidia-smi; one is unavailable"
    weights, kv_per_token = fp
    total, free = gm
    overhead = overhead_bytes(defaults, m)
    need_tok = required_tokens(m)

    # Bytes needed to hold weights + overhead + the target KV, plus a little headroom.
    need_bytes = weights + overhead + need_tok * kv_per_token
    need_bytes *= 1.03
    if need_bytes > free:
        return None, (f"{m['slug']}: cannot fit even auto-sized — need {human(need_bytes)} for "
                      f"{need_tok:,} tokens but only {human(free)} free. "
                      f"Use fp8 KV, lower maxlen, or free the card.")
    frac = round(need_bytes / total, 3)
    return frac, (f"{m['slug']}: --fit sized memory {m['memory']} → {frac} "
                  f"({human(need_bytes)} for {need_tok:,} tokens, {human(free)} free)")


# ── List ─────────────────────────────────────────────────────────────────────────
def fmt_list(defaults, models):
    hdr = f"{'SLUG':<7} {'NAME':<40} {'ON':<3} {'MEM':<5} {'TASK':<9} {'MAXLEN':<8} {'WEIGHTS':<9} {'PRED-KV-TOK'}"
    print(hdr)
    for m in models.values():
        fp = footprint(m)
        w = human(fp[0]) if fp else (str(m.get("weights", "?")))
        pred = "?"
        if fp and m.get("memory") and (gm := gpu_mem()):
            pred = f"~{predict_tokens(m['memory'] * gm[0], fp[0], fp[1], overhead_bytes(defaults, m)):,}"
        print(
            f"{m['slug']:<7} {m['name']:<40} {('yes' if m.get('enable') else 'no'):<3} "
            f"{m['memory']:<5} {m['task']:<9} {str(m.get('maxlen') or 'default'):<8} {w:<9} {pred}"
        )
    print("\nPRED-KV-TOK = KV tokens the configured memory yields given current free VRAM.")
    print("A unit refuses to start if its prediction is below maxlen (run with --fit to auto-size).")


# ── Launch ───────────────────────────────────────────────────────────────────────
def launch(defaults, m, fit=False):
    project = str((CONFIG.parent / defaults.get("project", ".")).resolve())
    bind_host = os.environ.get("BIND_HOST", defaults.get("bind_host", "0.0.0.0"))
    # Port is owned by infra.toml, which bakes Environment=PORT=<assigned> into the
    # systemd unit. It is the single source — not duplicated in models.toml. For a
    # manual run outside systemd, pass it yourself: `PORT=8001 ./serve.py chat`.
    port = os.environ.get("PORT")
    if not port:
        sys.exit(f"no PORT in env for '{m['slug']}' — infra sets it from infra.toml; "
                 f"for a manual run use `PORT=<n> {sys.argv[0]} {m['slug']}`")

    memory = m["memory"]
    if fit:
        # --fit: auto-size the budget instead of failing on a bad config (P1).
        frac, msg = auto_fit(defaults, m)
        print(f"[fit] {msg}", file=sys.stderr)
        if frac is None:
            sys.exit(2)
        memory = frac
    else:
        # Default: fail fast if the configured budget won't deliver the context (P0).
        verdict, msg = check_fit(defaults, m)
        print(f"[fit] {msg}", file=sys.stderr)
        if verdict == "fail":
            sys.exit(2)

    cmd = [
        UV, "run", "--project", project, "vllm", "serve", m["name"],
        "--served-model-name", m["slug"],
        "--host", bind_host,
        "--port", str(port),
        "--gpu-memory-utilization", str(memory),
    ]
    # vLLM 0.22 dropped --task; embedding and scoring (reranker) models are
    # selected with --runner pooling.
    if m.get("task") in ("embed", "score"):
        cmd += ["--runner", "pooling"]
    if m.get("maxlen"):
        cmd += ["--max-model-len", str(m["maxlen"])]
    if m.get("params"):
        cmd += m["params"].split()

    print(f"[+] '{m['slug']}' → {m['name']}  on {bind_host}:{port}  (gpu-mem {memory})")
    print(f"[+] {' '.join(cmd)}")
    # exec (replace this process) so Ctrl-C / systemd signals reach vllm directly.
    os.execvp(cmd[0], cmd)


def main(argv):
    defaults, models = load()
    args = argv[1:]
    fit = "--fit" in args
    positional = [a for a in args if not a.startswith("-")]
    flag = next((a for a in args if a in ("--list", "-l")), None)

    if flag:
        fmt_list(defaults, models)
        return
    slug = positional[0] if positional else ""
    if slug not in models:
        avail = ", ".join(models)
        sys.exit(f"unknown model slug: '{slug}'\navailable: {avail}   (try '{argv[0]} --list')")

    launch(defaults, models[slug], fit=fit)


if __name__ == "__main__":
    main(sys.argv)
