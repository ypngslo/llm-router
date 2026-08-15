#!/usr/bin/env python3
"""serve.py — launch a single VRAM-capped vLLM server for one model.

Each model lives in models/<slug>/model.toml (the folder name is the slug);
shared knobs in models/defaults.toml. This file just reads them and execs vllm.

Usage:
  ./src/serve.py <slug>          start the named model in the foreground
  ./src/serve.py <slug> --fit    auto-size the budget to currently-free VRAM (override config)
  ./src/serve.py --list          show the model table + predicted fit

Before launching, serve.py derives the model's footprint (weights from the HF cache,
KV bytes/token from config.json), reads how much VRAM is free *right now*, and checks
whether the configured `memory` fraction will actually deliver `maxlen` tokens of KV.

  default        → fail fast: if it won't fit, refuse to start with a clear reason.
  --fit          → instead of failing, compute a budget that fits and launch with that.

VRAM: each `vllm serve` reserves `memory` (a fraction of the card) up front. Models
co-run only if their reservations fit alongside what's already resident.
"""

import os
import re
import signal
import sys
import json
import time
import shutil
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent   # this file lives in src/
MODELS_DIR = REPO / "models"
# P2 (serve-sizing-plan.md): measured per-run stats, written by a detached recorder
# forked at launch. Gitignored — values are gpu/vllm-version/config specific.
RUNSTATS = REPO / ".runstats.json"
# uv lives under ~/.local/bin which is NOT on systemd's stripped PATH, so call it
# by absolute path. Override via the environment.
UV = os.environ.get("UV", "/home/beans/.local/bin/uv")

GiB = 1024**3
MiB = 1024**2
# Bytes vLLM reserves beyond weights+KV (activations, CUDA graphs, profiling). NOT
# derivable from config and highly model-dependent — ~8 GiB on chat (generate, 32k)
# vs ~1.4 GiB on embed (pooling). So it's set PER-MODEL via `overhead_gb` in
# each model.toml (bootstrap values measured by hand; P2 replaces them with
# recorded run-stats). This default is only a fallback for a model that sets none.
DEFAULT_OVERHEAD_GB = 3.0
# Fail only when the shortfall is gross. The footprint math is approximate (vLLM's
# real reservation can drift from memory×total by the CUDA context), so a model that
# lands within this fraction of its target is treated as fitting — we catch chat's
# 6k-vs-32k starvation, not embed's 4.0k-vs-4.1k rounding.
FIT_TOLERANCE = 0.10
DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float": 4, "half": 2}


def load():
    """models/defaults.toml + every models/*/model.toml — the folder name is the slug."""
    with (MODELS_DIR / "defaults.toml").open("rb") as f:
        defaults = tomllib.load(f)
    models = {}
    for path in sorted(MODELS_DIR.glob("*/model.toml")):
        with path.open("rb") as f:
            m = tomllib.load(f)
        m["slug"] = path.parent.name
        models[m["slug"]] = m
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
    # Weights: sum the actual on-disk shard sizes (works for bf16 and pre-quantized
    # checkpoints alike — for those, disk ≈ VRAM).
    weights = sum(f.stat().st_size for f in snap.glob("*.safetensors"))
    if weights == 0:
        weights = sum(f.stat().st_size for f in snap.glob("*.bin"))  # legacy
    if weights == 0:
        return None
    # ONLINE quantization (--quantization fp8 on a bf16 checkpoint) shrinks weights
    # at load, so disk size overestimates VRAM. Factor measured 2026-07-08:
    # Qwen3.5-9B, 18.0 GB on disk → 10.81 GiB in VRAM (embeddings/norms stay bf16).
    if "--quantization fp8" in m.get("params", ""):
        weights = int(weights * 0.60)

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


# ── Run-stats: measured capacity from real runs (P2) ─────────────────────────────
def load_runstats():
    try:
        return json.loads(RUNSTATS.read_text())
    except Exception:
        return {}


def measured_stats(m):
    """The recorded run for this slug, or None if absent/stale (model or GPU changed)."""
    rs = load_runstats().get(m["slug"])
    if not rs or rs.get("name") != m["name"] or not rs.get("kv_tokens"):
        return None
    g = gpu_name()
    if g and rs.get("gpu") and rs["gpu"] != g:
        return None
    return rs


def unit_for_slug(slug):
    """systemd unit name for a slug, from the root infra.toml + every model's own
    models/*/infra.toml (exec_start ends in 'serve.py <slug>')."""
    for path in [REPO / "infra.toml", *sorted(MODELS_DIR.glob("*/infra.toml"))]:
        try:
            with path.open("rb") as f:
                services = tomllib.load(f).get("service", [])
        except Exception:
            continue
        for svc in services:
            if svc.get("kind") == "systemd" and svc.get("exec_start", "").endswith(f"serve.py {slug}"):
                return svc["name"]
    return None


def watchdog_kill(m, launcher_pid):
    """Last-resort recovery for a launch that never became healthy: kill the unit's main
    process so systemd restarts it.

    vLLM can wedge silently at startup (observed: a boot-time hang before it ever binds
    its port or creates a CUDA context). Because serve.py *exec*s into the launcher, the
    unit's MainPID stays alive, systemd reports a healthy-looking `active`, and
    Restart=on-failure never fires — the model is simply gone until a human restarts it.
    The recorder already knows the launch failed (health deadline expired), so it doubles
    as the watchdog: it holds the launcher's pid (exec preserves the pid, so it IS the
    unit's MainPID) and kills it, handing recovery to systemd's normal RestartSec/start-
    limit machinery.

    SIGKILL, not SIGTERM — a trapped TERM can exit 0, which Restart=on-failure ignores.
    Fires only when the pid still belongs to this model's systemd unit (cgroup check), so
    a manual `PORT=… ./serve.py <slug>` run is never killed and a recycled pid can't be
    hit by mistake.
    """
    unit = unit_for_slug(m["slug"])
    if not unit:
        print(f"[watchdog] {m['slug']}: no systemd unit in infra.toml — not intervening", flush=True)
        return
    try:
        cgroup = Path(f"/proc/{launcher_pid}/cgroup").read_text()
    except OSError:
        print(f"[watchdog] {m['slug']}: launch pid {launcher_pid} already gone — nothing to kill", flush=True)
        return
    if f"{unit}.service" not in cgroup:
        print(f"[watchdog] {m['slug']}: pid {launcher_pid} is not in {unit}.service "
              "(manual run or recycled pid) — not intervening", flush=True)
        return
    print(f"[watchdog] {m['slug']}: killing hung launch (pid {launcher_pid}) so systemd "
          f"restarts {unit}", flush=True)
    try:
        os.kill(launcher_pid, signal.SIGKILL)
    except OSError as e:
        print(f"[watchdog] {m['slug']}: kill failed: {e}", flush=True)


def record_runstats(m, port, memory, launched_at, launcher_pid):
    """Runs in a detached child: wait for the server, scrape the journal, write stats.
    If the server never comes up, act as the watchdog instead (see watchdog_kill).

    Never raises past its caller (spawn_recorder wraps it); prints go to the unit's
    journal tagged [runstats].
    """
    import urllib.request

    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            break
        except Exception:
            time.sleep(10)
    else:
        print(f"[runstats] {m['slug']}: never became healthy — not recording", flush=True)
        watchdog_kill(m, launcher_pid)
        return
    time.sleep(15)  # let the allocation settle

    unit = unit_for_slug(m["slug"])
    if not unit:
        print(f"[runstats] {m['slug']}: no systemd unit in infra.toml — not recording", flush=True)
        return
    # --since launch time: only THIS boot's lines (a manual run logs to stdout, not
    # the journal, and must not pick up a previous boot's numbers).
    since = datetime.fromtimestamp(launched_at).strftime("%Y-%m-%d %H:%M:%S")
    try:
        out = subprocess.run(
            ["journalctl", "-u", unit, "--since", since, "--no-pager", "-o", "cat"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except Exception as e:
        print(f"[runstats] {m['slug']}: journalctl failed ({e}) — not recording", flush=True)
        return
    tok = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", out)
    ver = re.findall(r"Initializing a V1 LLM engine \(v([^)]+)\)", out)
    if not tok:
        print(f"[runstats] {m['slug']}: no KV-cache line in journal — not recording", flush=True)
        return
    kv_tokens = int(tok[-1].replace(",", ""))

    entry = {
        "name": m["name"],
        "gpu": gpu_name(),
        "vllm": ver[-1] if ver else None,
        "memory": memory,
        "maxlen": m.get("maxlen"),
        "kv_tokens": kv_tokens,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Formula-implied overhead = claim − weights − tokens×KV/tok. Negative means the
    # per-token formula overestimates (hybrid-attention models) — store null; the
    # measured kv_tokens is the useful number there.
    fp, gm = footprint(m), gpu_mem()
    if fp and gm:
        weights, kv_per_token = fp
        oh = memory * gm[0] - weights - kv_tokens * kv_per_token
        entry["overhead_gb_measured"] = round(oh / GiB, 2) if oh >= 0 else None

    stats = load_runstats()
    stats[m["slug"]] = entry
    tmp = RUNSTATS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stats, indent=2) + "\n")
    os.replace(tmp, RUNSTATS)
    print(f"[runstats] {m['slug']}: recorded {kv_tokens:,} KV tokens "
          f"(memory {memory}, vLLM {entry['vllm']})", flush=True)


def spawn_recorder(m, port, memory):
    """Double-fork a detached recorder so it survives the exec and isn't our child.

    The main process still execs vLLM — the recorder sits beside it, never between
    systemd and vLLM (see serve-sizing-plan.md P2).
    """
    launched_at = time.time()
    # Captured BEFORE the forks: this process execs into the launcher without changing
    # pid, so this is the systemd unit's MainPID — the watchdog's kill target.
    launcher_pid = os.getpid()
    try:
        pid = os.fork()
    except OSError:
        return
    if pid > 0:
        os.waitpid(pid, 0)   # reap the intermediate child immediately
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    try:
        record_runstats(m, int(port), float(memory), launched_at, launcher_pid)
    except Exception as e:
        print(f"[runstats] recorder crashed: {e}", flush=True)
    os._exit(0)


# ── Live GPU state ───────────────────────────────────────────────────────────────
def gpu_name():
    """GPU 0's product name, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "--id=0"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:
        return None


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
    # Prefer the overhead measured from a real run (P2) over the configured guess.
    rs = measured_stats(m)
    if rs and rs.get("overhead_gb_measured") is not None:
        return float(rs["overhead_gb_measured"]) * GiB
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
    need_tok = required_tokens(m)
    # P2 fast path: a real run with this exact config already told us the capacity.
    # (Formula-based prediction is badly wrong for hybrid-attention models — chat's
    # formula says ~13k tokens, measured is ~299k — so measured truth wins.)
    rs = measured_stats(m)
    if rs and rs.get("memory") == m["memory"] and need_tok:
        gm = gpu_mem()
        if gm:
            total, free = gm
            claim = m["memory"] * total
            detail = (f"{m['slug']}: measured {rs['kv_tokens']:,} KV tokens at memory={m['memory']} "
                      f"(recorded {rs['recorded_at'][:10]}, vLLM {rs.get('vllm')}); "
                      f"free {human(free)} of {human(total)}")
            if claim > free:
                return "fail", (f"{detail}\n  memory={m['memory']} wants {human(claim)} but only "
                                f"{human(free)} free — vLLM would OOM on load. Lower memory, or free the card.")
            if rs["kv_tokens"] < need_tok * (1 - FIT_TOLERANCE):
                return "fail", (f"{detail}\n  need {need_tok:,} (maxlen) — raise memory, "
                                f"add fp8 KV, or lower maxlen.")
            return "ok", f"{detail}\n  ≥ {need_tok:,} needed ✓"

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
        rs = measured_stats(m)
        if rs and rs.get("memory") == m.get("memory"):
            pred = f"{rs['kv_tokens']:,}*"          # * = measured from a real run
        elif fp and m.get("memory") and (gm := gpu_mem()):
            pred = f"~{predict_tokens(m['memory'] * gm[0], fp[0], fp[1], overhead_bytes(defaults, m)):,}"
        print(
            f"{m['slug']:<7} {m['name']:<40} {('yes' if m.get('enable') else 'no'):<3} "
            f"{m['memory']:<5} {m['task']:<9} {str(m.get('maxlen') or 'default'):<8} {w:<9} {pred}"
        )
    print("\nPRED-KV-TOK = KV tokens the configured memory yields (* = measured from a real")
    print("run via .runstats.json; ~ = formula estimate given current free VRAM).")
    print("A unit refuses to start if its prediction is below maxlen (run with --fit to auto-size).")


# ── Launch ───────────────────────────────────────────────────────────────────────
def launch(defaults, m, fit=False):
    project = str((REPO / defaults.get("project", ".")).resolve())
    bind_host = os.environ.get("BIND_HOST", defaults.get("bind_host", "0.0.0.0"))
    # Port is owned by infra.toml, which bakes Environment=PORT=<assigned> into the
    # systemd unit. It is the single source — not duplicated in model.toml. For a
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
    # P2: detached recorder scrapes this run's real KV capacity into .runstats.json.
    spawn_recorder(m, port, memory)
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
