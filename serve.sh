#!/usr/bin/env bash
# vllm/serve.sh — launch a single VRAM-capped vLLM server for one model.
#
# Usage:
#   ./serve.sh <key>     start the named model in the foreground
#   ./serve.sh --list    show the model table + total VRAM if all ran at once
#
# ── VRAM ──────────────────────────────────────────────────────────────────────
# One RTX 5090 = 32 GB. Each `vllm serve` process reserves a *fraction* of the
# card up front (--gpu-memory-utilization, default 0.9). Two models can share the
# GPU only if their fractions sum to < ~0.95. A 32B model needs most of the card,
# so run it ALONE — see the note on the coder entry below.
#
# ── Binding ─────────────────────────────────────────────────────────────────────
# vLLM runs on the HOST (it owns the GPU). The LiteLLM *container* reaches it via
# host.docker.internal, which resolves to the docker bridge gateway — NOT
# 127.0.0.1 — so vLLM must listen where the bridge can hit it (0.0.0.0). That's
# why this differs from the loopback-only rule for container services. Keep these
# ports off the LAN with ufw (e.g. `sudo ufw deny 8001:8003/tcp`); they're
# internal to the box and only LiteLLM should talk to them.

set -euo pipefail

# The uv project that has vllm installed (`uv add vllm` was run here).
# Absolute defaults, not $HOME-relative: systemd runs the unit with a stripped env, so $HOME
# and PATH can't be relied on. uv lives under ~/.local/bin which is NOT on systemd's PATH, so
# call it by absolute path. Override any of these via the environment.
LLM_PROJECT="${LLM_PROJECT:-/home/beans/code/llm-0}"
UV="${UV:-/home/beans/.local/bin/uv}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"

# ── Model table ─────────────────────────────────────────────────────────────────
# Edit these as you swap in real models. KEYS sets display/order.
# MAXLEN caps context length — it directly drives KV-cache VRAM, so lower it if a
# model won't fit. Leave it empty to use the model's own default.
KEYS=(chat embed)   # coder disabled for now — only chat+embed needed (they co-fit; coder runs solo)

# infra renders the systemd unit with `Environment=PORT=<assigned>`. Capture it before the
# PORT associative array below shadows the name, and unset so `declare -A PORT` starts clean.
# When set (i.e. launched by infra/systemd), it overrides the per-key port in the table below;
# when unset (manual `./serve.sh <key>`), the table value is used.
INFRA_PORT="${PORT:-}"
unset PORT
declare -A REPO PORT MEM TASK MAXLEN EXTRA

REPO[chat]="Qwen/Qwen3.5-4B"
# 0.55 ≈ 18 GB: ~8.6 GB weights leave ~8 GB KV cache for concurrency. Co-runs with embed
# (~10 GB) → ~28 GB total. Bumped from 0.35, which left ~0 for KV once embed was resident.
PORT[chat]=8001;  MEM[chat]=0.55; TASK[chat]="generate"; MAXLEN[chat]=32768
# --enable-auto-tool-choice + --tool-call-parser: required for OpenAI-style tool calls
# (the agent binds tools and sends tool_choice="auto"). Qwen3.5 emits XML-style
# <tool_call><function=NAME>...</function></tool_call>, so use the qwen3_xml parser
# (NOT hermes, which expects JSON). qwen3_coder is the coder-model variant.
EXTRA[chat]="--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml"   # Qwen3.5 needs vLLM from main; see README

REPO[embed]="Qwen/Qwen3-Embedding-4B"
# 0.30 ≈ 9.8 GB: Qwen3-Embedding-4B is a full 4B model (~8 GB weights), so the old 0.15 (~4.9 GB)
# couldn't even hold the weights. Co-runs with chat (0.35); raise further only if it still OOMs.
# MAXLEN caps context → KV-cache size. Empty = the model's 40960 default, which needs ~5.6 GB
# of KV cache and won't fit alongside the 7.5 GB of weights in this budget. Embeddings work on
# short chunks, so 4096 is plenty and keeps KV cache to ~0.6 GB.
PORT[embed]=8002; MEM[embed]=0.30; TASK[embed]="embed";  MAXLEN[embed]=4096
EXTRA[embed]=""

# --- coder: disabled for now (uncomment + re-add `coder` to KEYS to enable; run it SOLO) ---
# AWQ 4-bit build (~19 GB) so it fits on one 32 GB card; run it solo (MEM 0.90).
# Repo id: HF downloads/caches it on first run (~19 GB into ~/.cache/huggingface).
# REPO[coder]="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
# PORT[coder]=8003; MEM[coder]=0.90; TASK[coder]="generate"; MAXLEN[coder]=16384
# EXTRA[coder]="--quantization awq"

# ── List mode ─────────────────────────────────────────────────────────────────
list() {
  printf '%-7s %-42s %-6s %-5s %-9s %-8s\n' KEY REPO PORT MEM TASK MAXLEN
  local sum=0
  for k in "${KEYS[@]}"; do
    printf '%-7s %-42s %-6s %-5s %-9s %-8s\n' \
      "$k" "${REPO[$k]}" "${PORT[$k]}" "${MEM[$k]}" "${TASK[$k]}" "${MAXLEN[$k]:-default}"
    sum=$(awk "BEGIN{print $sum + ${MEM[$k]}}")
  done
  echo
  echo "sum of all mem fractions: $sum"
  echo "  < ~0.95  → those models can co-run on the card"
  echo "  > 1.0    → run them one at a time (Ctrl-C one before starting the next)"
}

# ── Launch ──────────────────────────────────────────────────────────────────────
main() {
  local key="${1:-}"
  if [[ "$key" == "--list" || "$key" == "-l" ]]; then list; exit 0; fi
  if [[ -z "$key" || -z "${REPO[$key]:-}" ]]; then
    echo "unknown model key: '${key:-}'" >&2
    echo "available: ${KEYS[*]}   (try '$0 --list')" >&2
    exit 2
  fi

  # infra's Environment=PORT (captured as INFRA_PORT) wins when present; else the table value.
  local port="${INFRA_PORT:-${PORT[$key]}}"

  local cmd=( "$UV" run --project "$LLM_PROJECT" vllm serve "${REPO[$key]}"
              --served-model-name "$key"
              --host "$BIND_HOST"
              --port "$port"
              --gpu-memory-utilization "${MEM[$key]}" )
  # vLLM 0.22 dropped --task; an embedding model is now selected with --runner pooling.
  [[ "${TASK[$key]}" == "embed" ]] && cmd+=( --runner pooling )
  [[ -n "${MAXLEN[$key]}" ]]       && cmd+=( --max-model-len "${MAXLEN[$key]}" )
  # Intentional word-splitting so EXTRA can hold multiple flags.
  # shellcheck disable=SC2206
  [[ -n "${EXTRA[$key]}" ]]        && cmd+=( ${EXTRA[$key]} )

  echo "[+] '$key' → ${REPO[$key]}  on ${BIND_HOST}:${port}  (gpu-mem ${MEM[$key]})"
  echo "[+] ${cmd[*]}"
  exec "${cmd[@]}"   # replace shell so Ctrl-C / systemd signals reach vLLM directly
}

main "$@"
