# Plan: add `user` / `environment` to the systemd run-spec

Goal: let an `infra.toml` systemd service declare `User=` and extra `Environment=` lines, so units
that must run as a non-root user (e.g. vLLM as `beans` for its uv venv + HF cache) stay declarative —
no `/etc` drop-ins, no hand-edits.

## Fields (infra.toml, systemd kind only)

```toml
user = "beans"
environment = { HOME = "/home/beans", HF_HOME = "/home/beans/.cache/huggingface" }
```

Both optional. Rejected for `kind = "compose"` (like `exec_start`/`working_dir` already are).

## Changes

1. **`cli/infra/registry.py`** — add `user: str = ""` and `environment: dict[str,str] = {}` to the
   `Service` dataclass; include both in TOML (de)serialization so `services.d/*.toml` round-trips.

2. **`cli/infra/config.py`** — add `user`, `environment` to the allowed-fields set; validate they're
   systemd-only and well-typed (str / str→str map); carry them into `ServiceSpec`.

3. **`cli/infra/scaffold.py`** — render into `[Service]`:
   - `User=<user>` when set
   - one `Environment=K=V` line per `environment` entry (deterministic key order for byte-stable output)

   Keep existing order: `WorkingDirectory` → `User` → `Environment=PORT` → other `Environment` → `ExecStart` → `Restart`.

4. **`cli/tests/`** — extend scaffold/registry tests: unit renders `User=`/`Environment=`, round-trips,
   and compose rejects both fields.

5. **`docs/infra-config.md`** — document the two fields in the `kind = "systemd"` table.

## Then for vLLM

Add to each `vllm-*` in `llm-router/infra.toml`:

```toml
user = "beans"
environment = { HOME = "/home/beans" }
```

`register` → `deploy` regenerates the units with `User=beans`. No `/etc` edits.

## Out of scope

`serve.sh` hardening (absolute `uv`, absolute `LLM_PROJECT`) is independent and still worth doing as
defense-in-depth, but `User=beans` alone fixes PATH/HOME/venv/cache.
