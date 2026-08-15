#!/usr/bin/env python3
"""waker.py — sleep-aware reverse proxy in front of the vLLM backends.

LiteLLM points every vLLM model at this one port with a slug prefix
(api_base http://localhost:$PORT/<slug>/v1). On each request the waker makes
sure that backend is awake — waking it, and first putting idle backends to
sleep if the card can't hold both — then streams the request through.
Clients just see a slow first token after a swap, never an error.

Config is read from the files that already own it (nothing is duplicated).
Each model lives in models/<slug>/ — the folder name is the slug:
  models/*/infra.toml  which vLLM units exist and their ports ("serve.py <slug>")
  models/*/model.toml  memory fraction, plus the waker knobs:
                 sleep_level  1 = weights→CPU RAM (~1s wake, costs RAM)
                              2 = discard weights (~seconds wake from NVMe, ~0 RAM)
                 sleep_ttl    seconds idle before the reaper parks it (0/absent =
                              stay awake until something else needs the room)
  models/defaults.toml  may set:
                 waker_budget  max sum of awake memory fractions (default 0.92)
                 waker_grace   seconds since last use before a model may be
                               evicted for someone else (anti-thrash, default 60)
  models/*/container.toml  non-vLLM GPU tenants (docling) that the waker
               docker-stops/-starts instead of sleeping/waking. Each gets a
               dedicated passthrough listener (`listen`) so its public port never
               changes, forwarding to the container's real port (`upstream`).

Backends that don't support sleeping (unit not restarted with
--enable-sleep-mode yet, or plain down) are detected and passed through
untouched, so this can front the stack before every unit is migrated.

Endpoints (loopback-only, same trust model as the vLLM ports):
  /<slug>/...           proxied to the backend        (main port)
  /waker/status         state of every backend
  /waker/sleep/<slug>   force-park now (e.g. before manually booting coder)
  /waker/wake/<slug>    force-wake now
  /...                  passthrough to that listener's container (extra ports)

vLLM sleep endpoints exist only when the unit runs with VLLM_SERVER_DEV_MODE=1
(each model's infra.toml bakes it in) and --enable-sleep-mode (model.toml params).
"""

import asyncio
import json
import os
import sys
import time
import tomllib
from pathlib import Path

import aiohttp
from aiohttp import web

HERE = Path(__file__).resolve().parent
HEALTH_TIMEOUT = 300     # max seconds to wait for a woken backend to serve
PROBE_INTERVAL = 30      # seconds between background state reconciliations
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}

AWAKE, ASLEEP, WAKING, DOWN = "awake", "asleep", "waking", "down"


def load_config():
    models_dir = HERE / "models"
    with (models_dir / "defaults.toml").open("rb") as f:
        defaults = tomllib.load(f)

    # slug -> port, from the systemd services that launch "serve.py <slug>"
    # (each model's own models/<slug>/infra.toml, plus the root infra.toml)
    ports = {}
    for path in [HERE / "infra.toml", *sorted(models_dir.glob("*/infra.toml"))]:
        with path.open("rb") as f:
            idata = tomllib.load(f)
        for svc in idata.get("service", []):
            exec_start = svc.get("exec_start", "")
            if svc.get("kind") == "systemd" and "serve.py" in exec_start:
                ports[exec_start.split()[-1]] = svc["port"]

    backends = {}
    for path in sorted(models_dir.glob("*/model.toml")):
        slug = path.parent.name
        if slug not in ports:
            continue          # disabled/bench model with no registered unit
        with path.open("rb") as f:
            m = tomllib.load(f)
        backends[slug] = Backend(
            slug=slug,
            port=ports[slug],
            memory=float(m["memory"]),
            sleep_level=int(m.get("sleep_level", 2)),
            sleep_ttl=int(m.get("sleep_ttl", 0)),
        )
    for path in sorted(models_dir.glob("*/container.toml")):
        slug = path.parent.name
        with path.open("rb") as f:
            c = tomllib.load(f)
        backends[slug] = Backend(
            slug=slug,
            port=int(c["upstream"]),
            memory=float(c["memory"]),
            sleep_level=0,
            sleep_ttl=int(c.get("sleep_ttl", 0)),
            kind="docker",
            container=c.get("container", slug),
            listen=int(c["listen"]),
            grace=float(c["grace"]) if "grace" in c else None,
        )
    budget = float(defaults.get("waker_budget", 0.92))
    grace = float(defaults.get("waker_grace", 60))
    return backends, budget, grace


class Backend:
    def __init__(self, slug, port, memory, sleep_level, sleep_ttl,
                 kind="vllm", container=None, listen=None, grace=None):
        self.slug = slug
        self.port = port
        self.memory = memory
        self.sleep_level = sleep_level
        self.sleep_ttl = sleep_ttl
        self.kind = kind             # "vllm" (sleep/wake) | "docker" (stop/start)
        self.container = container   # docker container name (kind == "docker")
        self.listen = listen         # extra passthrough listener port (kind == "docker")
        self.grace = grace           # per-backend eviction grace override (None = global)
        self.state = DOWN
        self.sleep_capable = kind == "docker"
        self.in_flight = 0
        self.last_used = time.monotonic()
        self.woke = asyncio.Event()   # set whenever state leaves WAKING

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def __repr__(self):
        return f"<{self.slug} {self.state} mem={self.memory} inflight={self.in_flight}>"


async def docker(*args):
    """Run a docker CLI command; returns (rc, output). Never raises."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=90)
        return proc.returncode, out.decode(errors="replace").strip()
    except Exception as e:
        return 1, str(e)


class Waker:
    def __init__(self, backends, budget, grace):
        self.backends = backends
        self.budget = budget
        self.grace = grace
        self.lock = asyncio.Lock()   # guards all state transitions + accounting
        self.session = None          # aiohttp.ClientSession, created in run()

    # ── backend control-plane calls (vLLM sleep/wake or docker stop/start) ──
    async def probe(self, b):
        """Reconcile b.state with reality. Never raises."""
        if b.kind == "docker":
            try:
                async with self.session.get(f"{b.base}/health",
                                            timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        if b.state != WAKING:
                            b.state = AWAKE
                        return
            except Exception:
                pass
            if b.state == WAKING:
                return
            rc, out = await docker("inspect", "-f", "{{.State.Running}}", b.container)
            # Stopped (or running-but-booting) container = wakeable; missing = down.
            b.state = ASLEEP if rc == 0 else DOWN
            return
        try:
            async with self.session.get(f"{b.base}/is_sleeping",
                                        timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    body = await r.json()
                    b.sleep_capable = True
                    if b.state != WAKING:
                        b.state = ASLEEP if body.get("is_sleeping") else AWAKE
                    return
        except Exception:
            pass
        # No /is_sleeping: either no dev-mode/sleep support, or the unit is down.
        try:
            async with self.session.get(f"{b.base}/health",
                                        timeout=aiohttp.ClientTimeout(total=5)) as r:
                b.sleep_capable = False
                if b.state != WAKING:
                    b.state = AWAKE if r.status == 200 else DOWN
        except Exception:
            if b.state != WAKING:
                b.state = DOWN

    async def do_sleep(self, b):
        if b.kind == "docker":
            log(f"stopping container {b.container}")
            rc, out = await docker("stop", "-t", "30", b.container)
            if rc != 0:
                raise RuntimeError(f"docker stop {b.container}: {out}")
            b.state = ASLEEP
            return
        log(f"sleeping {b.slug} (level {b.sleep_level})")
        async with self.session.post(f"{b.base}/sleep", params={"level": str(b.sleep_level)},
                                     timeout=aiohttp.ClientTimeout(total=120)) as r:
            if r.status != 200:
                raise RuntimeError(f"sleep {b.slug} -> HTTP {r.status}")
        b.state = ASLEEP

    async def do_wake(self, b):
        if b.kind == "docker":
            log(f"starting container {b.container}")
            rc, out = await docker("start", b.container)
            if rc != 0:
                raise RuntimeError(f"docker start {b.container}: {out}")
        else:
            log(f"waking {b.slug}")
            async with self.session.post(f"{b.base}/wake_up",
                                         timeout=aiohttp.ClientTimeout(total=180)) as r:
                if r.status != 200:
                    raise RuntimeError(f"wake_up {b.slug} -> HTTP {r.status}")
            if b.sleep_level == 2:
                # Level-2 sleep DISCARDS the weights; /wake_up alone re-allocates the
                # memory but leaves it uninitialized — the model serves garbage
                # (verified live on vLLM 0.22.1). reload_weights re-reads from disk.
                log(f"reloading weights for {b.slug} (level-2 wake)")
                async with self.session.post(f"{b.base}/collective_rpc",
                                             json={"method": "reload_weights"},
                                             timeout=aiohttp.ClientTimeout(total=600)) as r:
                    if r.status != 200:
                        raise RuntimeError(f"reload_weights {b.slug} -> HTTP {r.status}")
        deadline = time.monotonic() + HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            try:
                async with self.session.get(f"{b.base}/health",
                                            timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(2)
        raise RuntimeError(f"{b.slug} not healthy {HEALTH_TIMEOUT}s after wake")

    # ── orchestration ────────────────────────────────────────────────────────
    def awake_set(self):
        return [x for x in self.backends.values() if x.state in (AWAKE, WAKING)]

    async def ensure_awake(self, b):
        """Return with b awake and its in_flight incremented, or raise web errors."""
        while True:
            wait_for = None
            async with self.lock:
                if b.state == AWAKE:
                    b.in_flight += 1
                    b.last_used = time.monotonic()
                    return
                if b.state == WAKING:
                    b.woke.clear()
                    wait_for = b.woke
                elif b.state == DOWN:
                    await self.probe(b)   # maybe it came up since last look
                    if b.state == DOWN:
                        raise web.HTTPBadGateway(text=json.dumps({"error": {
                            "message": f"backend '{b.slug}' is not running "
                                       f"(systemctl start vllm unit first)"}}),
                            content_type="application/json")
                    continue
                elif b.state == ASLEEP:
                    # Make room: evict idle, out-of-grace backends, LRU first.
                    now = time.monotonic()
                    awake = self.awake_set()
                    need = b.memory
                    while sum(x.memory for x in awake) + need > self.budget:
                        victims = [x for x in awake
                                   if x.sleep_capable and x.in_flight == 0
                                   and x.state == AWAKE
                                   and now - x.last_used >= (x.grace if x.grace is not None
                                                             else self.grace)]
                        if not victims:
                            raise web.HTTPServiceUnavailable(
                                text=json.dumps({"error": {
                                    "message": f"GPU busy: cannot fit '{b.slug}' "
                                               f"(awake: {[x.slug for x in awake]})"}}),
                                content_type="application/json",
                                headers={"Retry-After": "30"})
                        victim = min(victims, key=lambda x: x.last_used)
                        try:
                            await self.do_sleep(victim)
                        except Exception as e:
                            log(f"ERROR sleeping {victim.slug}: {e}")
                            victim.sleep_capable = False   # stop considering it
                        awake = self.awake_set()
                    b.state = WAKING
                    b.woke.clear()
            if wait_for is not None:
                try:
                    await asyncio.wait_for(wait_for.wait(), HEALTH_TIMEOUT + 60)
                except asyncio.TimeoutError:
                    raise web.HTTPGatewayTimeout(text="timed out waiting for backend wake")
                continue   # re-check state from the top

            # We own the WAKING transition; do the slow part outside the lock.
            try:
                await self.do_wake(b)
                async with self.lock:
                    b.state = AWAKE
                    b.in_flight += 1
                    b.last_used = time.monotonic()
                    b.woke.set()
                return
            except Exception as e:
                log(f"ERROR waking {b.slug}: {e}")
                async with self.lock:
                    await self.probe(b)
                    if b.state == WAKING:
                        b.state = DOWN
                    b.woke.set()
                raise web.HTTPBadGateway(text=json.dumps({"error": {
                    "message": f"failed to wake '{b.slug}': {e}"}}),
                    content_type="application/json")

    async def release(self, b):
        async with self.lock:
            b.in_flight = max(0, b.in_flight - 1)
            b.last_used = time.monotonic()

    async def reaper(self):
        """Background: reconcile states and park backends idle past their ttl."""
        while True:
            await asyncio.sleep(PROBE_INTERVAL)
            async with self.lock:
                now = time.monotonic()
                for b in self.backends.values():
                    if b.state != WAKING:
                        await self.probe(b)
                    if (b.state == AWAKE and b.sleep_capable and b.sleep_ttl
                            and b.in_flight == 0
                            and now - b.last_used >= b.sleep_ttl):
                        try:
                            await self.do_sleep(b)
                        except Exception as e:
                            log(f"ERROR ttl-sleeping {b.slug}: {e}")

    # ── data plane ───────────────────────────────────────────────────────────
    async def proxy(self, request):
        slug = request.match_info["slug"]
        b = self.backends.get(slug)
        if b is None:
            raise web.HTTPNotFound(text=json.dumps({"error": {
                "message": f"unknown backend '{slug}' "
                           f"(known: {sorted(self.backends)})"}}),
                content_type="application/json")
        return await self.forward(request, b, request.match_info["tail"])

    def passthrough(self, b):
        """Handler for a dedicated listener: whole path goes to one backend."""
        async def handler(request):
            return await self.forward(request, b, request.match_info["tail"])
        return handler

    async def forward(self, request, b, tail):
        await self.ensure_awake(b)
        try:
            url = f"{b.base}/{tail}"
            headers = {k: v for k, v in request.headers.items()
                       if k.lower() not in HOP_HEADERS}
            body = await request.read()
            async with self.session.request(
                    request.method, url, headers=headers, data=body,
                    params=request.rel_url.query,
                    timeout=aiohttp.ClientTimeout(total=None, connect=10)) as resp:
                out_headers = {k: v for k, v in resp.headers.items()
                               if k.lower() not in HOP_HEADERS}
                out = web.StreamResponse(status=resp.status, headers=out_headers)
                await out.prepare(request)
                async for chunk in resp.content.iter_any():
                    await out.write(chunk)
                await out.write_eof()
                return out
        except aiohttp.ClientError as e:
            raise web.HTTPBadGateway(text=json.dumps({"error": {
                "message": f"backend '{b.slug}' request failed: {e}"}}),
                content_type="application/json")
        finally:
            await self.release(b)

    # ── admin ────────────────────────────────────────────────────────────────
    async def status(self, request):
        async with self.lock:
            for b in self.backends.values():
                if b.state != WAKING:
                    await self.probe(b)
            data = {b.slug: {
                "state": b.state, "port": b.port, "memory": b.memory,
                "sleep_capable": b.sleep_capable, "sleep_level": b.sleep_level,
                "sleep_ttl": b.sleep_ttl, "in_flight": b.in_flight,
                "idle_s": round(time.monotonic() - b.last_used, 1),
            } for b in self.backends.values()}
            data["_budget"] = self.budget
            data["_awake_sum"] = round(sum(x.memory for x in self.awake_set()), 3)
        return web.json_response(data)

    async def admin_sleep(self, request):
        b = self.backends.get(request.match_info["slug"])
        if b is None:
            raise web.HTTPNotFound()
        async with self.lock:
            if b.in_flight:
                raise web.HTTPConflict(text=f"{b.slug} has {b.in_flight} requests in flight")
            await self.do_sleep(b)
        return web.json_response({"slug": b.slug, "state": b.state})

    async def admin_wake(self, request):
        b = self.backends.get(request.match_info["slug"])
        if b is None:
            raise web.HTTPNotFound()
        await self.ensure_awake(b)
        await self.release(b)
        return web.json_response({"slug": b.slug, "state": b.state})


def log(msg):
    print(f"[waker] {msg}", flush=True)


async def run(port):
    backends, budget, grace = load_config()
    if not backends:
        sys.exit("no vLLM backends found under models/*/")
    w = Waker(backends, budget, grace)
    w.session = aiohttp.ClientSession()
    for b in backends.values():
        await w.probe(b)
    log(f"budget={budget} grace={grace}s backends: " +
        ", ".join(f"{b.slug}:{b.port}={b.state}{'' if b.sleep_capable else ' (no-sleep)'}"
                  for b in backends.values()))

    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_get("/waker/status", w.status)
    app.router.add_post("/waker/sleep/{slug}", w.admin_sleep)
    app.router.add_post("/waker/wake/{slug}", w.admin_wake)
    app.router.add_route("*", "/{slug}/{tail:.*}", w.proxy)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    log(f"listening on 127.0.0.1:{port}")

    # Dedicated passthrough listeners: one per managed container, so its public
    # port stays stable (Caddy/clients keep talking to e.g. :8007 for docling).
    for b in backends.values():
        if b.kind == "docker" and b.listen:
            papp = web.Application(client_max_size=256 * 1024 * 1024)
            papp.router.add_route("*", "/{tail:.*}", w.passthrough(b))
            prunner = web.AppRunner(papp)
            await prunner.setup()
            await web.TCPSite(prunner, "127.0.0.1", b.listen).start()
            log(f"passthrough 127.0.0.1:{b.listen} -> {b.slug} (:{b.port})")

    asyncio.create_task(w.reaper())
    await asyncio.Event().wait()   # run forever


if __name__ == "__main__":
    port = os.environ.get("PORT")
    if not port:
        sys.exit("no PORT in env — infra sets it from infra.toml; "
                 "for a manual run use `PORT=8008 ./waker.py`")
    asyncio.run(run(int(port)))
