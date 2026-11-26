"""Hunch Studio -- a small HTTP server for the interactive playground.

Standard library only. It shells out to the compiled `build/hunch-vm` for
everything that needs to actually run a world.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import cli

ROOT = cli.ROOT
STUDIO = ROOT / "studio"
EXAMPLES = ROOT / "worlds"
POLICIES = ROOT / "policies"
BENCH = ROOT / "benchmarks" / "results.json"

WORLDS = [
    {
        "id": "dungeon",
        "file": "dungeon.hunch",
        "blurb": "Find the key, dodge the trap, open the exit.",
        "lesson": "A delayed payoff fifteen steps after the decision that caused it.",
        "actions": ["Up", "Down", "Left", "Right"],
        "budget": 400_000,
        "demo_seed": 20,
    },
    {
        "id": "market",
        "file": "market.hunch",
        "blurb": "Quote a two-sided market without blowing up on inventory.",
        "lesson": "Noise is the enemy of learning: two branches see different coin flips unless you hold the randomness fixed.",
        "actions": ["Buy hard", "Buy", "Neutral", "Sell", "Sell hard"],
        "budget": 600_000,
        "demo_seed": 22,
    },
    {
        "id": "duel",
        "file": "duel.hunch",
        "blurb": "Two agents, hidden cards, no human data at all.",
        "lesson": "Self-play: both blanks learn by playing each other, so a bluff is only good if it is also a trap.",
        "actions": ["Check / Fold", "Bet / Call", "Bet big"],
        "budget": 600_000,
        "demo_seed": 12,
    },
    {
        "id": "cache",
        "file": "cache.hunch",
        "blurb": "Teach a four-slot cache how to choose its own eviction policy.",
        "lesson": "A real systems decision with a measurable answer, checked against LRU, LFU and an oracle.",
        "actions": ["Slot 1", "Slot 2", "Slot 3", "Slot 4"],
        "budget": 2_000_000,
        "demo_seed": 18,
    },
]

_by_id = {w["id"]: w for w in WORLDS}

_cache_lock = threading.Lock()
_trace_cache: dict[tuple, dict] = {}


# ---------------------------------------------------------------------------
# Running worlds
# ---------------------------------------------------------------------------

def compile_world(w: dict) -> Path:
    hbc, _ = cli.compile_file(EXAMPLES / w["file"])
    return hbc


def policy_path(w: dict) -> Path | None:
    p = POLICIES / f"{w['id']}.hpol"
    return p if p.exists() else None


def run_trace(w: dict, policy: Path | None, seed: int, forks: bool,
              actions: str | None = None, agent: str | None = None,
              episodes: int = 1) -> dict:
    cli.ensure_vm()
    hbc = compile_world(w)
    pol_mtime = policy.stat().st_mtime if policy and policy.exists() else 0
    key = (w["id"], str(policy), pol_mtime, seed, forks, actions, agent, episodes,
           hbc.stat().st_mtime if hbc.exists() else 0)
    with _cache_lock:
        if key in _trace_cache:
            return _trace_cache[key]

    cmd = [str(cli.VM_BIN), "trace", str(hbc), "--seed", str(seed),
           "--episodes", str(episodes)]
    if policy:
        cmd += ["--policy", str(policy)]
    if forks:
        cmd.append("--forks")
    if agent:
        cmd += ["--agent", agent, "--actions", actions or ""]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "trace failed")
    data = json.loads(proc.stdout)
    with _cache_lock:
        _trace_cache[key] = data
        if len(_trace_cache) > 64:
            _trace_cache.clear()
            _trace_cache[key] = data
    return data


def load_bench() -> dict:
    if not BENCH.exists():
        return {}
    try:
        data = json.loads(BENCH.read_text())
    except json.JSONDecodeError:
        return {}
    return data.get("worlds", {})


def world_payload(w: dict, seed: int) -> dict:
    pol = policy_path(w)
    gen0 = run_trace(w, None, seed, forks=False)
    trained = run_trace(w, pol, seed, forks=True) if pol else None
    bench = load_bench().get(w["id"])
    return {
        "id": w["id"],
        "world": gen0.get("world", w["id"]),
        "blurb": w["blurb"],
        "lesson": w["lesson"],
        "actions": w["actions"],
        "seed": seed,
        "has_policy": pol is not None,
        "gen0": gen0,
        "trained": trained,
        "bench": bench,
        "source": (EXAMPLES / w["file"]).read_text(),
    }


def train_now(w: dict, algo: str, budget: int, seed: int) -> dict:
    cli.ensure_vm()
    hbc = compile_world(w)
    tmp = ROOT / ".hunch-cache" / f"_live_{w['id']}.hpol"
    cmd = [str(cli.VM_BIN), "train", str(hbc), "--algo", algo,
           "--budget", str(budget), "--seed", str(seed),
           "--out", str(tmp), "--json", "--eval-episodes", "64"]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "training failed")
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    data["wall_sec"] = round(time.time() - t0, 2)
    trace = run_trace(w, tmp if tmp.exists() else None, seed, forks=True)
    data["trace"] = trace
    return data


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".txt": "text/plain; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "HunchStudio/0.1"

    def log_message(self, fmt, *args):
        if os.environ.get("HUNCH_QUIET"):
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers ---------------------------------------------------------
    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, msg, status=500):
        self.send_json({"error": str(msg)}, status)

    def send_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- routes ----------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        q = parse_qs(url.query)

        try:
            if path in ("/", "/index.html"):
                return self.send_file(STUDIO / "index.html")
            if path.startswith("/static/"):
                name = path[len("/static/"):]
                if ".." in name or name.startswith("/"):
                    return self.send_error_json("bad path", 400)
                return self.send_file(STUDIO / name)
            if path == "/llms.txt":
                return self.send_file(ROOT / "llms.txt")
            if path == "/api/worlds":
                return self.send_json({
                    "worlds": [{"id": w["id"], "blurb": w["blurb"],
                                "lesson": w["lesson"],
                                "has_policy": policy_path(w) is not None}
                               for w in WORLDS]
                })
            m = re.fullmatch(r"/api/world/([a-z]+)", path)
            if m:
                w = _by_id.get(m.group(1))
                if not w:
                    return self.send_error_json("unknown world", 404)
                seed = int(q.get("seed", [str(w.get("demo_seed", 7))])[0])
                return self.send_json(world_payload(w, seed))
            if path == "/api/health":
                return self.send_json({"ok": True})
            return self.send_error_json("not found", 404)
        except Exception as exc:  # noqa: BLE001 - surface errors to the UI
            return self.send_error_json(exc)

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self.send_error_json("bad json", 400)

        try:
            if url.path == "/api/train":
                w = _by_id.get(body.get("world", ""))
                if not w:
                    return self.send_error_json("unknown world", 404)
                algo = body.get("algo", "fork")
                if algo not in ("reinforce", "ppo", "fork", "random"):
                    return self.send_error_json("unknown algorithm", 400)
                budget = int(body.get("budget") or w["budget"])
                budget = max(20_000, min(budget, 8_000_000))
                seed = int(body.get("seed") or 1)
                return self.send_json(train_now(w, algo, budget, seed))

            if url.path == "/api/play":
                w = _by_id.get(body.get("world", ""))
                if not w:
                    return self.send_error_json("unknown world", 404)
                pol = policy_path(w)
                actions = body.get("actions", "")
                agent = body.get("agent")
                seed = int(body.get("seed") or 7)
                trace = run_trace(w, pol, seed, forks=False, actions=actions, agent=agent)
                return self.send_json(trace)

            return self.send_error_json("not found", 404)
        except Exception as exc:  # noqa: BLE001
            return self.send_error_json(exc)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="hunch studio")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args(argv)

    cli.ensure_vm()
    for w in WORLDS:
        try:
            compile_world(w)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not compile {w['file']}: {exc}", file=sys.stderr)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"hunch studio listening on http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
