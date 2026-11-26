#!/usr/bin/env python3
"""Cache world: compare the learned eviction policy against real baselines.

The world generates its request stream from the episode seed, so replaying the
same seeds reproduces the same 60-request trace. That lets us score LRU, LFU
and Belady's offline optimum on exactly the episodes the agent was measured on.

    python3 benchmarks/cache_baselines.py [--policy policies/cache.hpol]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hunch import cli  # noqa: E402

EVAL_SEED = 987654321          # must match EVAL_SEED in runtime/hunch_vm.c
EPISODES = 96
CAP = 4


def lru_hits(req, cap=CAP):
    cache, hits = [], 0
    for k in req:
        if k in cache:
            hits += 1
            cache.remove(k)
            cache.append(k)
        else:
            if len(cache) >= cap:
                cache.pop(0)
            cache.append(k)
    return hits


def lfu_hits(req, cap=CAP):
    cache, freq, hits = [], {}, 0
    for k in req:
        if k in cache:
            hits += 1
            freq[k] = freq.get(k, 0) + 1
        else:
            if len(cache) >= cap:
                victim = min(cache, key=lambda c: (freq.get(c, 0), cache.index(c)))
                cache.remove(victim)
            cache.append(k)
            freq[k] = 1
    return hits


def opt_hits(req, cap=CAP):
    """Belady: evict whatever will not be needed for the longest time."""
    cache, hits = [], 0
    next_use = {}
    nxt = [10 ** 9] * len(req)
    for i in range(len(req) - 1, -1, -1):
        nxt[i] = next_use.get(req[i], 10 ** 9)
        next_use[req[i]] = i
    for i, k in enumerate(req):
        if k in cache:
            hits += 1
            continue
        if len(cache) < cap:
            cache.append(k)
            continue
        best, far = None, -1
        for c in cache:
            j = i + 1
            while j < len(req) and req[j] != c:
                j += 1
            d = j if j < len(req) else 10 ** 9
            if d > far:
                far, best = d, c
        cache.remove(best)
        cache.append(k)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(ROOT / "policies" / "cache.hpol"))
    ap.add_argument("--episodes", type=int, default=EPISODES)
    args = ap.parse_args()

    hbc, _ = cli.compile_file(ROOT / "worlds" / "cache.hunch")
    cli.ensure_vm()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        trace_path = tmp.name

    cmd = [str(cli.VM_BIN), "trace", str(hbc), "--seed", str(EVAL_SEED),
           "--episodes", str(args.episodes), "--out", trace_path]
    if Path(args.policy).exists():
        cmd += ["--policy", args.policy]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return 1

    data = json.loads(Path(trace_path).read_text())
    fields = {f["name"]: (f["base"], f["len"]) for f in data["fields"]}
    base, n = fields["req"]

    learned, lru, lfu, opt = [], [], [], []
    for ep in data["episodes"]:
        req = [int(v) for v in ep["steps"][0]["s"][base:base + n]]
        hits = sum(v for st in ep["steps"] for v in st["r"].values())
        learned.append(hits)
        lru.append(lru_hits(req))
        lfu.append(lfu_hits(req))
        opt.append(opt_hits(req))

    k = len(learned)
    mean = lambda xs: sum(xs) / len(xs)
    rows = [
        ("hunch (trained)", mean(learned)),
        ("LRU", mean(lru)),
        ("LFU", mean(lfu)),
        ("Belady OPT", mean(opt)),
    ]

    print(f"cache eviction, {k} episodes of 60 requests, 4 slots\n")
    print(f"{'policy':<18}{'hits / 60':>10}{'hit rate':>11}")
    for name, m in rows:
        print(f"{name:<18}{m:>10.2f}{m / 60 * 100:>10.1f}%")

    if Path(args.policy).exists():
        best = max(v for _, v in rows)
        gap = best - mean(learned)
        print(f"\nlearned policy is {gap:.2f} hits/episode behind the best row "
              f"({gap / 60 * 100:.1f} points)")

    Path(trace_path).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
