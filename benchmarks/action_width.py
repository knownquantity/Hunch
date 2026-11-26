#!/usr/bin/env python3
"""Does the rewind-and-fork advantage grow with the number of actions?

Take one market, hold the economics fixed, and change only the resolution of
the (width, skew) quote grid: 4, 8, 16 and 32 actions. Then train PPO and fork
on each with identical budgets and seeds.

The intuition being tested is that a sampled advantage hands the learner one
action out of A, while a counterfactual gives it all A exactly -- so wider
action spaces should favour forking.

They do not. Forking costs O(A) per forked state, so the budget buys fewer and
fewer forks as A grows. This script records that.

    python3 benchmarks/action_width.py [--seeds 1,2,3,4,5] [--budget 600000]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from hunch import cli  # noqa: E402
import make_market_variants  # noqa: E402

CACHE = ROOT / ".hunch-cache"


def run_jobs(tasks, jobs=4):
    out = {}
    running = []

    def launch(t):
        p = subprocess.Popen(t[3], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        running.append((t, p))

    it = iter(tasks)
    for _ in range(min(jobs, len(tasks))):
        try:
            launch(next(it))
        except StopIteration:
            break

    while running:
        t, p = running.pop(0)
        key, algo, seed, _ = t
        stdout, stderr = p.communicate()
        if p.returncode == 0:
            out.setdefault(key, {}).setdefault(algo, []).append(
                json.loads(stdout.strip().splitlines()[-1])["final_return"])
        else:
            print(f"  ! {key} {algo} {seed}: {stderr.strip()[:200]}", file=sys.stderr)
        try:
            launch(next(it))
        except StopIteration:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--budget", type=int, default=600_000)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "action_width.json"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    make_market_variants.main()
    cli.ensure_vm()

    variants = [(4, "market_A4"), (8, "market_A8"), (16, "market_A16"), (32, "market_A32")]
    tasks = []
    for na, stem in variants:
        hbc, unit = cli.compile_file(CACHE / f"{stem}.hunch")
        assert unit["agent_acts"]["mm"] == na, f"{stem} has {unit['agent_acts']['mm']} actions"
        for algo in ("ppo", "fork"):
            for seed in seeds:
                pol = CACHE / f"aw.{stem}.{algo}.{seed}.hpol"
                cmd = [str(cli.VM_BIN), "train", str(hbc), "--algo", algo,
                       "--budget", str(args.budget), "--seed", str(seed),
                       "--out", str(pol), "--json", "--eval-episodes", "96"]
                tasks.append((stem, algo, seed, cmd))

    raw = run_jobs(tasks)

    print(f"\nmarket maker, identical economics, {args.budget:,} steps, {len(seeds)} seeds\n")
    print(f"{'actions':>8} {'PPO':>9} {'fork':>9} {'gap':>9}")
    result = {"budget": args.budget, "seeds": seeds, "variants": {}}
    for na, stem in variants:
        d = raw.get(stem, {})
        if "ppo" not in d or "fork" not in d:
            continue
        ppo = sum(d["ppo"]) / len(d["ppo"])
        fork = sum(d["fork"]) / len(d["fork"])
        gap = (fork - ppo) / abs(ppo) * 100
        print(f"{na:>8} {ppo:>9.3f} {fork:>9.3f} {gap:>8.1f}%")
        result["variants"][str(na)] = {
            "ppo": round(ppo, 4), "fork": round(fork, 4),
            "gap_pct": round(gap, 2),
            "ppo_runs": [round(x, 4) for x in d["ppo"]],
            "fork_runs": [round(x, 4) for x in d["fork"]],
        }

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
