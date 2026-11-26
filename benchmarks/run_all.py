#!/usr/bin/env python3
"""Run every world under every algorithm on an identical simulation budget.

Usage:
    python3 benchmarks/run_all.py              # full run (a few minutes)
    python3 benchmarks/run_all.py --quick      # smaller budgets, for a smoke test
    python3 benchmarks/run_all.py --world dungeon
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hunch import cli  # noqa: E402

WORLDS = {
    "dungeon": {"file": "dungeon.hunch", "budget": 400_000, "entropy": 0.02},
    "market":  {"file": "market.hunch",  "budget": 600_000, "entropy": None},
    "duel":    {"file": "duel.hunch",    "budget": 600_000, "entropy": None},
    "cache":   {"file": "cache.hunch",   "budget": 2_000_000, "entropy": None},
}

ALGOS = ["random", "reinforce", "ppo", "fork"]
SEEDS = [1, 2, 3, 4, 5]
QUICK_SEEDS = [1, 2]
QUICK_SCALE = 0.1

# Fixed-opponent probes: how well a self-play agent does against an opponent
# that has no learning at all. Raw self-play return is meaningless because both
# sides improve together.
FROZEN = {
    "duel": [("vs_fold", "bob:0"), ("vs_call", "bob:1")],
}


def run_world(name: str, cfg: dict, seeds: list[int], quick: bool) -> dict:
    file = ROOT / "worlds" / cfg["file"]
    budget = int(cfg["budget"] * (QUICK_SCALE if quick else 1))
    hbc, unit = cli.compile_file(file)
    cli.ensure_vm()

    out = ROOT / ".hunch-cache"
    tasks = []
    for algo in ALGOS:
        for seed in seeds:
            pol = out / f"bench.{name}.{algo}.{seed}.hpol"
            cmd = [str(cli.VM_BIN), "train", str(hbc), "--algo", algo,
                   "--budget", str(budget), "--seed", str(seed),
                   "--out", str(pol), "--eval-episodes", "96", "--json"]
            if cfg.get("entropy") is not None:
                cmd += ["--entropy", str(cfg["entropy"])]
            tasks.append((algo, seed, pol, cmd))

    results: dict[str, list] = {a: [] for a in ALGOS}
    running = []
    jobs = 4

    def launch(task):
        algo, seed, pol, cmd = task
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        running.append((algo, seed, pol, p))

    it = iter(tasks)
    for _ in range(min(jobs, len(tasks))):
        launch(next(it))

    while running:
        algo, seed, pol, p = running.pop(0)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print(f"  ! {name} {algo} seed {seed}: {stderr.strip()[:300]}", file=sys.stderr)
        else:
            data = json.loads(stdout.strip().splitlines()[-1])
            data["seed"] = seed
            data["policy"] = str(pol)
            results[algo].append(data)
        try:
            launch(next(it))
        except StopIteration:
            pass

    summary = {"world": unit["world"], "budget": budget, "horizon": unit["horizon"],
               "seeds": len(seeds), "algos": {}}

    for algo in ALGOS:
        runs = results[algo]
        if not runs:
            continue
        finals = [r["final_return"] for r in runs]
        mean = sum(finals) / len(finals)
        var = sum((x - mean) ** 2 for x in finals) / len(finals)
        entry = {
            "mean": round(mean, 6),
            "std": round(var ** 0.5, 6),
            "best": round(max(finals), 6),
            "walls": round(sum(r["wall_sec"] for r in runs) / len(runs), 3),
            "runs": [
                {"seed": r["seed"],
                 "final_return": round(r["final_return"], 6),
                 "wall_sec": round(r["wall_sec"], 3),
                 "curve": [{"step": c["step"], "return": round(c["return"], 6)}
                           for c in r["curve"]]}
                for r in runs
            ],
        }

        # extra fixed-opponent probes for self-play worlds
        probes = FROZEN.get(name)
        if probes and algo != "random":
            probe_vals = {}
            for label, freeze in probes:
                vals = []
                for r in runs:
                    proc = subprocess.run(
                        [str(cli.VM_BIN), "eval", str(hbc), "--policy", r["policy"],
                         "--episodes", "600", "--freeze", freeze],
                        capture_output=True, text=True)
                    if proc.returncode == 0:
                        vals.append(json.loads(proc.stdout.strip())["return"])
                if vals:
                    probe_vals[label] = round(sum(vals) / len(vals), 4)
            if probe_vals:
                entry["probes"] = probe_vals

        summary["algos"][algo] = entry
        print(f"  {name:8s} {algo:10s} mean={mean:9.4f} sd={entry['std']:7.4f} "
              f"wall={entry['walls']:5.1f}s" +
              (f"  {entry.get('probes', '')}" if entry.get("probes") else ""),
              file=sys.stderr)

    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--world", action="append")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results.json"))
    args = ap.parse_args()

    seeds = QUICK_SEEDS if args.quick else SEEDS
    names = args.world or list(WORLDS)

    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "quick": args.quick, "worlds": {}}

    t0 = time.time()
    for name in names:
        print(f"[{name}] budget={WORLDS[name]['budget']} seeds={seeds}", file=sys.stderr)
        out["worlds"][name] = run_world(name, WORLDS[name], seeds, args.quick)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out} in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
