#!/usr/bin/env python3
"""Check the exact answers against a brute force that shares no code with them.

Run with:  python3 tests/fuzz_exact.py [first_seed] [count]     (default 0 200)

Each seed makes a small random world: a few states, two or three actions, one
or two agents, observations that may or may not tell states apart, legal masks,
chance, sometimes two decisions by one agent in a step. For each world:

  scoring   `report` on uniform play, with and without merging positions, must
            give the return a plain Python recursion gives. A best reply the
            runtime calls sure must equal the best of every deterministic
            policy, tried one by one; one it does not call sure must not
            exceed that.
  training  after `train --algo auto`, the report's return must equal the
            Python evaluation of the saved table, and wherever the runtime
            says lookahead, proven, optimal or equilibrium, no deterministic
            reply of any one agent may gain more than 0.0001.

Exits 1 if any check fails. This was written by a reviewer looking for wrong
claims, not by the author of the code it checks.
"""
import itertools
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hunch import cli                               # noqa: E402

HERE = Path(tempfile.mkdtemp(prefix="hunch-fuzz-"))
VM = ""


def gen(rng):
    K = rng.randint(2, 4)          # states
    T = rng.randint(2, 4)          # horizon
    A = rng.randint(2, 3)          # actions
    M = rng.randint(1, 3)          # distinct observations
    two = rng.random() < 0.6       # second agent q
    q_sees_a = rng.random() < 0.4  # q is shown p's action of this step
    masks = rng.random() < 0.5
    twice = rng.random() < 0.3     # p decides twice in a step
    w = dict(K=K, T=T, A=A, M=M, two=two, q_sees_a=q_sees_a, masks=masks, twice=twice)
    w['nxt'] = [rng.randrange(K) for _ in range(K * A * A * 2)]   # (x, a, b, c)
    w['obp'] = [rng.randrange(M) for _ in range(K * T)]
    w['obq'] = [rng.randrange(M) for _ in range(K * T)]
    w['rwp'] = [rng.choice([0, 0, 1, 2, -1]) for _ in range(K * A * A)]
    w['rwq'] = [rng.choice([0, 0, 1, 2, -1]) for _ in range(K * A * A)]
    lg = []
    for _ in range(K):
        row = [1 if (not masks or rng.random() < 0.7) else 0 for _ in range(A)]
        if not any(row):
            row[rng.randrange(A)] = 1
        lg += row
    w['lg'] = lg
    return w


def source(w):
    K, T, A = w['K'], w['T'], w['A']
    arr = lambda xs: '[' + ', '.join(str(v) for v in xs) + ']'
    mask = '[' + ', '.join(f'lg[x * {A} + {i}]' for i in range(A)) + ']'
    s = f"""world Fz
horizon {T}
state {{
  x: int = 0;
  nxt: int[{len(w['nxt'])}] = {arr(w['nxt'])};
  obp: int[{K*T}] = {arr(w['obp'])};
  obq: int[{K*T}] = {arr(w['obq'])};
  rwp: float[{K*A*A}] = {arr(w['rwp'])};
  rwq: float[{K*A*A}] = {arr(w['rwq'])};
  lg: int[{K*A}] = {arr(w['lg'])};
}}
step {{
  let t = step_count() - 1;
  let a = hunch p([obp[x * {T} + t], 0], {A}, {mask});
  let b = 0;
"""
    if w['two']:
        seen = 'a' if w['q_sees_a'] else '0'
        s += f"  b = hunch q([obq[x * {T} + t], {seen}], {A}, {mask});\n"
    elif w['twice']:
        s += f"  b = hunch p([obp[x * {T} + t], 1], {A}, {mask});\n"
    s += f"""  reward(p, rwp[(x * {A} + a) * {A} + b]);
"""
    if w['two']:
        s += f"  reward(q, rwq[(x * {A} + a) * {A} + b]);\n"
    s += f"""  let c = randint(0, 1);
  x = nxt[((x * {A} + a) * {A} + b) * 2 + c];
}}
"""
    return s


def py_eval(w, pol, nag):
    """pol[agent]: dict (obs, mask) -> list of probabilities, a dict to ints for
    a deterministic policy, or None for uniform."""
    K, T, A = w['K'], w['T'], w['A']

    def legal(x):
        return [i for i in range(A) if w['lg'][x * A + i]]

    def dist(agent, obs, x):
        L = legal(x)
        m = sum(1 << i for i in L)
        row = None if pol[agent] is None else pol[agent].get((obs, m))
        if row is None:
            return [(i, 1.0 / len(L)) for i in L]
        if isinstance(row, int):
            return [(row, 1.0)]
        tot = sum(row[i] for i in L)
        return [(i, row[i] / tot) for i in L if row[i] > 0]

    def go(x, t):
        if t == T:
            return [0.0] * nag
        out = [0.0] * nag
        for a, pa in dist(0, (w['obp'][x * T + t], 0), x):
            if w['two']:
                seen = a if w['q_sees_a'] else 0
                bs = dist(1, (w['obq'][x * T + t], seen), x)
            elif w['twice']:
                bs = dist(0, (w['obp'][x * T + t], 1), x)
            else:
                bs = [(0, 1.0)]
            for b, pb in bs:
                r = [w['rwp'][(x * A + a) * A + b]]
                if w['two']:
                    r.append(w['rwq'][(x * A + a) * A + b])
                for c in (0, 1):
                    sub = go(w['nxt'][((x * A + a) * A + b) * 2 + c], t + 1)
                    for g in range(nag):
                        out[g] += pa * pb * 0.5 * (r[g] + sub[g])
        return out

    return go(0, 0)


def isets(w, agent):
    """Every (obs, mask) the agent can be shown, with its legal actions."""
    K, T, A = w['K'], w['T'], w['A']
    out = {}
    for x in range(K):
        L = [i for i in range(A) if w['lg'][x * A + i]]
        m = sum(1 << i for i in L)
        for t in range(T):
            if agent == 0:
                out[((w['obp'][x * T + t], 0), m)] = L
                if w['twice'] and not w['two']:
                    out[((w['obp'][x * T + t], 1), m)] = L
            else:
                for seen in (range(A) if w['q_sees_a'] else [0]):
                    out[((w['obq'][x * T + t], seen), m)] = L
    return out


def brute(w, me, nag, pol, limit=3000):
    """The most `me` can collect with a deterministic policy against `pol`, by
    trying every one. None when there are more than `limit` of them."""
    I = isets(w, me)
    keys = list(I)
    n = 1
    for k in keys:
        n *= len(I[k])
    if n > limit:
        return None
    best = -1e9
    for choice in itertools.product(*[I[k] for k in keys]):
        p2 = dict(pol)
        p2[me] = dict(zip(keys, choice))
        best = max(best, py_eval(w, p2, nag)[me])
    return best


def load_tab(path):
    lines = Path(path).read_text().split('\n')
    rows = int(lines[2].split()[1])
    pol = {0: {}, 1: {}}
    for ln in lines[3:3 + rows]:
        f = ln.split()
        agent, nact, mask, klen = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        obs = tuple(int(float(v)) for v in f[4:4 + klen])
        pol[agent][(obs, mask)] = [float(v) for v in f[4 + klen:4 + klen + nact]]
    return pol


def vm_json(*args):
    out = subprocess.run([VM, *map(str, args)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout.strip().splitlines()[-1])


def check_scoring(seed, stats):
    bad = []
    w = gen(random.Random(seed))
    nag = 2 if w['two'] else 1
    path = HERE / f'score{seed}.hunch'
    path.write_text(source(w))
    hbc, _ = cli.compile_file(path)
    uniform = {g: None for g in range(nag)}
    want = py_eval(w, uniform, nag)
    reports = (('merged', vm_json('report', hbc)), ('unmerged', vm_json('report', hbc, '--no-merge')))
    for g in range(nag):
        for name, rep in reports:
            if abs(rep['return'][g] - want[g]) > 1e-9:
                bad.append(f'seed {seed} agent {g}: {name} return {rep["return"][g]} != {want[g]}')
        best = brute(w, g, nag, uniform, 6000)
        if best is None:
            continue
        for name, rep in reports:
            got, sure = rep['best_reply'][g], rep['sure'][g]
            stats['sure' if sure else 'not sure'] = stats.get('sure' if sure else 'not sure', 0) + 1
            if got > best + 1e-9:
                bad.append(f'seed {seed} agent {g}: {name} best reply {got} > brute force {best}')
            if sure and abs(got - best) > 1e-9:
                bad.append(f'seed {seed} agent {g}: {name} sure best reply {got} != brute force {best}')
    return bad


def check_training(seed, stats):
    bad = []
    rng = random.Random(seed)
    w = gen(rng)
    if rng.random() < 0.5:      # every (state, step) is told apart
        K, T = w['K'], w['T']
        w['M'] = K * T
        w['obp'] = list(range(K * T))
        w['obq'] = list(range(K * T))
        w['q_sees_a'] = True
    if rng.random() < 0.5:      # zero-sum
        w['rwq'] = [-v for v in w['rwp']]
    nag = 2 if w['two'] else 1
    path = HERE / f'train{seed}.hunch'
    path.write_text(source(w))
    hbc, _ = cli.compile_file(path)
    hpol = HERE / 'policy.hpol'
    tr = vm_json('train', hbc, '--algo', 'auto', '--json', '--out', hpol,
                 '--max-sec', '2', '--target', '0.00001')
    rep = vm_json('report', hbc, '--policy', hpol)
    pol = load_tab(hpol)
    mine = py_eval(w, pol, nag)
    method = tr.get('method')
    stats[method] = stats.get(method, 0) + 1
    for g in range(nag):
        if abs(mine[g] - rep['return'][g]) > 1e-6:
            bad.append(f'seed {seed} agent {g}: report return {rep["return"][g]} != {mine[g]}')
    best = [brute(w, g, nag, pol) for g in range(nag)]
    if any(b is None for b in best):
        return bad
    gains = [best[g] - mine[g] for g in range(nag)]
    claims = []
    if method == 'lookahead':
        claims.append('lookahead')
    if tr.get('proven'):
        claims.append('proven')
    if all(rep['sure']) and sum(max(0.0, rep['best_reply'][g] - rep['return'][g])
                                for g in range(nag)) < 5e-5:
        claims.append('optimal' if nag == 1 else 'equilibrium')
    for c in claims:
        stats['claimed ' + c] = stats.get('claimed ' + c, 0) + 1
        if max(gains) > 1e-4:
            bad.append(f'seed {seed}: claimed {c}, but a reply gains {gains}')
    for g in range(nag):
        if rep['sure'][g] and abs(rep['best_reply'][g] - best[g]) > 1e-6:
            bad.append(f'seed {seed} agent {g}: sure best reply {rep["best_reply"][g]} '
                       f'!= brute force {best[g]}')
    return bad


def main() -> int:
    global VM
    VM = str(cli.ensure_vm())
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    bad, scoring, training = [], {}, {}
    for seed in range(first, first + count):
        bad += check_scoring(seed, scoring)
        bad += check_training(seed, training)
    for line in bad:
        print(line)
    print(f'{count} worlds scored: best replies checked {scoring}')
    print(f'{count} worlds trained: {training}')
    print('all checks held' if not bad else f'{len(bad)} check(s) FAILED; worlds are in {HERE}')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
