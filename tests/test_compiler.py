#!/usr/bin/env python3
"""Tests for the Hunch compiler, runtime and learning loop.

Run with:  python3 tests/test_compiler.py
No third-party dependencies.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hunch import cli, codegen                      # noqa: E402
from hunch.lang import HunchError, parse            # noqa: E402

FAILS: list[str] = []
PASSES = 0


def check(name: str, fn):
    global PASSES
    try:
        fn()
        PASSES += 1
        print(f"  ok   {name}")
    except AssertionError as exc:
        FAILS.append(f"{name}: {exc}")
        print(f"  FAIL {name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        FAILS.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")


def compile_src(src: str):
    return codegen.compile_program(parse(src, "<test>"))


def vm(args: list[str]) -> subprocess.CompletedProcess:
    cli.ensure_vm()
    return subprocess.run([str(cli.VM_BIN)] + args, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

def test_state_block_optional():
    """A world with nothing to remember needs no state block."""
    src = "world W\nhorizon 3\nstep {\n  let a = hunch p([1.0], 2);\n  reward(p, a);\n}\n"
    blob, unit = codegen.compile_program(parse(src, "nostate"))
    assert unit["agents"], "no agent declared"


def test_assert_passes():
    src = ("world W\nhorizon 3\nstate { n: int = 0; }\nstep {\n"
           "  n = n + 1;\n  assert n > 0;\n}\n")
    blob = compile_src(src)[0]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.hbc"
        path.write_bytes(blob)
        proc = vm(["trace", str(path), "--episodes", "1"])
        assert proc.returncode == 0, proc.stderr


def test_assert_fails_with_line():
    """A broken rule must stop the run and name the line, not train on garbage."""
    src = ("world W\nhorizon 3\nstate { n: int = 0; }\nstep {\n"
           "  n = n + 1;\n  assert n < 2;\n}\n")
    blob = compile_src(src)[0]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.hbc"
        path.write_bytes(blob)
        proc = vm(["trace", str(path), "--episodes", "1"])
        assert proc.returncode != 0, "a false assertion did not stop the run"
        assert "assertion failed" in proc.stderr, proc.stderr
        assert "line 6" in proc.stderr, proc.stderr


def test_diagnostic_shows_source():
    """Errors must point at the code, not just at a line number."""
    src = "world W\nhorizon 3\nstate { hits: int = 0; }\nstep {\n  let x = hots + 1;\n}\n"
    try:
        compile_src(src)
    except HunchError as exc:
        text = str(exc)
        assert "hots" in text, text
        assert "^" in text, "no caret in the diagnostic"
        assert "hits" in text, f"no suggested fix: {text}"
    else:
        raise AssertionError("unknown variable compiled")


def test_param_override_changes_behaviour():
    """A param must change what the world does, not just what it says."""
    src = ("world W\nhorizon 2\nparam r = 1.0;\n"
           "step { let a = hunch alice([1.0], 2); reward(alice, r); }\n")
    for override, want in ((None, 1.0), ({"r": 2.5}, 2.5)):
        prog = parse(src, "param-test")
        blob, _ = codegen.compile_program(prog, override)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "w.hbc"
            path.write_bytes(blob)
            proc = vm(["trace", str(path), "--episodes", "1"])
            assert proc.returncode == 0, proc.stderr
            got = json.loads(proc.stdout.strip().splitlines()[0])["episodes"][0]
            assert got["steps"][0]["r"]["alice"] == want, (override, got)


def test_param_override_does_not_reuse_cache():
    """Two overrides of one world must not share cached bytecode."""
    src = ("world W\nhorizon 2\nparam r = 1.0;\n"
           "step { let a = hunch alice([1.0], 2); reward(alice, r); }\n")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.hunch"
        path.write_text(src)
        first, _ = cli.compile_file(path, {"r": 1.0})
        second, _ = cli.compile_file(path, {"r": 2.0})
        assert first != second, "overrides collided in the bytecode cache"
        assert first.read_bytes() != second.read_bytes(), "cached bytecode is stale"


def test_cli_test_command():
    """hunch test must fail loudly on a world that violates its own rules."""
    good = "world W\nhorizon 2\nstate { n: int = 0; }\nstep { n = n + 1; assert n < 9; }\n"
    bad = "world W\nhorizon 2\nstate { n: int = 0; }\nstep { n = n + 1; assert n < 0; }\n"
    with tempfile.TemporaryDirectory() as d:
        gp, bp = Path(d) / "good.hunch", Path(d) / "bad.hunch"
        gp.write_text(good)
        bp.write_text(bad)
        assert cli.main(["test", str(gp)]) == 0, "a sound world failed its test"
        assert cli.main(["test", str(bp)]) != 0, "a broken world passed its test"


def test_cli_test_reports_what_ran():
    """hunch test must say which assertions ran, not how many were written,
    and must surface the clamped index and the zero divisor it papered over."""
    src = ("world W\n"                                   # 1
           "horizon 5\n"                                 # 2
           "state { g: int[4]; n: int = 0; z: int = 0; }\n"  # 3
           "step {\n"                                    # 4
           "  n = n + 1;\n"                              # 5
           "  assert n > 0;\n"                           # 6
           "  let v = g[n + 3];\n"                       # 7
           "  let w = n / z;\n"                          # 8
           "  if n > 100 {\n"                            # 9
           "    assert n < 0;\n"                         # 10
           "  }\n"
           "}\n")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.hunch"
        path.write_text(src)
        out = subprocess.run([str(ROOT / "bin" / "hunch"), "test", str(path)],
                             capture_output=True, text=True, cwd=str(ROOT))
        assert out.returncode == 0, f"exit {out.returncode}: {out.stderr}"
        # first what the compiler made of the file, then the result and its warnings
        shape, lines = out.stdout.strip().split(f"\n{path}: ", 1)
        assert shape.startswith("world      W\n") and "this world has no decisions" in shape, shape
        lines = lines.splitlines()
        assert len(lines) == 3, out.stdout
        assert "3 episode(s), 1 of 2 assertions ran and held, 1 never ran (line 10)" \
            in lines[0], lines[0]
        # 3 episodes of 5 steps: the index and the divisor are wrong every step.
        assert lines[1] == ("warning: line 7, column 12: index out of range on read, "
                            "clamped (15 times)"), lines[1]
        assert lines[2] == ("warning: line 8, column 13: division by zero, "
                            "result is 0 (15 times)"), lines[2]

        # A world whose only assertion runs, and that does nothing doubtful.
        path.write_text("world W\nhorizon 2\nstate { n: int = 0; }\n"
                        "step { n = n + 1; assert n < 9; }\n")
        out = subprocess.run([str(ROOT / "bin" / "hunch"), "test", str(path)],
                             capture_output=True, text=True, cwd=str(ROOT))
        assert out.returncode == 0, out.stderr
        assert "1 of 1 assertion ran and held, mean reward" in out.stdout, out.stdout
        assert "warning" not in out.stdout and "never ran" not in out.stdout, out.stdout

        # No assertions at all keeps the old wording.
        path.write_text("world W\nhorizon 2\nstate { n: int = 0; }\nstep { n = n + 1; }\n")
        out = subprocess.run([str(ROOT / "bin" / "hunch"), "test", str(path)],
                             capture_output=True, text=True, cwd=str(ROOT))
        assert out.returncode == 0, out.stderr
        assert "no assertions in this world, so nothing was checked" in out.stdout, out.stdout


# A world whose legal actions change every step. Agent p is masked by an
# array, agent q by a list; the asserts make any illegal choice stop the run.
MASKED_WORLD = ("world Masked\n"
                "horizon 6\n"
                "state { t: int = 0; }\n"
                "step {\n"
                "  let ok: int[4];\n"
                "  for i in 0..4 { ok[i] = 0; }\n"
                "  ok[t % 4] = 1;\n"
                "  ok[(t + 1) % 4] = 1;\n"
                "  let a = hunch p([t / 6.0], 4, ok);\n"
                "  assert ok[a] != 0;\n"
                "  let b = hunch q([1.0], 3, [t % 2, 1 - t % 2, 0]);\n"
                "  assert b == 1 - t % 2;\n"
                "  reward(p, 1.0);\n"
                "  reward(q, b);\n"
                "  t = t + 1;\n"
                "}\n")


def test_masked_actions_are_never_taken():
    """An untrained policy must not take an illegal action, nor give it any
    probability, and the trace must still list every action."""
    blob, _ = compile_src(MASKED_WORLD)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.hbc"
        path.write_bytes(blob)
        proc = vm(["trace", str(path), "--episodes", "20", "--seed", "5"])
        assert proc.returncode == 0, proc.stderr
        episodes = json.loads(proc.stdout)["episodes"]
        assert len(episodes) == 20, len(episodes)
        seen = set()
        for ep in episodes:
            assert len(ep["steps"]) == 6, len(ep["steps"])
            for t, st in enumerate(ep["steps"]):
                legal = {"p": {t % 4, (t + 1) % 4}, "q": {1 - t % 2}}
                for agent, n in (("p", 4), ("q", 3)):
                    probs = st["p"][agent]
                    assert len(probs) == n, f"{agent}: {probs}"
                    assert st["a"][agent] in legal[agent], \
                        f"step {t}: {agent} took illegal action {st['a'][agent]}"
                    for a in range(n):
                        if a not in legal[agent]:
                            assert probs[a] == 0, f"step {t}: {agent} gives {probs}"
                    assert abs(sum(probs) - 1.0) < 1e-4, f"{agent}: {probs}"
                seen.add(st["a"]["p"])
        assert seen == {0, 1, 2, 3}, f"the mask pinned p to {seen}"

        # The no-network policy takes a different path through the runtime; the
        # world's own asserts stop it (exit 3) if that path picks an illegal one.
        proc = vm(["train", str(path), "--algo", "random", "--budget", "3000", "--json"])
        assert proc.returncode == 0, proc.stderr

        # Counterfactual values: none for an illegal action, but the slot stays.
        proc = vm(["trace", str(path), "--seed", "5", "--forks"])
        assert proc.returncode == 0, proc.stderr
        for t, st in enumerate(json.loads(proc.stdout)["episodes"][0]["steps"]):
            q = st["q"]["p"]
            assert len(q) == 4, q
            for a in range(4):
                assert (q[a] is None) == (a not in (t % 4, (t + 1) % 4)), f"step {t}: {q}"


def test_mask_errors():
    cases = [
        ("world W\nhorizon 2\nstep {\n  let a = hunch p([1.0], 3, [1, 0]);\n  reward(p, a);\n}\n",
         "has 2 value(s) but hunch 'p' has 3 actions"),
        ("world W\nhorizon 2\nstate { ok: int[4]; }\nstep {\n"
         "  let a = hunch p([1.0], 3, ok);\n  reward(p, a);\n}\n",
         "'ok' has 4 elements but hunch 'p' has 3 actions"),
        ("world W\nhorizon 2\nstate { ok: int = 1; }\nstep {\n"
         "  let a = hunch p([1.0], 3, ok);\n  reward(p, a);\n}\n",
         "'ok' is not an array"),
        ("world W\nhorizon 2\nstate { legal: int[3]; }\nstep {\n"
         "  let a = hunch p([1.0], 3, leagl);\n  reward(p, a);\n}\n",
         "did you mean 'legal'?"),
        ("world W\nhorizon 2\nstate { ok: int[3]; }\nstep {\n"
         "  let a = hunch p([1.0], 3, ok[0]);\n  reward(p, a);\n}\n",
         "must be an array name or a [...] list"),
    ]
    for src, needle in cases:
        try:
            compile_src(src)
        except HunchError as exc:
            text = str(exc)
            assert needle in text, text
            assert "<test>:" in text and "^" in text, f"no location or caret: {text}"
            assert "= help:" in text, f"no hint: {text}"
        else:
            raise AssertionError(f"mask error not caught: {needle}")


def test_all_illegal_mask_stops_the_run():
    """A decision with nothing legal is a broken rule, like a false assert."""
    src = ("world W\nhorizon 3\nstate { t: int = 0; }\nstep {\n"
           "  t = t + 1;\n"
           "  let a = hunch p([1.0], 3, [t < 2, 0, 0]);\n"
           "  reward(p, a);\n}\n")
    blob = compile_src(src)[0]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.hbc"
        path.write_bytes(blob)
        proc = vm(["trace", str(path), "--episodes", "1"])
        assert proc.returncode == 3, f"exit {proc.returncode}: {proc.stderr}"
        assert "no legal action for 'p' at line 6, column 11" in proc.stderr, proc.stderr
        # So is asking for an action the mask rules out.
        proc = vm(["trace", str(path), "--agent", "p", "--actions", "1"])
        assert proc.returncode == 3, f"exit {proc.returncode}: {proc.stderr}"
        assert "scripted action 1 is illegal" in proc.stderr, proc.stderr
        assert "line 6, column 11" in proc.stderr, proc.stderr


def test_unknown_reward_agent_is_an_error():
    """A misspelt agent in reward() used to become an agent nobody plays."""
    src = ("world W\nhorizon 2\nstep {\n  let a = hunch alice([1.0], 2);\n"
           "  let b = hunch bob([1.0], 2);\n  reward(alice, a);\n  reward(bbo, b);\n}\n")
    try:
        compile_src(src)
    except HunchError as exc:
        text = str(exc)
        assert "<test>:7:10" in text, text
        assert "'bbo'" in text and "^^^" in text, text
        assert "did you mean 'bob'?" in text, text
    else:
        raise AssertionError("a reward to an agent with no hunch compiled")
    # The hunch may come later in the file than the reward that names it.
    late = ("world W\nhorizon 2\nstate { last: int = 0; }\nstep {\n"
            "  reward(alice, last);\n  last = hunch alice([1.0], 2);\n}\n")
    _, unit = compile_src(late)
    assert unit["agents"] == ["alice"], unit["agents"]


def _trace(src: str, name: str = "w"):
    blob, _ = compile_src(src)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.hbc"
        path.write_bytes(blob)
        proc = vm(["trace", str(path), "--episodes", "1", "--seed", "3"])
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.strip().splitlines()[0])


def test_helper_call_in_observation_is_harmless():
    """A function called inside an observation must not overwrite the values
    already staged before it. This silently rewrote the dungeon's observation."""
    with_call = _trace("world O\nhorizon 2\nstate { a: int = 0; }\n"
                       "fn id(v) { return v; }\n"
                       "step { let k = hunch p([7.0, 9.0, id(1.0)], 3); reward(p, k); }\n")
    literal = _trace("world O\nhorizon 2\nstate { a: int = 0; }\n"
                     "step { let k = hunch p([7.0, 9.0, 1.0], 3); reward(p, k); }\n")
    a = with_call["episodes"][0]["steps"][0]["p"]
    b = literal["episodes"][0]["steps"][0]["p"]
    assert a == b, f"observation changed by a helper call: {a} vs {b}"


def test_wide_observation_runs():
    """The compiler allows 64 observation values; the runtime must too."""
    obs = ", ".join(f"{i}.0" for i in range(40))
    out = _trace("world O\nhorizon 2\nstep { let k = hunch p([%s], 2); reward(p, k); }\n" % obs)
    probs = out["episodes"][0]["steps"][0]["p"]["p"]
    assert len(probs) == 2, probs
    assert all(0.0 <= v <= 1.0 for v in probs), probs


def test_twelve_by_twelve_board_fits():
    """Two 144-element arrays plus scalars must not outgrow the runtime."""
    src = ("world B\nhorizon 2\nstate { g: int[144]; s: int[144]; x: int = 0; "
           "y: int = 0; n: int = 0; m: int = 0; }\n"
           "step { g[y * 12 + x] = 1; s[y * 12 + x] = 1; n = n + 1; m = m + 1; }\n")
    _trace(src)


def test_array_decay_is_an_error():
    src = ("world D\nhorizon 2\nstate { a: int[3] = [1, 2, 3]; b: int[3] = [0, 0, 0]; }\n"
           "step { b = a; }\n")
    try:
        compile_src(src)
    except HunchError as exc:
        assert "used as a value" in str(exc), str(exc)
    else:
        raise AssertionError("array decay compiled")


def test_duplicate_definitions_are_errors():
    for src, needle in [
        ("world D\nhorizon 2\nstate { a: int = 0; a: int = 1; }\nstep { a = 1; }\n",
         "declared twice"),
        ("world D\nhorizon 2\nstate { a: int = 0; }\nfn f(x) { return x; }\n"
         "fn f(x) { return x; }\nstep { a = f(1); }\n", "defined twice"),
    ]:
        try:
            compile_src(src)
        except HunchError as exc:
            assert needle in str(exc), str(exc)
        else:
            raise AssertionError(f"duplicate definition compiled: {needle}")


def test_records_run():
    """A record is a named group of state slots, addressed by field."""
    src = ("world R\nhorizon 3\nrecord Slot { key: int; last: int; freq: int; }\n"
           "state { slots: Slot[2]; total: int = 0; }\n"
           "init { for i in 0..2 { slots[i].key = 0; slots[i].freq = 0; } }\n"
           "step { slots[1].freq = slots[0].key + 8; total = total + slots[1].freq;"
           " assert slots[1].freq == 8; }\n")
    out = _trace(src)
    assert out["episodes"][0]["steps"], "record world ran no steps"


def test_scalar_record_runs():
    src = ("world R\nhorizon 1\nrecord P { hp: int; mana: int; }\nstate { hero: P; }\n"
           "init { hero.hp = 10; hero.mana = 3; }\n"
           "step { hero.hp = hero.hp - 1; assert hero.hp == 9; }\n")
    _trace(src)


def test_record_errors():
    cases = [
        ("world W\nhorizon 2\nrecord P { hp: int; }\nstate { h: P; }\nstep { h.mana = 1; }\n",
         "has no field"),
        ("world W\nhorizon 2\nrecord P { hp: int; }\nstate { h: P; }\nstep { let x = h; }\n",
         "used as a value"),
        ("world W\nhorizon 2\nrecord P { hp: int; }\nstate { h: P[3]; }\nstep { let x = h.hp; }\n",
         "array of records"),
    ]
    for src, needle in cases:
        try:
            compile_src(src)
        except HunchError as exc:
            assert needle in str(exc), str(exc)
        else:
            raise AssertionError(f"record error not caught: {needle}")


def test_local_array_in_observation():
    """An observation can be built in a loop instead of written out by hand."""
    src = ("world L\nhorizon 2\nstate { total: int = 0; }\n"
           "step {\n"
           "  let v: float[3];\n"
           "  for i in 0..3 { v[i] = i * 2.0; }\n"
           "  assert v[0] == 0.0 && v[2] == 4.0;\n"
           "  let k = hunch p(v, 2);\n"
           "  total = total + k;\n"
           "  reward(p, 1.0);\n"
           "}\n")
    out = _trace(src)
    steps = out["episodes"][0]["steps"]
    assert len(steps[0]["p"]["p"]) == 2, steps[0]["p"]


def test_local_array_errors():
    cases = [
        ("world L\nhorizon 2\nstep { let v: float[3] = 1.0; }\n", "array"),
        ("world L\nhorizon 2\nstep { let v: float[2] = [1.0, 2.0, 3.0]; }\n", "elements"),
    ]
    for src, needle in cases:
        try:
            compile_src(src)
        except HunchError as exc:
            assert needle in str(exc), str(exc)
        else:
            raise AssertionError(f"local array error not caught: {needle}")


def test_worlds_compile():
    for name in ("dungeon", "market", "duel", "cache"):
        src = (ROOT / "worlds" / f"{name}.hunch").read_text()
        blob, unit = codegen.compile_program(parse(src, name))
        assert len(blob) > 100, f"{name} produced no bytecode"
        assert unit["agents"], f"{name} declared no hunch agents"


def test_syntax_error_has_location():
    try:
        parse("world X\nstate {\n  a: int = ;\n}\nstep { }\n", "bad.hunch")
    except HunchError as exc:
        assert exc.line > 0, "error did not carry a line number"
        assert "bad.hunch" in str(exc), "error did not carry the file name"
        return
    raise AssertionError("expected a syntax error")


def test_unknown_variable():
    src = """
world X
state { a: int = 0; }
step { a = missing + 1; }
"""
    try:
        codegen.compile_program(parse(src, "u.hunch"))
    except HunchError as exc:
        assert "missing" in str(exc)
        return
    raise AssertionError("expected an unknown-variable error")


def test_recursion_rejected():
    src = """
world X
state { a: int = 0; }
step { a = f(1); }
fn f(n) { return f(n - 1); }
"""
    try:
        codegen.compile_program(parse(src, "r.hunch"))
    except HunchError as exc:
        assert "recursi" in str(exc).lower()
        return
    raise AssertionError("expected a recursion error")


def test_inconsistent_obs_rejected():
    src = """
world X
state { a: int = 0; }
step {
  let x = hunch p([a], 2);
  let y = hunch p([a, a], 2);
}
"""
    try:
        codegen.compile_program(parse(src, "o.hunch"))
    except HunchError:
        return
    raise AssertionError("expected an observation-length error")


def test_ambiguous_reward_rejected():
    src = """
world X
state { a: int = 0; }
step {
  let x = hunch p([a], 2);
  let y = hunch q([a], 2);
  reward(1.0);
}
"""
    try:
        codegen.compile_program(parse(src, "amb.hunch"))
    except HunchError as exc:
        assert "ambiguous" in str(exc).lower()
        return
    raise AssertionError("expected an ambiguous-reward error")


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

def test_deterministic_replay():
    """Same seed, same world, same policy => bit-identical trace."""
    hbc, _ = cli.compile_file(ROOT / "worlds" / "dungeon.hunch")
    a = vm(["trace", str(hbc), "--seed", "1234"])
    b = vm(["trace", str(hbc), "--seed", "1234"])
    assert a.returncode == 0, a.stderr
    assert a.stdout == b.stdout, "two runs of the same seed diverged"


def test_snapshot_isolation():
    """A traced episode must not leak into the next one.

    Two episodes in one process must match two separate single-episode runs;
    otherwise the runtime is not really snapshotting the whole universe.
    """
    hbc, _ = cli.compile_file(ROOT / "worlds" / "market.hunch")
    one = vm(["trace", str(hbc), "--seed", "42", "--episodes", "1"])
    many = vm(["trace", str(hbc), "--seed", "42", "--episodes", "3"])
    assert one.returncode == 0 and many.returncode == 0
    first = json.loads(one.stdout)["episodes"][0]
    third = json.loads(many.stdout)["episodes"][0]
    assert first == third, "episode 1 changed when more episodes ran after it"


def test_scripted_actions_are_exact():
    """--actions must reproduce the hand-written path exactly."""
    hbc, _ = cli.compile_file(ROOT / "worlds" / "dungeon.hunch")
    out = vm(["trace", str(hbc), "--seed", "1", "--agent", "hero",
              "--actions", "3,3,3,3,1,1,1,1"])
    data = json.loads(out.stdout)
    total = sum(v for st in data["episodes"][0]["steps"] for v in st["r"].values())
    assert total > 1.0, f"optimal path scored only {total:.3f}, expected the win"


def test_budget_is_respected():
    """Training must stop at the budget, even with forked rollouts."""
    hbc, _ = cli.compile_file(ROOT / "worlds" / "dungeon.hunch")
    out = vm(["train", str(hbc), "--algo", "fork", "--budget", "40000",
              "--seed", "3", "--json", "--eval-episodes", "16"])
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["steps_used"] <= 40000 + 60, f"overshot budget: {data['steps_used']}"


def test_learning_beats_random():
    """A tiny world with an obvious action must be solved."""
    src = """
world Walk
horizon 12

state { x: int = 0; }

step {
  let a = hunch walker([x / 10.0], 2);
  if a == 0 { x = x - 1; } else { x = x + 1; }
  if x < 0 { x = 0; }
  if x > 10 { x = 10; }
  if x == 10 {
    reward(1.0);
    done();
  }
  reward(-0.01);
}
"""
    tmp = Path(tempfile.mkdtemp()) / "walk.hunch"
    tmp.write_text(src)
    hbc, _ = cli.compile_file(tmp)

    rand = vm(["train", str(hbc), "--algo", "random", "--budget", "60000",
               "--seed", "1", "--json", "--eval-episodes", "32"])
    trained = vm(["train", str(hbc), "--algo", "fork", "--budget", "60000",
                  "--seed", "1", "--json", "--eval-episodes", "32"])
    assert rand.returncode == 0 and trained.returncode == 0
    r0 = json.loads(rand.stdout.strip().splitlines()[-1])["final_return"]
    r1 = json.loads(trained.stdout.strip().splitlines()[-1])["final_return"]
    assert r1 > r0 + 0.2, f"learning did not help: random={r0:.3f} trained={r1:.3f}"


def test_learning_respects_the_mask():
    """Action 2 pays most but is never legal, so the best a learner can do is
    action 1. It must get there, and must not be credited with action 2."""
    src = """
world Pick
horizon 4

step {
  let a = hunch p([1.0], 3, [1, 1, 0]);
  if a == 2 { reward(p, 1.0); }
  else if a == 1 { reward(p, 0.5); }
  else { reward(p, 0.0); }
}
"""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "pick.hunch").write_text(src)
    hbc, _ = cli.compile_file(tmp / "pick.hunch")
    pol = tmp / "pick.hpol"
    trained = vm(["train", str(hbc), "--algo", "ppo", "--budget", "40000",
                  "--seed", "1", "--json", "--eval-episodes", "64", "--out", str(pol)])
    assert trained.returncode == 0, trained.stderr
    ret = json.loads(trained.stdout.strip().splitlines()[-1])["final_return"]
    # Always action 1 is 4 x 0.5; always action 2 would be 4.0.
    assert abs(ret - 2.0) < 0.1, f"trained return {ret:.3f}, expected 2.0 (action 1)"

    out = vm(["trace", str(hbc), "--policy", str(pol), "--episodes", "20", "--seed", "9"])
    assert out.returncode == 0, out.stderr
    for ep in json.loads(out.stdout)["episodes"]:
        for st in ep["steps"]:
            assert st["a"]["p"] != 2, "the trained policy took the masked action"
            assert st["p"]["p"][2] == 0, st["p"]["p"]
            assert st["p"]["p"][1] > 0.9, f"did not settle on action 1: {st['p']['p']}"


def test_fork_measures_all_actions():
    """--forks must emit a value for every action at every decision."""
    hbc, unit = cli.compile_file(ROOT / "worlds" / "dungeon.hunch")
    pol = ROOT / "policies" / "dungeon.hpol"
    args = ["trace", str(hbc), "--seed", "7", "--forks"]
    if pol.exists():
        args += ["--policy", str(pol)]
    out = vm(args)
    assert out.returncode == 0, out.stderr
    steps = json.loads(out.stdout)["episodes"][0]["steps"]
    assert any("q" in st for st in steps), "no fork values in the trace"
    for st in steps:
        if "q" in st:
            for agent, q in st["q"].items():
                n = next(a["n_act"] for a in json.loads(out.stdout)["agents"]
                         if a["name"] == agent)
                assert len(q) == n, f"{agent}: {len(q)} values for {n} actions"


def test_self_play_agent_beats_a_folder():
    """The trained duel agent should punish an opponent that always folds."""
    pol = ROOT / "policies" / "duel.hpol"
    if not pol.exists():
        return
    hbc, _ = cli.compile_file(ROOT / "worlds" / "duel.hunch")
    out = vm(["eval", str(hbc), "--policy", str(pol), "--episodes", "800",
              "--freeze", "bob:0"])
    r = json.loads(out.stdout)["return"]
    assert r > 0.8, f"alice only scored {r:.3f} against a folder"


# ---------------------------------------------------------------------------
# Named actions, exact answers, and the commands that explain a policy
# ---------------------------------------------------------------------------

def hunch(*args, stdin: str = None) -> subprocess.CompletedProcess:
    """The command as a person runs it."""
    return subprocess.run([str(ROOT / "bin" / "hunch")] + [str(a) for a in args],
                          capture_output=True, text=True, cwd=str(ROOT), input=stdin)


def hunch_json(*args) -> dict:
    out = hunch(*args)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_named_actions():
    """`hunch p(obs, [left, right])` names the actions, and the names are values."""
    src = ("world W\nhorizon 4\nstate { x: int = 0; }\nstep {\n"
           "  let a = hunch p([x], [left, right]);\n"
           "  if a == right { x = x + 1; }\n"
           "  if a == left { x = x - 1; }\n}\n")
    _, unit = compile_src(src)
    assert unit["agent_acts"]["p"] == 2
    assert unit["agent_action_names"]["p"] == ["left", "right"]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.hunch"
        path.write_text(src)
        hbc, _ = cli.compile_file(path)
        # always action 1, which the world calls `right`: x counts the steps
        out = vm(["trace", str(hbc), "--agent", "p", "--actions", "1,1,1,1"])
        steps = json.loads(out.stdout)["episodes"][0]["steps"]
        assert [s["s"][0] for s in steps] == [1, 2, 3, 4], steps
    for bad, why in [
        (src.replace("state { x: int = 0; }", "state { x: int = 0; left: int = 0; }"),
         "name of an action"),
        (src.replace("let a = hunch", "let b = hunch p([x], [right, left]);\n  let a = hunch"),
         "is number 0 here and number 1 at line"),
        (src.replace("[left, right]", "[left, 2]"), "an action list holds names"),
        (src.replace("let a = hunch", "let b = hunch p([x], [left, stay]);\n  let a = hunch"),
         "two different action lists"),
    ]:
        try:
            compile_src(bad)
        except HunchError as exc:
            assert why in str(exc), str(exc)
        else:
            raise AssertionError(f"compiled, but should have said: {why}")


def kuhn_by_hand():
    """Kuhn poker worked out here, with nothing from the runtime: what alice
    collects when both play at random, and what the best reply of each player
    to a random opponent collects."""
    import itertools
    deals = [(a, b) for a in range(3) for b in range(3) if a != b]

    def play(a, b, alice, bob):
        """alice: (card, facing a bet) -> chance of betting; likewise bob."""
        show = 1 if a > b else -1
        total = 0.0
        for a1 in (0, 1):                       # alice opens: pass or bet
            p1 = alice[(a, 0)] if a1 else 1 - alice[(a, 0)]
            for b1 in (0, 1):
                p2 = bob[(b, a1)] if b1 else 1 - bob[(b, a1)]
                if a1 == 1:                     # bob faces a bet: fold or call
                    total += p1 * p2 * (2 * show if b1 else 1)
                elif b1 == 0:                   # pass, pass
                    total += p1 * p2 * show
                else:                           # pass, bet: alice folds or calls
                    call = alice[(a, 1)]
                    total += p1 * p2 * (call * 2 * show + (1 - call) * -1)
        return total

    def value(alice, bob):
        return sum(play(a, b, alice, bob) for a, b in deals) / len(deals)

    keys = [(c, f) for c in range(3) for f in (0, 1)]
    rand = {k: 0.5 for k in keys}
    pures = [dict(zip(keys, bits)) for bits in itertools.product((0.0, 1.0), repeat=6)]
    return (value(rand, rand),
            max(value(pure, rand) for pure in pures),
            max(-value(rand, pure) for pure in pures))


def test_exact_scoring_matches_a_hand_count():
    """The exact report against a brute force that shares no code with it."""
    want, best_a, best_b = kuhn_by_hand()
    rep = hunch_json("score", ROOT / "worlds" / "kuhn.hunch", "--policy", "none", "--json")
    assert rep["exact"], rep
    assert abs(rep["return"][0] - want) < 1e-9, (rep["return"], want)
    assert abs(rep["return"][1] + want) < 1e-9, (rep["return"], want)
    assert abs(rep["best_reply"][0] - best_a) < 1e-9, (rep["best_reply"], best_a)
    assert abs(rep["best_reply"][1] - best_b) < 1e-9, (rep["best_reply"], best_b)
    assert rep["sure"] == [True, True], rep["sure"]


def test_train_solves_kuhn():
    """Two players and hidden cards: regret matching over every outcome, to the
    known value of the game, -1/18 to the first player."""
    with tempfile.TemporaryDirectory() as d:
        pol = Path(d) / "kuhn.hpol"
        got = hunch_json("train", ROOT / "worlds" / "kuhn.hunch", "--out", pol, "--json")
        assert got["train"]["method"] == "regret", got["train"]
        rep = got["report"]
        gain = sum(b - r for b, r in zip(rep["best_reply"], rep["return"]))
        assert 0 <= gain < 0.002, f"NashConv {gain}"
        assert abs(rep["return"][0] + 1 / 18) < 0.002, rep["return"]
        # Merging positions that share a state is a saving, never a change.
        hbc, _ = cli.compile_file(ROOT / "worlds" / "kuhn.hunch")
        merged = json.loads(vm(["report", str(hbc), "--policy", str(pol)]).stdout)
        whole = json.loads(vm(["report", str(hbc), "--policy", str(pol), "--no-merge"]).stdout)
        for key in ("return", "best_reply", "vs_random"):
            for x, y in zip(merged[key], whole[key]):
                assert abs(x - y) < 1e-9, (key, merged[key], whole[key])
        # What an action is worth in a situation: an agent that mixes two actions
        # at an equilibrium is indifferent between them, and the values weighed
        # by the policy add up to what the agent collects.
        total = 0.0
        for row in merged["rows"]:
            p, v = row["probs"], row["value"]
            if min(p) > 0.05:
                assert abs(v[0] - v[1]) < 0.01, row
            if row["agent"] == 0 and row["obs"][1] == 0:      # alice's opening move
                total += row["share"] * sum(pi * vi for pi, vi in zip(p, v))
        assert abs(total - merged["return"][0]) < 1e-6, (total, merged["return"])
        # A policy trained on one world is refused by another.
        out = hunch("score", ROOT / "worlds" / "rps.hunch", "--policy", pol)
        assert out.returncode != 0 and "does not fit this world" in out.stderr, out.stderr


def test_train_looks_ahead_when_nothing_is_hidden():
    """Tic-tac-toe is a draw, and the policy found never loses to anything."""
    with tempfile.TemporaryDirectory() as d:
        got = hunch_json("train", ROOT / "worlds" / "tictactoe.hunch",
                         "--out", Path(d) / "t.hpol", "--json")
    assert got["train"]["method"] == "lookahead", got["train"]
    rep = got["report"]
    assert rep["return"] == [0, 0], rep["return"]
    assert rep["best_reply"] == [0, 0] and rep["sure"] == [True, True], rep
    # against random play it must do better than a draw on average
    assert rep["vs_random"][0] > 0.5 and rep["vs_random"][1] > 0.5, rep["vs_random"]


CORRIDOR = """world Corridor
horizon 8
state { x: int = 0; }
step {
  // the goal is three squares to the right; every step costs a tenth
  let a = hunch walker([x], [left, right]);
  if a == right { x = x + 1; } else if x > 0 { x = x - 1; }
  reward(-0.1);
  if x == 3 { reward(1); done(); }
}
"""


def test_train_proves_a_one_agent_optimum():
    """Three steps right is the best there is: 1 - 3 * 0.1."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "corridor.hunch"
        path.write_text(CORRIDOR)
        got = hunch_json("train", path, "--json")
        rep = got["report"]
        assert abs(rep["return"][0] - 0.7) < 1e-9, rep["return"]
        assert abs(rep["best_reply"][0] - 0.7) < 1e-9 and rep["sure"] == [True], rep
        # train leaves the policy beside the world, and the other commands find it
        assert path.with_suffix(".hpol").exists()
        text = hunch("train", path).stdout
        assert "this one is optimal" in text, text
        run = hunch("run", path).stdout
        assert run.count("plays right (100%)") == 3 and "ended after 3 steps" in run, run
        assert "walker collected +0.7000" in run, run


def test_sampled_regret_closes_the_gap():
    """With the exact graph turned off, self-play learns from sampled episodes;
    the result is still scored exactly."""
    world = ROOT / "worlds" / "kuhn.hunch"
    start = hunch_json("score", world, "--policy", "none", "--json")
    before = sum(b - r for b, r in zip(start["best_reply"], start["return"]))
    with tempfile.TemporaryDirectory() as d:
        pol = Path(d) / "kuhn.hpol"
        got = hunch_json("train", world, "--out", pol, "--json",
                         "--algo", "regret", "--exact-nodes", "0", "--budget", "300000")
        assert got["train"]["method"] == "sampled", got["train"]
        rep = hunch_json("score", world, "--policy", pol, "--json")
    after = sum(b - r for b, r in zip(rep["best_reply"], rep["return"]))
    assert before > 0.8, before
    assert after < 0.05, f"NashConv {after} after sampled self-play (random play: {before})"


def test_why_replays_each_decision():
    """`hunch why` forks every decision: the action taken, replayed as it
    happened, must collect what the episode did from there on."""
    world = ROOT / "worlds" / "leduc.hunch"
    with tempfile.TemporaryDirectory() as d:
        pol = Path(d) / "leduc.hpol"
        hunch_json("train", world, "--out", pol, "--json")
        why = hunch_json("why", world, "--policy", pol, "--seed", "3", "--json")
        hbc, _ = cli.compile_file(world)
        trace = json.loads(vm(["trace", str(hbc), "--policy", str(pol), "--seed", "3"]).stdout)
        steps = trace["episodes"][0]["steps"]
        names = ["alice", "bob"]
        assert len(why["decisions"]) == len(steps)
        for dec in why["decisions"]:
            to_go = sum(s["r"].get(names[dec["agent"]], 0.0) for s in steps[dec["step"] - 1:])
            assert abs(dec["actual"][dec["action"]] - to_go) < 1e-9, (dec, to_go)
            assert dec["diff"][dec["action"]] == 0
            # At an equilibrium no action is better going by what the agent sees,
            # however the hidden cards made it look this time.
            shown = dec["shown_value"]
            for a, v in enumerate(shown):
                if v is not None:
                    assert v - shown[dec["action"]] < 0.01, dec
        text = hunch("why", world, "--policy", pol, "--seed", "3").stdout
        assert "whenever shown this" in text and "this episode" in text, text
        assert "no decision had a better action" in text, text


def test_play_takes_a_seat():
    """A person plays one agent, by action name or number; illegal input is asked again."""
    world = ROOT / "worlds" / "kuhn.hunch"
    out = hunch("play", world, "--policy", "none", "--seed", "5", "--as", "bob",
                stdin="fold\n7\nbet\n")
    assert out.returncode == 0, out.stderr
    text = out.stdout
    assert "you are bob" in text and "you see card_b = " in text, text
    assert text.count("not one of the moves you can make here") == 2, text
    assert "bob sees" in text and "plays bet (you)" in text, text
    assert "ended after" in text, text
    out = hunch("play", world, "--as", "carol", stdin="")
    assert out.returncode != 0 and "no agent named 'carol'" in out.stderr, out.stderr


NIM = """world Nim
horizon 10
state { pile: int = 10; turn: int = 0; }
step {
  // take 1, 2 or 3 stones; whoever takes the last one wins
  let take = 1;
  if turn == 0 {
    take = 1 + hunch first([pile], [take1, take2, take3], [1, pile >= 2, pile >= 3]);
  } else {
    take = 1 + hunch second([pile], [take1, take2, take3], [1, pile >= 2, pile >= 3]);
  }
  pile = pile - take;
  if pile == 0 {
    if turn == 0 { reward(first, 1); reward(second, -1); }
    else { reward(first, -1); reward(second, 1); }
    done();
  }
  turn = 1 - turn;
}
"""


def test_positions_never_reached_are_still_played_well():
    """Ten stones is a win for the first player, so correct play never shows the
    second player five stones. A policy must still know to take one there: the
    opponent may be a person, or random."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nim.hunch"
        path.write_text(NIM)
        # The pile is the same observation at different steps, and still all
        # that matters: looking ahead applies, and the answer is exact.
        got = hunch_json("train", path, "--json")
        assert got["train"]["method"] == "lookahead", got["train"]
        rep = got["report"]
        assert rep["return"] == [1, -1] and rep["best_reply"] == [1, -1], rep
        assert rep["sure"] == [True, True], rep["sure"]
        assert rep["vs_random"][1] > 0.86, rep["vs_random"]
        # The same world through regret matching, which only weighs what is reached.
        hbc, _ = cli.compile_file(path)
        pol = Path(d) / "regret.hpol"
        out = vm(["train", str(hbc), "--algo", "regret", "--no-lookahead", "--out", str(pol),
                  "--json"])
        assert json.loads(out.stdout.strip().splitlines()[-1])["method"] == "regret"
        rep = json.loads(vm(["report", str(hbc), "--policy", str(pol), "--rows", "100"]).stdout)
        five = [r for r in rep["rows"] if r["agent"] == 1 and r["obs"] == [5]]
        assert five and five[0]["probs"][0] > 0.99, five
        assert rep["vs_random"][1] > 0.86, rep["vs_random"]
        # Scripted moves go by name, and a wrong name is refused, not played as 0.
        run = hunch("run", path, "--agent", "first", "--actions", "take2,take1", "--policy", "none")
        assert run.stdout.count("(scripted)") == 2 and "plays take2 (scripted)" in run.stdout, run.stdout
        bad = hunch("run", path, "--agent", "first", "--actions", "take9")
        assert bad.returncode != 0 and "is not an action of first" in bad.stderr, bad.stderr


def test_only_state_lasts_between_steps():
    """Two instants with the same state and step number must have the same
    future: the exact answers merge them. So nothing but `state` may carry
    from one step to the next."""
    carry = ("world W\nhorizon 3\nstate { t: int = 0; }\nstep {\n"
             "  let a = hunch p([step_count()], [down, up]);\n"
             "  let c = c + a;\n  t = t + 1;\n"
             "  if t == 3 { if c == 3 { reward(1); } done(); }\n}\n")
    array = ("world W\nhorizon 2\ninit { let memo: int[2]; }\n"
             "step {\n  let a = hunch p([1], 2);\n  memo[0] = memo[0] + a;\n  reward(memo[0]);\n}\n")
    for src, why in [(carry, "unknown variable 'c'"), (array, "is not in scope here")]:
        try:
            compile_src(src)
        except HunchError as exc:
            assert why in str(exc), str(exc)
        else:
            raise AssertionError(f"compiled, but should have said: {why}")
    # A local left unassigned on some path reads as zero, not as last step's value.
    src = ("world W\nhorizon 4\nstate { t: int = 0; }\nstep {\n"
           "  let a = hunch p([t], [down, up]);\n"
           "  let seen = 0;\n  let kept: int[2];\n"
           "  if t == 0 { kept[0] = 5; seen = 5; }\n"
           "  reward(kept[0]);\n  t = t + 1;\n}\n")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.hunch"
        path.write_text(src)
        rep = hunch_json("score", path, "--policy", "none", "--json")
        assert rep["exact"] and rep["return"] == [5], rep["return"]
        hbc, _ = cli.compile_file(path)
        whole = json.loads(vm(["report", str(hbc), "--no-merge"]).stdout)
        assert whole["return"] == [5], whole["return"]


def test_exact_claims_survive_extremes():
    """Two places where a number once went wrong without a word."""
    # 250 decisions with 32 actions each: more ways to a position than a double
    # can count. Random play must not be called optimal.
    deep = ("world Deep\nhorizon 250\nstate { alive: int = 1; }\nstep {\n"
            "  let a = hunch p([alive, step_count()], 32);\n"
            "  if a == 0 { alive = 0; }\n"
            "  if step_count() == 250 { reward(alive); done(); }\n}\n")
    # A die with two thousand million faces cannot be followed face by face.
    wide = ("world Wide\nhorizon 1\nstate { s: int = 0; }\nstep {\n"
            "  let a = hunch p([0], 2);\n"
            "  if a == 0 { s = randint(0, 2147483647); } else { reward(1); }\n  done();\n}\n")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "deep.hunch"
        path.write_text(deep)
        rep = hunch_json("score", path, "--policy", "none", "--json")
        assert rep["return"][0] < 0.01 and abs(rep["best_reply"][0] - 1) < 1e-9, rep
        got = hunch_json("train", path, "--json")
        assert got["report"]["return"] == [1] and got["report"]["sure"] == [True], got["report"]
        path = Path(d) / "wide.hunch"
        path.write_text(wide)
        rep = hunch_json("score", path, "--policy", "none", "--json", "--episodes", "400")
        assert not rep["exact"] and rep["why"].startswith("rand:"), rep
        assert 0.4 < rep["return"][0] < 0.6, rep["return"]


# ---------------------------------------------------------------------------
# Search: the policy for a world too big to follow whole with nothing hidden
# ---------------------------------------------------------------------------

def _train_vm(world: Path, pol: Path, *flags) -> dict:
    """`train` in the runtime itself: the choice of method is made there."""
    hbc, _ = cli.compile_file(world)
    proc = vm(["train", str(hbc), "--out", str(pol), "--json"] + [str(f) for f in flags])
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_train_makes_a_search_for_connect_four():
    """Connect Four cannot be followed whole and hides nothing, so `auto`
    saves a search. (--exact-nodes only makes "too big" quick to find out;
    --games 0 leaves out the network, which test_train_learns_to_search
    covers.)"""
    from hunch import show
    world = ROOT / "worlds" / "connect4.hunch"
    with tempfile.TemporaryDirectory() as d:
        pol = Path(d) / "c4.hpol"
        tr = _train_vm(world, pol, "--algo", "auto", "--exact-nodes", "50000", "--think", "0.02",
                       "--games", "0")
        assert tr["algo"] == "mcts" and tr["method"] == "mcts" and tr["chosen"] is True, tr
        assert tr["hidden"] == "nothing" and tr["not_whole"] == "big:50000", tr
        seen = tr["hidden_check"]
        assert seen["conflicts"] == 0 and seen["repeats"] >= seen["min_repeats"], seen
        assert seen["distinct"] >= seen["min_distinct"], seen
        assert tr["playouts"] >= 16 and tr["playouts_per_sec"] > 0, tr
        # about --think seconds of the playouts measured, to two figures
        want = tr["playouts_per_sec"] * 0.02
        assert 0.9 * want - 1 <= tr["playouts"] <= 1.1 * want + 1, tr
        head = pol.read_text().split()
        assert head[:4] == ["HUNCHSEARCH", "1", "world", "ConnectFour"], head
        assert head[4:6] == ["playouts", str(tr["playouts"])], head
        said = show.how_trained(tr)
        assert "too big to follow whole (more than 50,000 positions)" in said, said
        assert "in 40,000 sampled positions nothing an agent was shown hid anything from it" \
            in said and "so the policy is a search" in said, said
        assert f"{tr['playouts']:,} playouts a decision" in said, said
        out = hunch("score", world, "--policy", pol, "--score-sec", "1")
        assert out.returncode == 0, out.stderr
        text = out.stdout
        assert "the policy is a search" in text and "± is one standard error" in text, text
        assert "searching against opponents playing at random" in text, text
        assert "played for about 1 s" in text and "episode" in text, text
        rep = hunch_json("score", world, "--policy", pol, "--score-sec", "1", "--json")
        assert rep["exact"] is False and rep["why"] == "search" and rep["rows"] == [], rep
        assert rep["episodes"] >= 1 and all(n >= 1 for n in rep["vs_random_episodes"]), rep
        assert rep["sec"] < 10, f"scoring for 1 s took {rep['sec']} s"


def test_search_replays_and_explains():
    """A search draws nothing from the world's random stream and is seeded by
    the instant, so an episode replays exactly; `play` relies on that, and
    `why` says what the search found."""
    world = ROOT / "worlds" / "tictactoe.hunch"
    with tempfile.TemporaryDirectory() as d:
        pol = Path(d) / "t.hpol"
        tr = _train_vm(world, pol, "--algo", "mcts", "--playouts", "2000", "--games", "0")
        assert tr["method"] == "mcts" and tr["chosen"] is False and tr["playouts"] == 2000, tr
        assert tr["playouts_set"] is True and tr["hidden"] == "nothing", tr
        assert tr["learned"] is None, tr
        one = hunch("run", world, "--policy", pol, "--seed", "7")
        two = hunch("run", world, "--policy", pol, "--seed", "7")
        assert one.returncode == 0, one.stderr
        assert one.stdout == two.stdout, "the same seed played two different episodes"
        assert "a search, 2,000 playouts a decision" in one.stdout, one.stdout
        assert "of 2,000 playouts tried it" in one.stdout, one.stdout
        assert "x collected 0, o collected 0" in one.stdout, one.stdout
        # the same episode through the raw trace, twice, in one process and two
        hbc, _ = cli.compile_file(world)
        a = vm(["trace", str(hbc), "--policy", str(pol), "--seed", "7", "--episodes", "2"]).stdout
        b = vm(["trace", str(hbc), "--policy", str(pol), "--seed", "7", "--episodes", "2"]).stdout
        assert a == b and json.loads(a)["episodes"][0]["steps"], a[:200]

        # play re-runs the episode from the start for every move: it only works
        # if the search answers the same instant the same way each time
        out = hunch("play", world, "--policy", pol, "--seed", "3", "--as", "o",
                    stdin="".join(f"{k}\n" for k in list(range(9)) * 5))
        assert out.returncode == 0, out.stderr
        assert "(a search, 2,000 playouts a decision)" in out.stdout, out.stdout
        assert "ended after" in out.stdout and "(you)" in out.stdout, out.stdout
        assert "o collected +1" not in out.stdout, "taking the first free square beat the search"

        why = hunch_json("why", world, "--policy", pol, "--seed", "7", "--json")
        assert why["search"]["playouts"] == 2000 and "samples" not in why, why
        trace = json.loads(a)["episodes"][0]
        assert len(why["decisions"]) == len(trace["steps"])
        for dec, step in zip(why["decisions"], trace["steps"]):
            assert dec["action"] == step["d"][0]["action"], (dec, step["d"])
            tries, value = dec["tries"], dec["value"]
            assert sum(tries) == 2000 and tries[dec["action"]] == max(tries), dec
            for k in range(9):
                legal = (dec["legal"] >> k) & 1
                assert legal or (tries[k] == 0 and value[k] is None), dec
                assert not legal or (tries[k] > 0 and -1 <= value[k] <= 1), dec
        text = hunch("why", world, "--policy", pol, "--seed", "7").stdout
        assert "Nothing was replayed" in text and "the closest call: step" in text, text
        text = hunch("why", world, "--policy", pol, "--seed", "7", "--step", "2").stdout
        assert "the action most of its 2,000 playouts tried" in text, text
        assert "playouts   they collected" in text and "taken" in text, text


def test_train_learns_to_search():
    """By default a search learns by playing itself: a network trained on its
    games guides it. What train makes depends on the seed and the games, not
    on how many cores played them; the file is all that play needs, and one
    that does not fit is refused."""
    from hunch import show
    world = ROOT / "worlds" / "tictactoe.hunch"
    with tempfile.TemporaryDirectory() as d:
        one, two = Path(d) / "one.hpol", Path(d) / "two.hpol"
        tr = _train_vm(world, one, "--algo", "mcts", "--games", "256", "--cores", "1",
                       "--playouts", "300")
        _train_vm(world, two, "--algo", "mcts", "--games", "256", "--cores", "3",
                  "--playouts", "300")
        learned = tr["learned"]
        assert learned["games"] == 256 and learned["rounds"] == 4 and learned["cores"] == 1, tr
        assert learned["decisions"] > 256 * 4, tr
        a, b = one.read_text().splitlines(), two.read_text().splitlines()
        assert a[0] == "HUNCHSEARCH 2" and a[6] == "rollout 0.5" and a[7].startswith("guide "), a[:8]
        # only the time taken and the cores differ
        assert [l for l in a if not l.startswith("guide")] == \
            [l for l in b if not l.startswith("guide")], "the cores changed what was learned"
        said = show.how_trained(tr)
        assert "it learned by playing itself" in said and "self-play: 256 games, " in said, said
        assert "300 playouts a decision, as asked" in said, said

        run = hunch("run", world, "--policy", one, "--seed", "7")
        assert run.returncode == 0, run.stderr
        assert run.stdout == hunch("run", world, "--policy", one, "--seed", "7").stdout
        assert "guided by a network it trained by playing itself" in run.stdout, run.stdout
        why = hunch_json("why", world, "--policy", one, "--seed", "7", "--json")
        assert why["search"]["guide"]["games"] == 256, why["search"]
        for dec in why["decisions"]:
            assert sum(dec["tries"]) == 300 and dec["tries"][dec["action"]] == max(dec["tries"])
        text = hunch("why", world, "--policy", one, "--seed", "7").stdout
        assert "the network" in text and "Nothing was replayed" in text, text
        text = hunch("score", world, "--policy", one, "--score-sec", "1").stdout
        assert "guided by the network it trained by playing itself (256 games" in text, text

        # a file cut short, or with weights that are not numbers, is refused
        cut = Path(d) / "cut.hpol"
        cut.write_text("\n".join(a[:-3]) + "\n")
        bad = Path(d) / "bad.hpol"
        last = a[-1].split()
        bad.write_text("\n".join(a[:-1] + [" ".join(["nan"] + last[1:])]) + "\n")
        for pol in (cut, bad):
            out = hunch("run", world, "--policy", pol)
            assert out.returncode != 0 and "could not read policy" in out.stderr, out.stderr


def test_hidden_information_keeps_its_method():
    """A search reads the whole state, so `auto` picks it only on evidence
    that the agents are shown all of it."""
    from hunch import show
    with tempfile.TemporaryDirectory() as d:
        pol = Path(d) / "p.hpol"
        # Kuhn poker, kept out of the exact path: a card is hidden from each player.
        tr = _train_vm(ROOT / "worlds" / "kuhn.hunch", pol, "--algo", "auto",
                       "--exact-nodes", "0", "--budget", "20000")
        assert tr["algo"] == "regret" and tr["method"] == "sampled", tr
        assert tr["hidden"] == "found" and tr["hidden_check"]["conflicts"] > 0, tr
        assert pol.read_text().startswith("HUNCHTAB 1"), pol.read_text()[:40]
        # The market maker is shown measurements that never repeat: no evidence
        # either way, so it keeps the network.
        tr = _train_vm(ROOT / "worlds" / "market.hunch", pol, "--algo", "auto",
                       "--budget", "4000")
        assert tr["algo"] == "ppo" and tr["hidden"] == "unknown", tr
        assert tr["hidden_check"]["conflicts"] == 0, tr
        assert pol.read_text().startswith("HUNCHPOL 1"), pol.read_text()[:40]
        # Forced, the search is made anyway, and says what it looks at.
        tr = _train_vm(ROOT / "worlds" / "kuhn.hunch", pol, "--algo", "mcts", "--playouts", "200",
                       "--games", "0")
        assert tr["method"] == "mcts" and tr["chosen"] is False and tr["hidden"] == "found", tr
        said = show.how_trained(tr)
        assert "as asked (--algo mcts)" in said, said
        assert "looks at the whole state, including what its agent is not shown" in said, said
        text = hunch("score", ROOT / "worlds" / "kuhn.hunch", "--policy", pol,
                     "--score-sec", "0.5").stdout
        assert text.count("looks at the whole state, including what its agent is not shown") \
            == 2, text
        run = hunch("run", ROOT / "worlds" / "kuhn.hunch", "--policy", pol).stdout
        assert "including what its agent is not shown" in run, run
    # a policy file that names another world is refused, whatever its kind
    with tempfile.TemporaryDirectory() as d:
        pol = Path(d) / "other.hpol"
        pol.write_text("HUNCHSEARCH 1\nworld Elsewhere\nplayouts 100\nexplore 1\n")
        out = hunch("run", ROOT / "worlds" / "kuhn.hunch", "--policy", pol)
        assert out.returncode != 0 and "could not read policy" in out.stderr, out.stderr


def test_cli_test_shows_shape():
    # `test` first says what the compiler made of the file
    out = subprocess.run([str(ROOT / "bin" / "hunch"), "test",
                          str(ROOT / "worlds" / "cache.hunch")],
                         capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    assert "hunch #0" in out.stdout
    assert "17 observation dims" in out.stdout
    horizon = next(line for line in out.stdout.splitlines() if line.startswith("horizon"))
    assert horizon.split()[2:] == ["steps"], horizon
    # an array shown whole reads as one range, not 42 names
    out = subprocess.run([str(ROOT / "bin" / "hunch"), "test",
                          str(ROOT / "worlds" / "connect4.hunch")],
                         capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    assert "sees     board[0..41]\n" in out.stdout, out.stdout
    assert "1 of 1 assertion ran and held" in out.stdout, out.stdout
    # the commands that were folded away are gone
    for gone in ("check", "build", "bench"):
        out = subprocess.run([str(ROOT / "bin" / "hunch"), gone,
                              str(ROOT / "worlds" / "kuhn.hunch")],
                             capture_output=True, text=True, cwd=str(ROOT))
        assert out.returncode == 2 and "invalid choice" in out.stderr, out.stderr


def test_reference_page_worlds():
    """The complete worlds printed in llms.txt compile and train to what the
    page says of them, and the page's claims about labels and grids hold."""
    page = (ROOT / "llms.txt").read_text()
    blocks = [b for b in page.split("```") if b.lstrip().startswith("world ")]
    by_name = {b.split()[1]: b.lstrip() for b in blocks}
    assert {"Corridor", "Coin"} <= set(by_name), sorted(by_name)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "corridor.hunch"
        path.write_text(by_name["Corridor"])
        got = hunch_json("train", path, "--out", Path(d) / "c.hpol", "--json")
        assert abs(got["report"]["return"][0] - 0.92) < 1e-9, got["report"]["return"]

        path = Path(d) / "coin.hunch"
        path.write_text(by_name["Coin"])
        got = hunch_json("train", path, "--out", Path(d) / "k.hpol", "--json")
        alice, bob = got["report"]["return"]
        assert abs(bob - 0.5) < 2e-3 and abs(alice + 0.5) < 2e-3, got["report"]["return"]
        run = hunch("run", path, "--policy", Path(d) / "k.hpol").stdout
        assert "including what the agents are not shown" in run, run

        # an observation built in a local array is named by what was put in it
        path = Path(d) / "labels.hunch"
        path.write_text("world L\nhorizon 2\nstate { pile: int = 3; top: int = 1; shown: int[4]; }\n"
                        "step {\n  let obs: float[5];\n  obs[0] = pile;\n"
                        "  if top == 1 { obs[0] = top; }\n"
                        "  for i in 0..4 { obs[i + 1] = shown[i]; }\n"
                        "  let a = hunch p(obs, 2);\n"
                        "  let seen: float[4];\n  for i in 0..4 { seen[i] = shown[i] * 2; }\n"
                        "  let b = hunch q(seen, 2);\n  reward(p, a); reward(q, b);\n}\n")
        out = hunch("test", path).stdout
        # Two things put in one element are both named. Only a literal index
        # or the loop variable itself names an element: obs[i + 1] does not.
        assert "sees     pile | top, obs[1..4]\n" in out, out
        assert "sees     shown[0] * 2, shown[1] * 2, shown[2] * 2, shown[3] * 2\n" in out, out

        # `wide`: a board is printed as rows, in play and at the end of run
        bad = Path(d) / "bad.hunch"
        bad.write_text("world B\nstate { g: int[10] wide 3; }\nstep { let a = hunch p(g, 2); }\n")
        out = hunch("test", bad)
        assert out.returncode != 0 and "do not make rows of 3" in out.stderr, out.stderr
        bad.write_text("world Two Words\nstep { let a = hunch p([0], 2); }\n")
        out = hunch("test", bad)
        assert out.returncode != 0 and "a world's name is one word" in out.stderr, out.stderr
    run = hunch("run", ROOT / "worlds" / "tictactoe.hunch", "--policy", "none", "--seed", "3").stdout
    assert "x sees board, plays" in run and "\nboard at the end:\n" in run, run
    grid = run.split("board at the end:\n")[1].splitlines()[:3]
    assert all(len(row.split()) == 3 for row in grid), grid
    play = subprocess.run([str(ROOT / "bin" / "hunch"), "play", str(ROOT / "worlds" / "connect4.hunch"),
                           "--policy", "none", "--as", "first", "--seed", "1"],
                          input="3\nq\n", capture_output=True, text=True, cwd=str(ROOT))
    assert play.returncode == 0, play.stderr
    first, second = play.stdout.split("you see")[1:3]
    assert first.count(". . . . . . .") == 6, first
    # `up`: the disc dropped in column 3 is on the bottom row, which is printed last
    rows = [line.strip() for line in second.splitlines() if set(line.strip()) <= set(". 12") and line.strip()]
    assert len(rows) == 6 and rows[-1].split()[3] == "1" and rows[0] == ". . . . . . .", second


def main() -> int:
    print("language")
    for name, fn in [
        ("worlds compile", test_worlds_compile),
        ("syntax error location", test_syntax_error_has_location),
        ("unknown variable", test_unknown_variable),
        ("recursion rejected", test_recursion_rejected),
        ("inconsistent observation", test_inconsistent_obs_rejected),
        ("ambiguous reward", test_ambiguous_reward_rejected),
    ]:
        check(name, fn)

    print("runtime")
    for name, fn in [
        ("deterministic replay", test_deterministic_replay),
        ("episode isolation", test_snapshot_isolation),
        ("scripted actions", test_scripted_actions_are_exact),
        ("budget respected", test_budget_is_respected),
        ("fork measures all actions", test_fork_measures_all_actions),
    ]:
        check(name, fn)

    print("learning")
    for name, fn in [
        ("learning beats random", test_learning_beats_random),
        ("learning respects the mask", test_learning_respects_the_mask),
        ("self-play beats a folder", test_self_play_agent_beats_a_folder),
    ]:
        check(name, fn)

    print("language")
    for name, fn in [
        ("state block optional", test_state_block_optional),
        ("assert passes", test_assert_passes),
        ("assert fails with line", test_assert_fails_with_line),
        ("diagnostic shows source", test_diagnostic_shows_source),
        ("param override changes behaviour", test_param_override_changes_behaviour),
        ("param override bypasses cache", test_param_override_does_not_reuse_cache),
        ("cli test command", test_cli_test_command),
        ("cli test reports what ran", test_cli_test_reports_what_ran),
        ("masked actions never taken", test_masked_actions_are_never_taken),
        ("mask errors", test_mask_errors),
        ("all-illegal mask stops the run", test_all_illegal_mask_stops_the_run),
        ("unknown reward agent rejected", test_unknown_reward_agent_is_an_error),
        ("helper call in observation", test_helper_call_in_observation_is_harmless),
        ("wide observation runs", test_wide_observation_runs),
        ("twelve by twelve board", test_twelve_by_twelve_board_fits),
        ("array decay rejected", test_array_decay_is_an_error),
        ("duplicate definitions rejected", test_duplicate_definitions_are_errors),
        ("records run", test_records_run),
        ("scalar record runs", test_scalar_record_runs),
        ("record errors", test_record_errors),
        ("local array in observation", test_local_array_in_observation),
        ("local array errors", test_local_array_errors),
    ]:
        check(name, fn)

    print("answers")
    for name, fn in [
        ("named actions", test_named_actions),
        ("exact scoring matches a hand count", test_exact_scoring_matches_a_hand_count),
        ("train solves Kuhn poker", test_train_solves_kuhn),
        ("train looks ahead when nothing is hidden", test_train_looks_ahead_when_nothing_is_hidden),
        ("train proves a one-agent optimum", test_train_proves_a_one_agent_optimum),
        ("sampled regret closes the gap", test_sampled_regret_closes_the_gap),
        ("why replays each decision", test_why_replays_each_decision),
        ("play takes a seat", test_play_takes_a_seat),
        ("positions never reached are still played well",
         test_positions_never_reached_are_still_played_well),
        ("only state lasts between steps", test_only_state_lasts_between_steps),
        ("exact claims survive extremes", test_exact_claims_survive_extremes),
    ]:
        check(name, fn)

    print("search")
    for name, fn in [
        ("train makes a search for Connect Four", test_train_makes_a_search_for_connect_four),
        ("a search replays, plays and explains", test_search_replays_and_explains),
        ("train learns to search by self-play", test_train_learns_to_search),
        ("hidden information keeps its method", test_hidden_information_keeps_its_method),
    ]:
        check(name, fn)

    print("cli")
    check("test shows the shape", test_cli_test_shows_shape)
    check("the reference page's worlds and claims", test_reference_page_worlds)

    print()
    if FAILS:
        print(f"{len(FAILS)} failure(s), {PASSES} passed")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print(f"all {PASSES} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
