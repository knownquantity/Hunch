#!/usr/bin/env python3
"""Tests for the Python package: environments, forks, grading and bindings.

Run with:  python3 tests/test_bindings.py
The core needs only the standard library. The numpy, gymnasium and pettingzoo
checks are skipped when those are not installed.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hunch                                        # noqa: E402
from hunch import cli                               # noqa: E402

FAILS: list[str] = []
PASSES = 0
SKIPS = 0
TMP = Path(tempfile.mkdtemp(prefix="hunch-test-"))


class Skip(Exception):
    pass


def check(name: str, fn):
    global PASSES, SKIPS
    try:
        fn()
        PASSES += 1
        print(f"  ok   {name}")
    except Skip as exc:
        SKIPS += 1
        print(f"  skip {name}: {exc}")
    except AssertionError as exc:
        FAILS.append(f"{name}: {exc}")
        print(f"  FAIL {name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        FAILS.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")


def world(name: str) -> hunch.World:
    return hunch.World(ROOT / "worlds" / f"{name}.hunch")


def source(text: str, name: str = "w") -> hunch.World:
    path = TMP / f"{name}.hunch"
    path.write_text(text)
    return hunch.World(path)


_KUHN = None


def kuhn_solved():
    global _KUHN
    if _KUHN is None:
        w = world("kuhn")
        _KUHN = (w, w.train(out=TMP / "kuhn.hpol"))
    return _KUHN


def need(module: str):
    try:
        return __import__(module)
    except ImportError:
        raise Skip(f"{module} is not installed") from None


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

def test_env_plays_the_episode_hunch_run_plays():
    """An Env with one seat decided from outside sees, and is paid, exactly
    what `hunch run --agent seat --actions ...` shows for the same seed."""
    rng = random.Random(7)
    for name in ("kuhn", "leduc", "liars_dice", "tictactoe", "dungeon", "auction"):
        w = world(name)
        for seed in range(1, 9):
            seat = w.agents[seed % len(w.agents)]
            env = w.env(agents=[seat])
            env.reset(seed)
            mine = []
            while not env.over:
                assert env.agent == seat, "asked about a seat it does not decide"
                a = rng.choice(env.legal)
                mine.append(([round(v, 4) for v in env.obs], a))
                env.act(a)
            out = subprocess.run(
                [str(cli.VM_BIN), "trace", w.hbc, "--seed", str(seed), "--agent", seat,
                 "--actions", ",".join(str(a) for _, a in mine) or "0"],
                capture_output=True, text=True, check=True)
            ep = json.loads(out.stdout)["episodes"][0]
            k = w.agents.index(seat)
            theirs = [([round(v, 4) for v in d["obs"]], d["action"])
                      for st in ep["steps"] for d in st["d"] if d["agent"] == k]
            assert mine == theirs, f"{name} seed {seed}: decisions differ"
            for a in w.agents:
                paid = sum(st["r"][a] for st in ep["steps"])
                assert abs(paid - env.returns[a]) < 1e-4, f"{name} seed {seed}: {a} paid " \
                    f"{env.returns[a]}, run shows {paid}"


def test_env_names_and_legal_actions():
    w = world("kuhn")
    assert w.agents == ["alice", "bob"] and w.actions["alice"] == ["pass", "bet"]
    env = w.env()
    assert env.reset(3) and env.agent == "alice" and env.legal == [0, 1]
    assert "you may: pass, bet" in env.describe()
    env.act("bet")
    assert env.agent == "bob" and env.obs[1] == 1.0, "bob should be facing a bet"
    paid = env.act("pass")
    assert env.over and env.terminated and not env.truncated
    assert paid == {"alice": 1.0, "bob": -1.0} and env.returns == paid
    assert set(env.state) == {"card_a", "card_b", "phase"}
    t = world("tictactoe").env()
    t.reset(1)
    t.act(4)
    assert 4 not in t.legal and len(t.legal) == 8, "a taken square must not be legal"


def test_errors_are_raised_not_fatal():
    """An illegal action, a failed assert or a bad name raises; the process
    and the environment go on."""
    t = world("tictactoe").env()
    t.reset(1)
    t.act(4)
    try:
        t.act(4)
        raise AssertionError("an illegal action was accepted")
    except hunch.HunchRuntimeError as exc:
        assert "legal" in str(exc), exc
    assert not t.over and t.agent == "o", "the environment should still be waiting"
    t.act(0)
    try:
        t.act("middle")
        raise AssertionError("an unknown action name was accepted")
    except ValueError:
        pass
    w = source("world Trip\nhorizon 2\nstep {\n  let a = hunch p([1], [safe, trip]);\n"
               "  assert a == safe;\n  reward(1);\n}\n", "trip")
    env = w.env()
    env.reset(1)
    try:
        env.act("trip")
        raise AssertionError("a failed assert went unnoticed")
    except hunch.HunchRuntimeError as exc:
        assert "assert" in str(exc), exc
    # The world failed on the way to the next decision: this one still waits.
    assert env.agent == "p" and env.obs == [1.0] and not env.over and env.steps == 0
    assert env.act("safe") == {"p": 1.0}
    for bad in (5, -1, 1.5, True, None):
        try:
            env.act(bad)
            raise AssertionError(f"{bad!r} was accepted as an action")
        except ValueError:
            pass
    assert env.act("safe") == {"p": 1.0} and env.over
    # A snapshot belongs to its world.
    try:
        env.restore(t.snapshot())
        raise AssertionError("another world's snapshot was restored")
    except hunch.HunchRuntimeError:
        pass
    try:
        env.restore(b"garbage")
        raise AssertionError("garbage was restored")
    except hunch.HunchRuntimeError:
        pass


def test_snapshot_restore_and_reseed_fork():
    w = world("leduc")
    env = w.env()
    env.reset(11)
    env.act(env.legal[0])
    snap = env.snapshot()

    def finish(e):
        seen = []
        while not e.over:
            seen.append(tuple(e.obs))
            e.act(e.legal[-1])
        return seen, dict(e.returns)

    first = finish(env)
    env.restore(snap)
    assert finish(env) == first, "the same snapshot and actions gave a different future"
    other = w.env()
    other.restore(snap)
    assert finish(other) == first, "a snapshot restored in another Env differs"
    # The public card is still to come: other streams must be able to deal others.
    futures = set()
    for seed in range(1, 40):
        env.restore(snap)
        env.reseed(seed)
        futures.add(tuple(finish(env)[0]))
    assert len(futures) > 1, "reseed changed nothing that was still to be drawn"
    env.restore(snap)
    env.reseed(5)
    a = finish(env)
    env.restore(snap)
    env.reseed(5)
    assert finish(env) == a, "the same reseed gave two futures"


def test_playout_and_memory():
    """Random playouts from a fork average to the exact value of random play,
    and positions reached by different move orders have the same memory."""
    w = world("tictactoe")
    exact = w.score()["return"][0]                  # x under uniform play
    env = w.env()
    env.reset(1)
    snap, total, n = env.snapshot(), 0.0, 20000
    for i in range(n):
        env.restore(snap)
        env.reseed(i + 1)
        total += env.playout()["x"]
        assert env.over
    assert abs(total / n - exact) < 0.03, f"playouts average {total / n:.3f}, exact {exact:.3f}"
    keys = []
    for order in ((0, 4, 8), (8, 4, 0)):
        env.reset(1)
        for sq in order:
            env.act(sq)
        keys.append(env.memory())
    assert keys[0] == keys[1], "the same position has two memories"
    env.act(1)
    assert env.memory() != keys[0]


PAIR = """world Pair
horizon 1
step {
  let first = hunch p([0, 0], 2);
  let second = hunch p([1, first], 2);
  if first == 0 {
    if second == 1 { reward(1); } else { reward(-1); }
  } else {
    reward(0.5);
  }
  done();
}
"""


def test_two_decisions_in_one_step_are_two_positions():
    """Nothing in `state` changes between the two decisions of this step, so
    the key has to say where in the step each one is asked. A search that
    confused them took the safe 0.5 instead of the 1 that 0-then-1 pays."""
    w = source(PAIR, "pair")
    env = w.env()
    env.reset(1)
    first = env.memory()
    env.act(0)
    assert env.memory() != first, "two decisions of one step share a key"
    for seed in range(1, 6):
        env.reset(seed)
        search = hunch.Search(w, playouts=400, seed=seed)
        assert search.choose(env) == 0, f"search took the worse first move: {search.tried}"
        assert sum(n for n, _ in search.tried.values()) == 400
        env.act(0)
        assert search.choose(env) == 1


def test_policies_as_opponents():
    w, pol = kuhn_solved()
    assert pol.exact and pol.proven is not None and abs(pol.value["alice"] + 1 / 18) < 2e-3
    p = pol.probs("bob", [2, 1])                    # the king, facing a bet
    assert abs(sum(p) - 1) < 1e-9 and p[1] > 0.99, f"bob should call with the king: {p}"
    p = pol.probs("bob", [0, 1])
    assert p[0] > 0.99, f"bob should fold the jack: {p}"
    # A table policy in the other seat: bob never calls a bet with the jack.
    env = w.env(agents=["alice"], policy=pol)
    for seed in range(1, 200):
        env.reset(seed)
        env.act("bet")
        if env.state["card_b"] == 0:
            assert env.over and env.returns["alice"] == 1.0
    # A network policy file.
    d = world("duel")
    net = hunch.Policy(d, ROOT / "policies" / "duel.hpol")
    env = d.env(agents=["alice"], policy=net)
    env.reset(1)
    probs = net.probs("bob", [1.0, 0.6, 0.5])
    assert abs(sum(probs) - 1) < 1e-6 and all(v >= 0 for v in probs)
    while not env.over:
        env.act(env.legal[0])


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def test_grade_matches_a_hand_count():
    """Alice always bets against a bob who calls half the time: +1 when he
    folds, an even showdown for 2 when he calls, so 1/2 exactly. Bob's best
    reply calls with the king and queen and folds the jack: 1/3."""
    w = world("kuhn")
    g = w.grade({"alice": lambda obs, legal: "bet"})
    assert g.exact and abs(g.collects["alice"] - 0.5) < 1e-9, g.collects
    assert abs(g.reference["alice"] - 0.125) < 1e-9, g.reference
    assert g.sure["bob"] and abs(g.best_reply["bob"] - 1 / 3) < 1e-9, g.best_reply
    assert abs(g.gives_away("alice") - (1 / 3 + 0.125)) < 1e-9
    assert "alice collects +0.5" in str(g), str(g)
    # Probabilities by list, by name and from numbers all mean the same agent.
    g2 = w.grade({"alice": lambda obs, legal: [0.0, 1.0]})
    g3 = w.grade(lambda obs, legal: {"bet": 2})
    assert g2.collects == g.collects == g3.collects


def test_grade_finds_the_mistakes():
    w, pol = kuhn_solved()
    own = w.grade({"alice": lambda obs, legal: pol.probs("alice", obs, legal)}, against=pol)
    assert abs(own.collects["alice"] - pol.value["alice"]) < 1e-9
    assert own.gives_away("alice") < 2e-3, own.gives_away("alice")
    assert all(m["cost"] < 2e-3 for m in own.mistakes), own.mistakes

    def folds_the_king(obs, legal):                 # passes when facing a bet with the king
        if obs == [2, 1]:
            return "pass"
        return pol.probs("alice", obs, legal)
    g = w.grade({"alice": folds_the_king}, against=pol)
    fold = [m for m in g.mistakes if m["obs"] == [2, 1]]
    assert fold and fold[0]["chose"] == 0 and fold[0]["best"] == 1, g.mistakes
    assert g.collects["alice"] < own.collects["alice"] - 0.01
    assert abs((own.collects["alice"] - g.collects["alice"]) - fold[0]["cost"]) < 1e-6, \
        "the cost of the one changed decision should be the whole loss"
    # Passing first with the king is now a mistake too: she will fold it.
    assert any(m["obs"] == [2, 0] and m["best"] == 1 for m in g.mistakes), g.mistakes
    assert "pass" in str(g) and "was better" in str(g)
    try:
        w.grade({"alice": lambda obs, legal: 5})
        raise AssertionError("an action that does not exist was accepted")
    except (ValueError, hunch.HunchRuntimeError):
        pass


def test_grade_of_observations_that_are_not_whole_numbers():
    """A policy row is matched on what the agent is shown bit for bit, so the
    table a grade writes has to carry observations such as 1/3 exactly."""
    w = source("world Frac\nhorizon 1\nstate { card: int = 0; }\n"
               "init { card = randint(0, 2); }\n"
               "step {\n  let a = hunch p([card / 3.0], 3);\n"
               "  if a == card { reward(1); }\n  done();\n}\n", "frac")
    g = w.grade(lambda obs, legal: round(obs[0] * 3))
    assert g.exact and abs(g.collects["p"] - 1.0) < 1e-12, g.collects
    assert not g.mistakes and g.gives_away("p") is None
    wrong = w.grade(lambda obs, legal: 0)
    assert abs(wrong.collects["p"] - 1 / 3) < 1e-12 and len(wrong.mistakes) == 2
    assert abs(sum(m["cost"] for m in wrong.mistakes) - 2 / 3) < 1e-9


def test_grade_says_what_its_yardstick_is():
    w, pol = kuhn_solved()
    own = lambda obs, legal: pol.probs("alice", obs, legal)     # noqa: E731
    g = w.grade({"alice": own}, against=pol)
    assert g.reference_gap is not None and abs(g.reference_gap - pol.nashconv) < 1e-9
    assert abs(g.gives_away("alice")) <= g.reference_gap + 1e-9
    assert "give or take" in str(g)
    # Random play is no equilibrium: the figure is not printed as exploitability.
    rnd = w.grade({"alice": own})
    assert rnd.random_reference and rnd.reference_gap > 0.5
    assert "gives away" not in str(rnd).replace("what alice gives away", "")
    assert "random play" in str(rnd)


def test_grade_by_sampling_agrees():
    w = world("kuhn")
    g = w.grade({"alice": lambda obs, legal: "bet"}, episodes=4000, exact_nodes=1)
    assert not g.exact and g.episodes == 4000 and g.gives_away("alice") is None
    assert abs(g.collects["alice"] - 0.5) < 5 * g.sem["alice"], (g.collects, g.sem)
    assert "sampled" in str(g)


def test_situations_and_write_policy():
    w = world("kuhn")
    rows = w.situations("alice")
    assert len(rows) == 6 and all(r["agent"] == "alice" for r in rows)
    assert abs(sum(r["share"] for r in rows if r["obs"][1] == 0) - 1) < 1e-9, \
        "every episode opens with alice"
    table = [dict(r, probs=[0.0, 1.0]) for r in rows]
    pol = w.write_policy(TMP / "bets.hpol", table)
    assert pol.probs("alice", [1, 0]) == [0.0, 1.0]
    assert abs(w.score(pol)["return"][0] - 0.5) < 1e-9


def test_grade_command_reads_programs_and_answers():
    """`hunch grade` gives the same exact grade whether the agent is a program
    asked line by line or a file of written answers, in words or in JSON."""
    w, pol = kuhn_solved()
    hunch_cmd = [sys.executable, str(ROOT / "bin" / "hunch"), "grade", str(w.path),
                 "--agent", "alice", "--policy", str(pol.path), "--json"]
    want = w.grade({"alice": lambda obs, legal: "bet"}, against=pol)

    def run(*extra):
        out = subprocess.run(hunch_cmd + list(extra), capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        return out.stdout

    agent = f"{sys.executable} {ROOT / 'examples' / 'always_bet.py'}"
    got = json.loads(run("--cmd", agent))
    assert got["exact"] and abs(got["collects"]["alice"] - want.collects["alice"]) < 1e-9
    assert abs(got["gives_away"] - want.gives_away("alice")) < 1e-9
    qs = [json.loads(line) for line in run("--questions").splitlines()]
    assert len(qs) == 6 and qs[0]["legal"] == ["pass", "bet"] and "You are alice" in qs[0]["prompt"]
    assert "hunch train" not in qs[0]["prompt"], "a command line leaked into the rules"
    answers = TMP / "answers.jsonl"
    # In order: a sentence, JSON probabilities, a name, a number, a list.
    in_order = ["I would bet here.", '{"pass": 0, "bet": 1}', "bet", "1", "[0, 1]", "Bet!"]
    answers.write_text("\n".join(in_order) + "\n")
    got = json.loads(run("--answers", str(answers)))
    assert abs(got["collects"]["alice"] - want.collects["alice"]) < 1e-9
    # By id, in any order.
    by_id = [json.dumps({"id": k, "answer": a}) for k, a in enumerate(in_order)]
    answers.write_text("\n".join(reversed(by_id)) + "\n")
    got = json.loads(run("--answers", str(answers)))
    assert abs(got["collects"]["alice"] - want.collects["alice"]) < 1e-9

    def refused(lines, why):
        answers.write_text("\n".join(lines) + "\n")
        out = subprocess.run(hunch_cmd + ["--answers", str(answers)],
                             capture_output=True, text=True)
        assert out.returncode == 1 and why in out.stderr, (lines, out.stderr)
        assert "Traceback" not in out.stderr, out.stderr

    # Nothing is guessed: an answer that cannot be read stops the grade and
    # says which one.
    refused(["pass or bet, hard to say"] * 6, "could not read an action")
    refused(["bet"] * 5 + ["I will not bet"], "question 5")
    refused(["bet", "", "bet", "bet", "bet", "bet"], "line 2 is blank")
    refused(["bet"] * 7, "6 questions, 7 answers")
    refused(by_id + [by_id[0]], "a second answer to question 0")
    refused(by_id[:5] + ["bet"], "use one form throughout")
    refused(by_id[:5] + ['{"id": 5}'], 'no "answer"')
    refused(["bet"] * 5 + ["true"], "could not read an action")
    refused(["bet"] * 5 + ["1.9"], "could not read an action")
    refused(["bet"] * 5 + ["[0, Infinity]"], "finite")
    out = subprocess.run(hunch_cmd + ["--cmd", "true"], capture_output=True, text=True)
    assert out.returncode == 1 and "stopped before answering" in out.stderr, out.stderr


def test_search_plays_without_training():
    """Tree search over forks never loses tic-tac-toe to the solved policy as
    the first player, and beats random play at Connect Four."""
    t = world("tictactoe")
    pol = t.train(out=TMP / "ttt.hpol")
    env = t.env(agents=["x"], policy=pol)
    search = hunch.Search(t, playouts=600, seed=3)
    for seed in range(1, 11):
        env.reset(seed)
        before = env.snapshot()
        action = search.choose(env)
        assert env.snapshot() == before, "choose changed the environment it was asked about"
        assert sum(n for n, _ in search.tried.values()) == 600
        env.act(action)
        while not env.over:
            env.act(search.choose(env))
        assert env.returns["x"] >= 0, f"search lost as x on seed {seed}"
    c = world("connect4")
    env = c.env(agents=["first"])                   # second plays at random
    search = hunch.Search(c, playouts=150, seed=1)
    wins = 0
    for seed in range(1, 7):
        env.reset(seed)
        while not env.over:
            env.act(search.choose(env))
        wins += env.returns["first"] == 1
    assert wins >= 5, f"search won only {wins} of 6 against random play"


def train_vm(w: hunch.World, out: Path, *flags) -> dict:
    """`train` in the runtime itself, without the report `World.train` adds:
    scoring a search means playing it, and the tests below do that."""
    cli.ensure_vm()
    proc = subprocess.run([str(cli.VM_BIN), "train", str(w.hbc), "--out", str(out), "--json"]
                          + [str(f) for f in flags], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_trained_search_never_loses_tictactoe():
    """`train --algo mcts` at the playouts it chooses itself, against the
    exactly solved policy, in each seat."""
    t = world("tictactoe")
    solved = t.train(out=TMP / "ttt-solved.hpol")
    assert solved.method == "lookahead"
    tr = train_vm(t, TMP / "ttt-search.hpol", "--algo", "mcts", "--games", "0")
    assert tr["method"] == "mcts" and tr["hidden"] == "nothing", tr
    search = hunch.Policy(t, TMP / "ttt-search.hpol")
    rng = random.Random(5)
    for mine, theirs in (("x", "o"), ("o", "x")):
        env = t.env(agents=[theirs], policy=search)         # `mine` is the search's seat
        for seed in range(1, 25):
            env.reset(seed)
            while not env.over:
                env.act(solved.act(theirs, env.obs, env.legal, rng))
            assert env.returns[mine] >= 0, \
                f"the search ({tr['playouts']} playouts) lost as {mine} on seed {seed}"
    # The same instant gets the same reply: a fork that is not acted on
    # differently has the same future.
    env = t.env(agents=["x"], policy=search)
    env.reset(4)
    snap = env.snapshot()
    env.act(env.legal[0])
    first = env.memory()
    env.restore(snap)
    env.act(env.legal[0])
    assert env.memory() == first, "the search answered one instant in two ways"
    # A search chooses from an instant, not from an observation.
    try:
        search.probs("x", [0] * 9)
        raise AssertionError("probs of a search did not raise")
    except hunch.HunchRuntimeError as exc:
        assert "search" in str(exc), exc
    # World.train reports it the way the command does.
    got = t.train(algo="mcts", out=TMP / "ttt-quick.hpol", playouts=300, score_sec=0.5, games=0)
    assert got.method == "mcts" and not got.exact and got.nashconv is None, got.train
    assert "the policy is a search" in str(got) and "300 playouts a decision" in str(got), str(got)


def test_trained_search_wins_connect_four():
    """`train` on Connect Four picks the search by itself, and it beats random
    play from either seat. (--exact-nodes only makes "too big" quick to find
    out; the choice and the playouts are train's own.)"""
    c = world("connect4")
    tr = train_vm(c, TMP / "c4.hpol", "--algo", "auto", "--exact-nodes", "50000", "--games", "0")
    assert tr["method"] == "mcts" and tr["chosen"] is True and tr["hidden"] == "nothing", tr
    rng = random.Random(11)
    for mine, theirs in (("first", "second"), ("second", "first")):
        env = c.env(agents=[theirs], policy=TMP / "c4.hpol")
        wins = 0
        for seed in range(1, 11):
            env.reset(seed)
            while not env.over:
                env.act(rng.choice(env.legal))
            wins += env.returns[mine] == 1
        assert wins >= 9, (f"the search ({tr['playouts']} playouts) won only {wins} of 10 "
                           f"as {mine} against random play")


def test_learned_search_beats_random_playouts():
    """At 100 playouts a decision the search with random playouts loses
    tic-tac-toe as o to the solved policy; after 512 games of self-play the
    network-guided search, played through the library with no Python in the
    loop, loses no game in either seat."""
    t = world("tictactoe")
    solved = t.train(out=TMP / "ttt-solved.hpol")
    plain = train_vm(t, TMP / "ttt-plain.hpol", "--algo", "mcts", "--games", "0",
                     "--playouts", "100")
    got = t.train(algo="mcts", out=TMP / "ttt-learned.hpol", playouts=100, games=512, cores=2,
                  score_sec=0.5)
    assert plain["learned"] is None and got.train["learned"]["games"] == 512, got.train
    assert "it learned by playing itself" in str(got), str(got)

    def lost(policy, mine, theirs):
        env = t.env(agents=[theirs], policy=policy)
        rng = random.Random(5)
        n = 0
        for seed in range(1, 25):
            env.reset(seed)
            while not env.over:
                env.act(solved.act(theirs, env.obs, env.legal, rng))
            n += env.returns[mine] < 0
        return n

    assert lost(TMP / "ttt-plain.hpol", "o", "x") > 0, "random playouts no longer lose here"
    for mine, theirs in (("x", "o"), ("o", "x")):
        assert lost(TMP / "ttt-learned.hpol", mine, theirs) == 0, f"the learned search lost as {mine}"
    # The same instant gets the same reply.
    env = t.env(agents=["x"], policy=TMP / "ttt-learned.hpol")
    env.reset(4)
    snap = env.snapshot()
    env.act(env.legal[0])
    first = env.memory()
    env.restore(snap)
    env.act(env.legal[0])
    assert env.memory() == first, "the learned search answered one instant in two ways"


# ---------------------------------------------------------------------------
# numpy, gymnasium, pettingzoo
# ---------------------------------------------------------------------------

def test_vec_matches_the_exact_value():
    np = need("numpy")
    w = world("kuhn")
    exact = w.grade({"alice": lambda obs, legal: "bet"}).collects["alice"]
    envs = w.vec(4096, agent="alice")
    obs, legal = envs.reset(seed=3)
    assert obs.shape == (4096, 2) and legal.shape == (4096, 2) and legal.all()
    total, done = 0.0, 0
    for _ in range(40):
        obs, reward, over, legal = envs.step(np.ones(4096, np.int32))
        total += float(reward.sum())
        done += int((over != 0).sum())
    assert done > 100_000
    assert abs(total / done - exact) < 0.02, f"vec mean {total / done:.4f}, exact {exact:.4f}"
    # The arrays are the caller's, and a refused action steps no row.
    t = world("tictactoe").vec(3)
    first, _ = t.reset(seed=1)
    kept = first.copy()
    obs, _, _, legal = t.step(np.full(3, 4, np.int32))
    assert (first == kept).all() and obs is not first
    try:
        t.step(np.array([0, 1, 4], np.int32))
        raise AssertionError("an illegal action in the last row was accepted")
    except hunch.HunchRuntimeError as exc:
        assert "slot 2" in str(exc), exc
    assert (t.obs == obs).all(), "rows before the refused one were stepped"
    t.step(np.array([0, 1, 5], np.int32))
    try:
        w.vec(0)
        raise AssertionError("vec(0) was accepted")
    except ValueError:
        pass
    both = w.vec(64)                                # every agent decided from outside
    obs, legal = both.reset(seed=1)
    assert set(both.agent.tolist()) == {0}, "alice opens every episode"
    obs, reward, over, legal = both.step(np.zeros(64, np.int32))
    assert reward.shape == (64, 2) and set(both.agent.tolist()) == {1}


def test_gymnasium_env():
    need("numpy")
    need("gymnasium")
    from gymnasium.utils.env_checker import check_env
    env = world("dungeon").gym()
    check_env(env, skip_render_check=True)
    w, pol = kuhn_solved()
    env = w.gym("alice", policy=pol)
    total = 0.0
    for seed in range(300):
        obs, info = env.reset(seed=seed)
        assert info["action_mask"].tolist() == [1, 1] and env.action_masks().all()
        over = False
        while not over:
            obs, reward, term, trunc, info = env.step(1)
            total += reward
            over = term or trunc
    exact = w.grade({"alice": lambda o, legal: "bet"}, against=pol).collects["alice"]
    assert abs(total / 300 - exact) < 0.35, f"gym mean {total / 300:.3f}, exact {exact:.3f}"


def test_pettingzoo_env():
    need("numpy")
    need("gymnasium")
    need("pettingzoo")
    from pettingzoo.test import api_test
    for name in ("kuhn", "tictactoe", "rps"):
        api_test(world(name).pettingzoo(), num_cycles=200, verbose_progress=False)
    # What is paid before anyone decides still reaches the agents.
    late = source("world Late\nhorizon 3\nstate { t: int = 0; }\nstep {\n  t = t + 1;\n"
                  "  if t == 1 { reward(a, 5); reward(b, 7); }\n"
                  "  if t == 2 { let x = hunch a([t], 2); reward(a, x); }\n"
                  "  if t == 3 { let y = hunch b([t], 2); reward(b, y); done(); }\n}\n", "late")
    api_test(late.pettingzoo(), num_cycles=50, verbose_progress=False)
    env, totals = late.pettingzoo(), {"a": 0.0, "b": 0.0}
    env.reset(seed=1)
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        totals[agent] += reward
        env.step(None if term or trunc else 1)
    assert totals == {"a": 6.0, "b": 8.0}, totals
    g = late.gym("a")
    g.reset(seed=1)
    assert g.step(1)[1] == 6.0
    env = world("tictactoe").pettingzoo()
    env.reset(seed=1)
    seen = 0
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
        mask = obs["action_mask"] if isinstance(obs, dict) else info["action_mask"]
        env.step(int(mask.argmax()))
        seen += 1
    assert seen >= 5


def main() -> int:
    groups = [
        ("environments", [
            ("Env plays the episode `hunch run` plays", test_env_plays_the_episode_hunch_run_plays),
            ("names and legal actions", test_env_names_and_legal_actions),
            ("errors are raised, not fatal", test_errors_are_raised_not_fatal),
            ("snapshot, restore and reseed fork", test_snapshot_restore_and_reseed_fork),
            ("playout and memory", test_playout_and_memory),
            ("two decisions in one step are two positions",
             test_two_decisions_in_one_step_are_two_positions),
            ("table and network policies as opponents", test_policies_as_opponents),
        ]),
        ("grading", [
            ("grade matches a hand count", test_grade_matches_a_hand_count),
            ("grade finds the mistakes", test_grade_finds_the_mistakes),
            ("grade of observations that are not whole numbers",
             test_grade_of_observations_that_are_not_whole_numbers),
            ("grade says what its yardstick is", test_grade_says_what_its_yardstick_is),
            ("grade by sampling agrees", test_grade_by_sampling_agrees),
            ("situations and write_policy", test_situations_and_write_policy),
            ("the grade command reads programs and answers",
             test_grade_command_reads_programs_and_answers),
        ]),
        ("search", [
            ("search plays without training", test_search_plays_without_training),
            ("a trained search never loses tic-tac-toe", test_trained_search_never_loses_tictactoe),
            ("a trained search wins Connect Four", test_trained_search_wins_connect_four),
            ("a learned search beats random playouts", test_learned_search_beats_random_playouts),
        ]),
        ("bindings", [
            ("vec matches the exact value", test_vec_matches_the_exact_value),
            ("gymnasium", test_gymnasium_env),
            ("pettingzoo", test_pettingzoo_env),
        ]),
    ]
    for title, tests in groups:
        print(title)
        for name, fn in tests:
            check(name, fn)
        print()
    if FAILS:
        print(f"{len(FAILS)} failed, {PASSES} passed, {SKIPS} skipped")
        for f in FAILS:
            print("  " + f)
        return 1
    print(f"all {PASSES} tests passed" + (f" ({SKIPS} skipped)" if SKIPS else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
