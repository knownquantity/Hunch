"""A Hunch world as a Python object.

    import hunch
    world = hunch.World("worlds/kuhn.hunch")
    solved = world.train()                    # the exact answer, when the world is small enough
    env = world.env(agents=["alice"], policy=solved)      # bob plays the solved policy
    world.grade({"alice": my_agent}, against=solved)      # what any agent collects, exactly

The world is stepped by the same C runtime the commands use, loaded as a
shared library, so an environment here plays exactly the episodes `hunch run`
plays. Nothing but the standard library is needed for this module; numpy,
gymnasium and pettingzoo are imported only by the methods that return their
objects.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import operator
import os
import re
import select
import subprocess
import time
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from . import cli, show

MAX_OBS = 64
MAX_AGENTS = 8
MAX_ACT = 32
_U64 = (1 << 64) - 1


class HunchRuntimeError(RuntimeError):
    """A world broke one of its own rules while running, or was given an action
    it does not allow."""


_LIB = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL(str(cli.ensure_lib()))
    vp, i32, u32, u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64
    dp, cp, lng = ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_long
    sigs = {
        "hunch_world_open": (vp, [cp, cp, i32]),
        "hunch_world_close": (None, [vp]),
        "hunch_env_new": (vp, [vp, i32, u32, cp, i32, cp, i32]),
        "hunch_env_free": (None, [vp]),
        "hunch_env_error": (cp, [vp]),
        "hunch_env_set_random": (None, [vp, u32]),
        "hunch_env_reset": (i32, [vp, i32, u64]),
        "hunch_env_act": (i32, [vp, i32, i32]),
        "hunch_env_agent": (i32, [vp, i32]),
        "hunch_env_legal": (u32, [vp, i32]),
        "hunch_env_obs": (i32, [vp, i32, dp]),
        "hunch_env_over": (i32, [vp, i32]),
        "hunch_env_rewards": (None, [vp, i32, dp, i32]),
        "hunch_env_state": (i32, [vp, i32, dp]),
        "hunch_env_steps": (lng, [vp, i32]),
        "hunch_env_snapshot_size": (i32, []),
        "hunch_env_save": (None, [vp, i32, vp]),
        "hunch_env_load": (None, [vp, i32, vp]),
        "hunch_env_reseed": (None, [vp, i32, u64]),
        "hunch_env_memory": (i32, [vp, i32, dp]),
        "hunch_env_key": (u64, [vp, i32]),
        "hunch_env_playout": (i32, [vp, i32, dp]),
        "hunch_env_policy_probs": (i32, [vp, i32, dp, u32, dp]),
        "hunch_vec_reset": (i32, [vp, u64, vp, i32, vp, vp]),
        "hunch_vec_step": (i32, [vp, vp, vp, i32, vp, vp, vp, vp]),
        "hunch_vec_random": (lng, [vp, lng, u64]),
    }
    for name, (res, args) in sigs.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    _LIB = lib
    return lib


def _legal_list(mask: int, n: int) -> List[int]:
    return [a for a in range(n) if (mask >> a) & 1]


class Policy:
    """A policy file for a world: what `hunch train` saves. A table (one row
    for everything an agent can be shown), a small network, or a search (for
    a world too big to follow whole in which nothing is hidden: it plays in an
    `Env`, and has no `probs` for an observation)."""

    def __init__(self, world: "World", path: Union[str, Path]):
        self.world = world
        self.path = Path(path)
        self._env = None

    def __repr__(self) -> str:
        return f"<hunch.Policy {self.path} for {self.world.name}>"

    def __fspath__(self) -> str:
        return str(self.path)

    def probs(self, agent: str, obs: Sequence[float], legal: Optional[Sequence[int]] = None
              ) -> List[float]:
        """How likely the policy is to play each action where `agent` is shown
        `obs`. `legal` lists the actions allowed there (default: all)."""
        w = self.world
        if self._env is None:
            self._env = w.env(agents=[], policy=self)
        k, n = w.agents.index(agent), w.n_actions[agent]
        if len(obs) != w.obs_dim[agent]:
            raise ValueError(f"{agent} is shown {w.obs_dim[agent]} values, got {len(obs)}")
        mask = 0
        for a in (range(n) if legal is None else legal):
            mask |= 1 << w.action_index(agent, a)
        out = (ctypes.c_double * MAX_ACT)()
        buf = (ctypes.c_double * MAX_OBS)(*[float(v) for v in obs])
        got = w._lib.hunch_env_policy_probs(self._env._p, k, buf, mask, out)
        if got == -2:
            self._env._fail()                       # a search: the runtime says why not
        if got < 0:
            raise ValueError("no legal action")
        return list(out[:n])

    def act(self, agent: str, obs: Sequence[float], legal: Optional[Sequence[int]] = None,
            rng=None) -> int:
        """An action drawn from the policy."""
        import random
        p = self.probs(agent, obs, legal)
        u, acc, last = (rng or random).random(), 0.0, 0
        for a, pa in enumerate(p):
            if pa > 0:
                last = a
                acc += pa
                if u < acc:
                    return a
        return last


class Trained(Policy):
    """What `World.train` returns: the saved policy, how it was found, and its
    report. Prints as `hunch train` prints."""

    def __init__(self, world: "World", path, train: dict, report: dict):
        super().__init__(world, path)
        self.train, self.report = train, report
        self.method: str = train.get("method") or train.get("algo")
        self.exact: bool = bool(report.get("exact"))
        self.value: Dict[str, float] = dict(zip(world.agents, report["return"]))
        # How much the agents could gain between them by each changing alone;
        # None when the world was too big to say.
        self.nashconv: Optional[float] = (
            sum(max(0.0, b - r) for b, r in zip(report["best_reply"], report["return"]))
            if self.exact else None)
        self.proven: bool = self.exact and all(report.get("sure", [False]))

    def __str__(self) -> str:
        names = show.Names(self.world.unit)
        return "\n".join([f"{self.world.name}: {show.how_trained(self.train)}"]
                         + show.report_lines(self.report, names))

    def __repr__(self) -> str:
        vals = ", ".join(f"{a} {show.num(v)}" for a, v in self.value.items())
        gap = "" if self.nashconv is None else f", NashConv {self.nashconv:.4f}"
        return f"<hunch.Trained {self.world.name} by {self.method}: {vals}{gap}>"


class Grade:
    """What `World.grade` returns. Attributes:

    exact       True when every outcome was weighed and nothing sampled
    collects    {agent: what it collects per episode}, with the graded agents
                in their seats and the reference policy in the others
    reference   {agent: what it collects when every seat plays the reference:
                the `against` policy, or uniform random play without one}
    reference_gap   how far the reference is from an equilibrium: the sum of
                what each agent could gain by replying to it as well as it can
                (its NashConv). None when sampled, when there is one agent, or
                when a best reply to it could not be proven
    best_reply  {agent: what the best reply by that agent to everyone else
                collects}; for a graded agent's opponent this is how hard the
                graded agent can be punished (exact only)
    sure        {agent: whether that best reply is proven best}
    mistakes    the graded agents' decisions that cost most, worst first: each
                a dict with agent, obs, probs (what it played), chose (its most
                likely action), best, share (how many times an episode comes
                to it), value (what each action goes on to collect, everything
                else as it is) and cost: share times what one visit loses by
                not choosing `best`. Where an episode comes to a situation at
                most once, cost is what always choosing `best` there would
                add; where it comes back to the same situation, visits are
                changed one at a time and the costs need not add up
    sem         {agent: standard error} when sampled
    """

    def __init__(self, world, exact, collects, reference, best_reply, sure, mistakes, sem,
                 graded, episodes=0, reference_gap=None, random_reference=False):
        self.world, self.exact, self.collects, self.reference = world, exact, collects, reference
        self.best_reply, self.sure, self.mistakes, self.sem = best_reply, sure, mistakes, sem
        self.graded, self.episodes = graded, episodes
        self.reference_gap, self.random_reference = reference_gap, random_reference

    def gives_away(self, agent: str) -> Optional[float]:
        """In a world of two agents whose rewards sum to zero: what the best
        reply to `agent` collects, less what that seat collects in the
        reference. When the reference is an equilibrium this is how exploitable
        `agent` is, and zero means it cannot be beaten. It is off by at most
        `reference_gap`, so against a reference that is far from an equilibrium
        (random play, say) it says little. None when the grade is sampled, the
        world is not two-agent zero-sum, or the reference's gap is unknown."""
        others = [a for a in self.world.agents if a != agent]
        if not self.exact or len(others) != 1 or self.reference_gap is None:
            return None
        if abs(sum(self.collects.values())) > 1e-9 or abs(sum(self.reference.values())) > 1e-9:
            return None
        return self.best_reply[others[0]] - self.reference[others[0]]

    def __str__(self) -> str:
        w = self.world
        how = ("every outcome weighed, nothing sampled" if self.exact
               else f"{self.episodes:,} sampled episodes, ± is one standard error")
        out = [f"{w.name}: graded {', '.join(self.graded)} ({how})"]
        for a in self.graded:
            err = "" if self.exact else f" ± {self.sem[a]:.4f}"
            ref = "random play" if self.random_reference else "the reference policy"
            out.append(f"  {a} collects {show.num(self.collects[a])}{err} per episode; "
                       f"{ref} collects {show.num(self.reference[a])} there")
            if self.exact:
                proof = "" if self.sure[a] else " (found by search, not proven the most)"
                short = max(0.0, self.best_reply[a] - self.collects[a])
                rest = " against the others" if len(w.agents) > 1 else ""
                out.append(f"    the most {a} could collect{rest} is "
                           f"{show.num(self.best_reply[a])}: it falls {short:.4f} short{proof}")
            gap = self.gives_away(a)
            if gap is not None and not self.random_reference:
                other = [x for x in w.agents if x != a][0]
                proof = "" if self.sure[other] else " (found by search, not proven the most)"
                out.append(f"    the best reply by {other} collects "
                           f"{show.num(self.best_reply[other])}: {a} gives away "
                           f"{max(0.0, gap):.4f}{proof}")
                if self.reference_gap >= 5e-5:
                    out.append(f"      (give or take {self.reference_gap:.4f}: how far the "
                               "reference policy is from an equilibrium)")
            elif self.exact and len(w.agents) > 1:
                for other in (x for x in w.agents if x != a):
                    proof = "" if self.sure[other] else " (found by search, not proven the most)"
                    out.append(f"    the best reply by {other} to the others collects "
                               f"{show.num(self.best_reply[other])}{proof}")
                if gap is not None:
                    out.append(f"      (to see what {a} gives away, grade against a trained "
                               "policy: random play is no yardstick)")
        shown = [m for m in self.mistakes if m["cost"] >= 5e-5][:5]
        if shown:
            out.append("  decisions that cost most (per episode):")
            for m in shown:
                acts = w.actions[m["agent"]]
                if m["probs"][m["chose"]] > 0.999:
                    did = f"chose {acts[m['chose']]}"
                else:
                    did = "played " + ", ".join(f"{acts[a]} {100 * pa:.0f}%"
                                                for a, pa in enumerate(m["probs"]) if pa > 0.0005)
                seen = w.describe(m["agent"], m["obs"], []).rsplit("; you may:", 1)[0]
                out.append(f"    {m['cost']:.4f}  {m['agent']} shown {seen}: {did}; "
                           f"{acts[m['best']]} was better")
            if self.exact and not all(self.sure[m["agent"]] for m in shown):
                out.append("    (an agent here is shown the same thing at more than one point, "
                           "so a cost is for changing one visit and leaving the others)")
            gap = self.reference_gap
            if gap and not self.random_reference and any(m["cost"] < gap for m in shown):
                out.append(f"    (a cost under {gap:.4f} is within the reference's own distance "
                           "from an equilibrium: train with a smaller --target to resolve it)")
        elif self.exact:
            out.append("  no decision costs as much as 0.0001"
                       + (" against these opponents" if len(w.agents) > 1 else ""))
        return "\n".join(out)

    __repr__ = __str__


class World:
    """A compiled world: its shape, its environments and its answers.

    agents      the names given to `hunch` blanks, in order
    actions     {agent: action names} ("0", "1", ... when the world names none)
    sees        {agent: the source text of each observation value}
    obs_dim, n_actions, horizon, rules (the comment at the top of the file)
    """

    def __init__(self, path: Union[str, Path], params: Optional[dict] = None):
        self.path = Path(path)
        self.params = dict(params or {})
        self.hbc, self.unit = cli.compile_file(self.path, self.params or None)
        u = self.unit
        self.name: str = u["world"]
        self.agents: List[str] = list(u["agents"])
        self.obs_dim: Dict[str, int] = {a: u["agent_obs"].get(a, 1) for a in self.agents}
        self.n_actions: Dict[str, int] = {a: u["agent_acts"].get(a, 2) for a in self.agents}
        named = u.get("agent_action_names", {})
        self.actions: Dict[str, List[str]] = {
            a: list(named.get(a) or [str(i) for i in range(self.n_actions[a])])
            for a in self.agents}
        self.sees: Dict[str, List[str]] = {
            a: list(u.get("agent_obs_labels", {}).get(a, [])) for a in self.agents}
        self.horizon: int = u["horizon"]
        self.n_slots: int = u["nslots"]
        # name -> (first slot, length) for everything declared in `state`
        self.fields: Dict[str, tuple] = {d[0]: (d[1], d[2]) for d in u["decls"]}
        lines = []
        for line in self.path.read_text().splitlines():
            if not line.startswith("//"):
                break
            text = line[2:].lstrip(" ") if not line.startswith("//  ") else line[3:]
            # A line that shows a command to run is for the reader of the file,
            # not part of the rules.
            if not re.match(r"\s*(Try:\s*)?(\./bin/)?hunch\s", text):
                lines.append(text)
        self.rules: str = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        # Marks a snapshot as this world's: the bytecode is the rules.
        self._digest = hashlib.sha256(Path(self.hbc).read_bytes()).digest()[:8]
        self._lib = _lib()
        err = ctypes.create_string_buffer(320)
        self._w = self._lib.hunch_world_open(str(self.hbc).encode(), err, 320)
        if not self._w:
            raise HunchRuntimeError(err.value.decode())

    def __repr__(self) -> str:
        return (f"<hunch.World {self.name}: agents {', '.join(self.agents) or 'none'}, "
                f"horizon {self.horizon}>")

    # ---- names ----

    def action_index(self, agent: str, action) -> int:
        """An action by number or by the name the world gives it."""
        if isinstance(action, str):
            try:
                return self.actions[agent].index(action)
            except ValueError:
                raise ValueError(f"'{action}' is not an action of {agent}; it has "
                                 + ", ".join(self.actions[agent])) from None
        try:
            if isinstance(action, bool):
                raise TypeError
            k = operator.index(action)
        except TypeError:
            raise ValueError(f"{action!r} is not an action: give a whole number from 0 to "
                             f"{self.n_actions[agent] - 1}, or a name") from None
        if not 0 <= k < self.n_actions[agent]:
            raise ValueError(f"{agent} has actions 0 to {self.n_actions[agent] - 1}, got {k}")
        return k

    def describe(self, agent: str, obs: Sequence[float], legal: Optional[Sequence[int]] = None
                 ) -> str:
        """What `agent` is shown and may do, in the world's own words:
        "card_a = 1, facing_bet = 0; you may: pass, bet"."""
        labels = self.sees[agent]
        if len(labels) == len(obs):
            seen = ", ".join(f"{lab} = {show.plain(v)}" for lab, v in zip(labels, obs))
        else:
            seen = "[" + ", ".join(show.plain(v) for v in obs) + "]"
        acts = self.actions[agent]
        can = range(len(acts)) if legal is None else legal
        return f"{seen}; you may: " + ", ".join(acts[a] for a in can)

    def prompt(self, agent: str, obs: Sequence[float], legal: Optional[Sequence[int]] = None
               ) -> str:
        """A question for a language model in `agent`'s seat: the rules (the
        comment at the top of the world's file), what it is shown, and the
        actions to answer with."""
        acts = self.actions[agent]
        can = [acts[a] for a in (range(len(acts)) if legal is None else legal)]
        how = "Answer with one action name."
        if len(can) > 1:
            mix = json.dumps({can[0]: 0.7, can[1]: 0.3})
            how += f" To randomise, answer with JSON probabilities instead, such as {mix}."
        return (f"{self.rules}\n\nYou are {agent}. You see: "
                f"{self.describe(agent, obs, legal)}.\n{how}")

    def question(self, agent: str, obs: Sequence[float], legal: Optional[Sequence[int]] = None
                 ) -> dict:
        """One decision as a JSON-ready question: what `hunch grade` sends a
        program and what `questions` lists."""
        acts = self.actions[agent]
        legal = list(range(len(acts)) if legal is None else legal)
        labels = self.sees[agent]
        q = {"agent": agent, "obs": [float(v) for v in obs], "legal": [acts[a] for a in legal],
             "prompt": self.prompt(agent, obs, legal)}
        if len(labels) == len(obs):
            q["sees"] = dict(zip(labels, q["obs"]))
        return q

    def questions(self, agent: Optional[str] = None, exact_nodes: Optional[int] = None
                  ) -> List[dict]:
        """Every decision `agent` (default: the first) can face, as questions:
        dicts with id, agent, obs, sees, legal (action names) and prompt. Ask
        them of anything, in any order or all at once, and hand the answers to
        `grade_answers`. Raises ValueError for a world too big to list."""
        agent = agent or self.agents[0]
        out = []
        for r in self.situations(agent, exact_nodes=exact_nodes):
            out.append(dict(self.question(agent, r["obs"], r["legal"]), id=len(out)))
        return out

    def parse_answer(self, agent: str, answer, legal: Sequence[int]):
        """What an outside agent said, as something `grade` takes. Accepts an
        action number or name, a JSON list or object of probabilities, or a
        plain sentence that names exactly one action ("I will bet."). A
        sentence that names two actions, an illegal one, or says "not" is
        refused instead of guessed at."""
        acts = self.actions[agent]

        def refuse(why: str = ""):
            raise HunchRuntimeError(
                f"could not read an action for {agent} from {str(answer).strip()[:80]!r}{why}; "
                f"expected one of: {', '.join(acts[a] for a in legal)}")
        if answer is None or isinstance(answer, (bool, float)):
            refuse()
        if not isinstance(answer, str):
            return answer                           # a number, a list or a dict
        text = answer.strip()
        if text in acts:
            return text
        try:
            got = json.loads(text)
        except ValueError:
            pass
        else:
            if isinstance(got, dict) and set(got) & {"answer", "action", "probs"}:
                inner = [got[k] for k in ("answer", "action", "probs") if k in got][0]
                return self.parse_answer(agent, inner, legal)
            if got is None or isinstance(got, (bool, float)):
                refuse()
            if not isinstance(got, str):
                return got
            text = got.strip()
            if text in acts:
                return text
        if all(a.isdigit() for a in acts):
            refuse(" (this world's actions are numbers: answer with the number alone)")
        low = text.lower()
        named = [a for a in range(len(acts))
                 if re.search(r"(?<![A-Za-z0-9_-])" + re.escape(acts[a].lower())
                              + r"(?![A-Za-z0-9_-])", low)]
        if len(named) != 1:
            refuse()
        if named[0] not in legal:
            refuse(f" ({acts[named[0]]} is not legal there)")
        if re.search(r"\b(not|never|no|neither|nor|without|instead|rather|cannot)\b|n['’]t", low):
            refuse(" (it reads as a refusal; answer with the action name alone)")
        return named[0]

    def grade_answers(self, answers, agent: Optional[str] = None, against=None,
                      exact_nodes: Optional[int] = None) -> Grade:
        """Grade the answers to `questions(agent)`: a list in the same order,
        or a dict by question id. Each answer is anything `parse_answer` reads.
        Every question needs one answer and there may be no others."""
        agent = agent or self.agents[0]
        qs = self.questions(agent, exact_nodes)
        if not isinstance(answers, dict):
            if len(answers) != len(qs):
                raise ValueError(f"{len(qs)} questions, {len(answers)} answers")
            answers = dict(enumerate(answers))
        missing = [q["id"] for q in qs if q["id"] not in answers]
        if missing:
            raise ValueError(f"no answer to question {missing[0]}"
                             + (f" and {len(missing) - 1} more" if len(missing) > 1 else ""))
        extra = sorted(set(answers) - {q["id"] for q in qs}, key=str)
        if extra:
            raise ValueError(f"an answer to question {extra[0]!r}, but the questions are "
                             f"0 to {len(qs) - 1}")
        by_obs = {(tuple(q["obs"]), tuple(q["legal"])): q["id"] for q in qs}
        acts = self.actions[agent]

        def agent_fn(obs, legal):
            qid = by_obs[(tuple(float(v) for v in obs), tuple(acts[a] for a in legal))]
            try:
                return self._as_probs(agent, self.parse_answer(agent, answers[qid], legal), legal)
            except (ValueError, HunchRuntimeError) as exc:
                raise type(exc)(f"question {qid}: {exc}") from None
        return self.grade({agent: agent_fn}, against, exact_nodes=exact_nodes)

    def program(self, agent: str, command: str, timeout: float = 120.0) -> Callable:
        """An agent function that asks a program: it is started once and sent
        one JSON question per line (see `question`). It must print one line
        back for each as it reads it, and flush: an action name or number, or
        JSON probabilities. The next question is not sent until the answer
        arrives, and `timeout` seconds without one is an error."""
        proc = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, bufsize=0)
        pending = bytearray()

        def ask(obs, legal):
            try:
                proc.stdin.write((json.dumps(self.question(agent, obs, legal)) + "\n").encode())
            except OSError:
                pass                                # it has gone: the read below says so
            deadline = time.monotonic() + timeout
            while b"\n" not in pending:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([proc.stdout], [], [], left)[0]:
                    raise HunchRuntimeError(
                        f"`{command}` did not answer within {timeout:g} s. It must print one "
                        "line for each question as it reads it, and flush its output "
                        "(in Python: print(answer, flush=True))")
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    try:
                        code = proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        code = None
                    raise HunchRuntimeError(f"`{command}` stopped before answering "
                                            f"(exit code {code})")
                pending.extend(chunk)
            line, _, rest = bytes(pending).partition(b"\n")
            pending[:] = rest
            return self.parse_answer(agent, line.decode(errors="replace"), legal)

        def close():
            proc.stdin.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        ask.close = close
        return ask

    # ---- environments ----

    def _policy_path(self, policy) -> Optional[str]:
        if policy is None:
            return None
        return os.fspath(policy)

    def _mask_of(self, agents) -> int:
        if agents is None:
            agents = self.agents
        if isinstance(agents, str):
            agents = [agents]
        mask = 0
        for a in agents:
            if a not in self.agents:
                raise ValueError(f"no agent named '{a}'; this world has "
                                 + ", ".join(self.agents))
            mask |= 1 << self.agents.index(a)
        return mask

    def env(self, agents=None, policy=None, horizon: Optional[int] = None) -> "Env":
        """One environment. `agents` are decided by you (default: all of
        them); the others play `policy` (a Policy or a .hpol path), or
        uniformly at random when there is none."""
        return Env(self, self._mask_of(agents), policy, horizon)

    def vec(self, n: int, agent: Optional[str] = None, policy=None, seed: int = 0) -> "VecEnv":
        """n environments stepped together in C, as numpy arrays. `agent` is
        the one you decide (default: every agent, each on its turn)."""
        return VecEnv(self, n, self._mask_of(agent), policy, seed)

    def gym(self, agent: Optional[str] = None, policy=None, **kwargs):
        """A gymnasium.Env with you as `agent`; the others play `policy`."""
        from . import bindings
        return bindings.make_gym(self, agent, policy, **kwargs)

    def pettingzoo(self, policy=None, agents=None, **kwargs):
        """A pettingzoo AECEnv over `agents` (default: all)."""
        from . import bindings
        return bindings.make_pettingzoo(self, agents, policy, **kwargs)

    # ---- answers ----

    def _vm(self, args: List[str]) -> dict:
        cli.ensure_vm()
        proc = subprocess.run([str(cli.VM_BIN)] + [str(a) for a in args],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise HunchRuntimeError(proc.stderr.strip() or "the runtime failed")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def train(self, algo: str = "auto", out: Union[str, Path, None] = None, seed: int = 1,
              budget: Optional[int] = None, target: Optional[float] = None,
              max_sec: Optional[float] = None, exact_nodes: Optional[int] = None,
              think: Optional[float] = None, playouts: Optional[int] = None,
              score_sec: Optional[float] = None, train_sec: Optional[float] = None,
              games: Optional[int] = None, cores: Optional[int] = None) -> Trained:
        """Fill the blanks, as `hunch train` does, and return the policy with
        its report. The policy is saved beside the world unless `out` says
        where (`.path` on the result is where it went; for a world that ships
        with an installed Hunch, that is under HUNCH_HOME). `target` is the
        NashConv the solver stops at (default 0.001): make it smaller for a
        sharper reference to grade against. When the policy is a search
        (`algo="mcts"`, or chosen by "auto"), `think` is the seconds a decision
        is given on this machine and `playouts` sets the count directly;
        `train_sec` is how long it plays itself to train the network that
        guides it (default 600), `games` a number of games instead (0: no
        network), `cores` how many processes play them (default: all);
        `score_sec` is how long it is played to score it (default 5)."""
        if not self.agents:
            raise ValueError(f"{self.name} has no `hunch` blank, so there is nothing to train")
        out = Path(out) if out else cli.policy_beside(self.path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["train", self.hbc, "--algo", algo, "--seed", seed, "--out", out, "--json"]
        for flag, value in (("--budget", budget), ("--target", target),
                            ("--max-sec", max_sec), ("--exact-nodes", exact_nodes),
                            ("--think", think), ("--playouts", playouts),
                            ("--train-sec", train_sec), ("--games", games), ("--cores", cores)):
            if value is not None:
                cmd += [flag, value]
        train = self._vm(cmd)
        if train.get("method") == "mcts" and score_sec is None:
            score_sec = cli.TRAIN_SCORE_SEC
        return Trained(self, out, train,
                       self.score(out, exact_nodes=exact_nodes, score_sec=score_sec))

    def score(self, policy=None, exact_nodes: Optional[int] = None, episodes: Optional[int] = None,
              rows: int = 400, score_sec: Optional[float] = None) -> dict:
        """The report `hunch score --json` gives for a policy (None: uniform
        random play): what each agent collects, the best reply to it, and what
        it does in the situations it meets most."""
        cmd = ["report", self.hbc, "--rows", rows]
        if policy is not None:
            cmd += ["--policy", self._policy_path(policy)]
        if exact_nodes is not None:
            cmd += ["--exact-nodes", exact_nodes]
        if episodes is not None:
            cmd += ["--episodes", episodes]
        if score_sec is not None:                   # a search is played for this long
            cmd += ["--score-sec", score_sec]
        return self._vm(cmd)

    def situations(self, agent: Optional[str] = None, policy=None,
                   exact_nodes: Optional[int] = None) -> List[dict]:
        """Everything an agent can be shown, when the world is small enough to
        follow whole: dicts with agent, obs, legal (action numbers), probs
        (what `policy` does there), share (how often an episode comes to it)
        and, where it is reached, value (what the agent goes on to collect
        after each action). Raises ValueError for a world too big to list."""
        rep = self.score(policy, exact_nodes=exact_nodes, rows=100_000_000)
        if not rep.get("exact"):
            raise ValueError(f"{self.name} cannot be followed whole "
                             f"({show.why_not_exact(rep['why'], rep.get('limit', 0))})")
        out = []
        for r in rep["rows"]:
            name = self.agents[r["agent"]]
            if agent is not None and name != agent:
                continue
            row = dict(r, agent=name, legal=_legal_list(r["legal"], self.n_actions[name]))
            out.append(row)
        return out

    def _as_probs(self, agent: str, answer, legal: List[int]) -> List[float]:
        """What an agent function returned, as one probability per action."""
        n = self.n_actions[agent]
        if isinstance(answer, dict):
            probs = [0.0] * n
            for k, v in answer.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"{agent}: the probability of {k} must be a number, "
                                     f"got {v!r}")
                probs[self.action_index(agent, k)] = float(v)
        elif isinstance(answer, (list, tuple)) or hasattr(answer, "tolist"):
            probs = [float(v) for v in (answer.tolist() if hasattr(answer, "tolist") else answer)]
            if len(probs) != n:
                raise ValueError(f"{agent} has {n} actions, got {len(probs)} probabilities")
        else:
            a = self.action_index(agent, answer)
            probs = [1.0 if i == a else 0.0 for i in range(n)]
        if any(p < 0 or not math.isfinite(p) for p in probs):
            raise ValueError(f"{agent}: probabilities must be finite and not negative: {probs}")
        if any(probs[a] > 0 for a in range(n) if a not in legal):
            bad = [self.actions[agent][a] for a in range(n) if probs[a] > 0 and a not in legal]
            raise HunchRuntimeError(f"{agent} chose {', '.join(bad)}, which is not legal there; "
                                    f"legal: {', '.join(self.actions[agent][a] for a in legal)}")
        total = sum(probs)
        if total <= 0:
            raise ValueError(f"{agent}: no action was given any probability")
        return [p / total for p in probs]

    def write_policy(self, path: Union[str, Path], rows: List[dict]) -> Policy:
        """Save a table policy: `rows` are dicts with agent, obs, legal (action
        numbers) and probs. A situation with no row is played uniformly."""
        lines = ["HUNCHTAB 1", f"world {self.name}", f"rows {len(rows)}"]
        for r in rows:
            k = self.agents.index(r["agent"])
            mask = sum(1 << a for a in r["legal"])
            lines.append(" ".join(
                [str(k), str(self.n_actions[r["agent"]]), str(mask), str(len(r["obs"]))]
                + [repr(float(v)) for v in r["obs"]] + [f"{p:.12g}" for p in r["probs"]]))
        Path(path).write_text("\n".join(lines) + "\n")
        return Policy(self, path)

    def grade(self, agents: Union[Callable, Dict[str, Callable]], against=None,
              episodes: int = 2000, seed: int = 1, exact_nodes: Optional[int] = None) -> Grade:
        """What any agent collects in this world, and where it goes wrong.

        `agents` maps an agent's name to a function `f(obs, legal)`, called
        with what the agent is shown (a list of numbers) and the legal action
        numbers. It returns an action (a number or a name), or probabilities
        (a list with one per action, or a dict by name). One function alone
        grades the first agent. Every other agent plays `against` (a Policy or
        .hpol path; None for uniform random play).

        When the world is small enough to follow whole, the function is asked
        once about everything its agent can be shown, and the grade is exact:
        every outcome weighed, the best reply to the agent worked out, and each
        decision compared with the best action there. Otherwise `episodes`
        episodes are played and the result is a mean with a standard error.
        """
        if callable(agents):
            agents = {self.agents[0]: agents}
        for a in agents:
            if a not in self.agents:
                raise ValueError(f"no agent named '{a}'; this world has "
                                 + ", ".join(self.agents))
        ref = self.score(against, exact_nodes=exact_nodes, rows=100_000_000)
        reference = dict(zip(self.agents, ref["return"]))
        if not ref.get("exact"):
            return self._grade_sampled(agents, against, episodes, seed, reference)
        gap = None
        if len(self.agents) > 1 and all(ref["sure"]):
            gap = sum(max(0.0, b - r) for b, r in zip(ref["best_reply"], ref["return"]))
        rows = []
        for r in ref["rows"]:
            name = self.agents[r["agent"]]
            legal = _legal_list(r["legal"], self.n_actions[name])
            probs = r["probs"]
            if name in agents:
                probs = self._as_probs(name, agents[name](list(r["obs"]), list(legal)), legal)
            rows.append({"agent": name, "obs": r["obs"], "legal": legal, "probs": probs})
        with tempfile.TemporaryDirectory() as d:
            pol = self.write_policy(Path(d) / "graded.hpol", rows)
            rep = self.score(pol, exact_nodes=exact_nodes, rows=100_000_000)
        # The grade is of the table just written. If the runtime did not find
        # a row there (it matches what an agent is shown bit for bit), it
        # played that situation at random, and the numbers would be of
        # something else.
        wrote = {(r["agent"], tuple(r["obs"]), tuple(r["legal"])): r["probs"] for r in rows}
        mistakes = []
        for r in rep["rows"]:
            name = self.agents[r["agent"]]
            mine = wrote.get((name, tuple(r["obs"]), tuple(_legal_list(r["legal"],
                                                                       self.n_actions[name]))))
            if mine is None or any(abs(a - b) > 1e-9 for a, b in zip(mine, r["probs"])):
                raise HunchRuntimeError(
                    f"{self.name}: the graded policy was not played as written where {name} "
                    f"is shown {r['obs']}; this is a bug in hunch, please report it")
            if name not in agents or "value" not in r:
                continue
            vals = r["value"]
            legal = _legal_list(r["legal"], self.n_actions[name])
            best = max(legal, key=lambda a: vals[a])
            got = sum(r["probs"][a] * vals[a] for a in legal)
            cost = r["share"] * (vals[best] - got)
            if cost > 1e-9:
                chose = max(legal, key=lambda a: r["probs"][a])
                mistakes.append({"agent": name, "obs": r["obs"], "chose": chose, "best": best,
                                 "probs": list(r["probs"]), "cost": cost, "share": r["share"],
                                 "value": {self.actions[name][a]: vals[a] for a in legal}})
        mistakes.sort(key=lambda m: -m["cost"])
        return Grade(self, True, dict(zip(self.agents, rep["return"])), reference,
                     dict(zip(self.agents, rep["best_reply"])),
                     dict(zip(self.agents, rep["sure"])), mistakes, {}, list(agents),
                     reference_gap=gap, random_reference=against is None)

    def play(self, agents: Dict[str, Callable], against=None, episodes: int = 1,
             seed: int = 1) -> List[Dict[str, float]]:
        """Play whole episodes with functions in some seats (see `grade`) and
        `against` in the others. Returns what each agent collected, per episode."""
        import random
        rng = random.Random(seed)
        env = self.env(agents=list(agents), policy=against)
        out = []
        for ep in range(episodes):
            env.reset(rng.getrandbits(63))
            while not env.over:
                name = env.agent
                legal = env.legal
                probs = self._as_probs(name, agents[name](env.obs, list(legal)), legal)
                u, acc, action = rng.random(), 0.0, legal[-1]
                for a in legal:
                    acc += probs[a]
                    if probs[a] > 0 and u < acc:
                        action = a
                        break
                env.act(action)
            out.append(env.returns)
        return out

    def _grade_sampled(self, agents, against, episodes, seed, reference) -> Grade:
        totals = self.play(agents, against, episodes, seed)
        collects, sem = {}, {}
        for a in self.agents:
            xs = [t[a] for t in totals]
            mean = sum(xs) / len(xs)
            var = sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)
            collects[a], sem[a] = mean, math.sqrt(var / len(xs))
        return Grade(self, False, collects, reference, {}, {}, [], sem, list(agents), episodes,
                     random_reference=against is None)


class Env:
    """One episode at a time, with some agents decided from outside.

        env.reset(seed)
        while not env.over:
            env.act(choose(env.agent, env.obs, env.legal))
        env.returns          # what each agent collected

    `env.agent` is the name of the agent that must decide now. Everything the
    world does between your decisions (chance, the other agents, steps in which
    nobody decides) has already happened by the time `act` returns.
    """

    def __init__(self, world: World, outside: int, policy=None, horizon: Optional[int] = None,
                 n: int = 1):
        self.world = world
        self._lib = world._lib
        self._policy = policy       # kept alive: the path may be a temporary file's
        err = ctypes.create_string_buffer(320)
        path = world._policy_path(policy)
        self._p = self._lib.hunch_env_new(world._w, n, outside, path.encode() if path else None,
                                          horizon or 0, err, 320)
        if not self._p:
            raise HunchRuntimeError(err.value.decode())
        self.outside = [a for i, a in enumerate(world.agents) if (outside >> i) & 1]
        self._nag = len(world.agents)
        self._pay = (ctypes.c_double * MAX_AGENTS)()
        self._obs = (ctypes.c_double * MAX_OBS)()
        self.horizon: int = horizon or world.horizon
        self.returns: Dict[str, float] = {a: 0.0 for a in world.agents}
        self.last_paid: Dict[str, float] = dict(self.returns)
        self._next_seed = 0

    def __del__(self):
        p, self._p = getattr(self, "_p", None), None
        if p:
            self._lib.hunch_env_free(p)

    def _fail(self) -> None:
        raise HunchRuntimeError(self._lib.hunch_env_error(self._p).decode())

    def _collect(self) -> Dict[str, float]:
        self._lib.hunch_env_rewards(self._p, 0, self._pay, 1)
        paid = {a: self._pay[i] for i, a in enumerate(self.world.agents)}
        for a, v in paid.items():
            self.returns[a] += v
        return paid

    def reset(self, seed: Optional[int] = None) -> bool:
        """Start the episode `seed` names: the one `hunch run --seed` plays.
        Without a seed, the one after the last (0, 1, 2, ... from a new Env).
        True if an outside agent has a decision to make; False if the episode
        ended without one."""
        if seed is None:
            seed = self._next_seed
        self._next_seed = seed + 1
        if self._lib.hunch_env_reset(self._p, 0, seed & _U64) < 0:
            self._fail()
        self.returns = {a: 0.0 for a in self.world.agents}
        self.last_paid = self._collect()
        return not self.over

    def act(self, action) -> Dict[str, float]:
        """Play `action` (a number or a name) for the agent whose turn it is.
        Returns what each agent was paid before the next decision or the end.
        If the action is refused, or the world fails on the way (an assert),
        the error is raised and the environment is still waiting at this
        decision."""
        name = self.agent
        if name is None:
            raise HunchRuntimeError("no decision is waiting: the episode is over, or reset() "
                                    "was never called")
        if self._lib.hunch_env_act(self._p, 0, self.world.action_index(name, action)) < 0:
            self._fail()
        self.last_paid = self._collect()
        return self.last_paid

    @property
    def agent(self) -> Optional[str]:
        k = self._lib.hunch_env_agent(self._p, 0)
        return None if k < 0 else self.world.agents[k]

    @property
    def obs(self) -> List[float]:
        n = self._lib.hunch_env_obs(self._p, 0, self._obs)
        return list(self._obs[:n])

    @property
    def legal(self) -> List[int]:
        name = self.agent
        if name is None:
            return []
        return _legal_list(self._lib.hunch_env_legal(self._p, 0), self.world.n_actions[name])

    @property
    def over(self) -> bool:
        return self._lib.hunch_env_over(self._p, 0) != 0

    @property
    def terminated(self) -> bool:
        return self._lib.hunch_env_over(self._p, 0) == 1

    @property
    def truncated(self) -> bool:
        return self._lib.hunch_env_over(self._p, 0) == 2

    @property
    def steps(self) -> int:
        return self._lib.hunch_env_steps(self._p, 0)

    @property
    def state(self) -> dict:
        """The world's declared state as the current step began (for you to
        look at; an agent is shown only `obs`)."""
        buf = (ctypes.c_double * max(1, self.world.n_slots))()
        self._lib.hunch_env_state(self._p, 0, buf)
        return {name: (buf[base] if n == 1 else list(buf[base:base + n]))
                for name, (base, n) in self.world.fields.items()}

    def describe(self) -> str:
        return self.world.describe(self.agent, self.obs, self.legal)

    def snapshot(self) -> bytes:
        """This instant, random numbers included. `restore` it, here or in
        another Env of the same world, and the future is the same again: act
        differently after it and you have a fork. A snapshot holds the
        decision that was waiting, so an Env that restores it is asked that
        decision even if it was made to decide other agents."""
        buf = ctypes.create_string_buffer(self._lib.hunch_env_snapshot_size())
        self._lib.hunch_env_save(self._p, 0, buf)
        return self.world._digest + buf.raw + json.dumps(self.returns).encode()

    def restore(self, snap: bytes) -> None:
        size = self._lib.hunch_env_snapshot_size()
        mark = self.world._digest
        if len(snap) < len(mark) + size or snap[:len(mark)] != mark:
            raise HunchRuntimeError(f"this is not a snapshot of {self.world.name} as it is now")
        self._lib.hunch_env_load(self._p, 0, snap[len(mark):len(mark) + size])
        self.returns = json.loads(snap[len(mark) + size:].decode())

    def reseed(self, seed: int) -> None:
        """Keep everything as it is, but draw the random numbers from here on
        from a new stream: after a `restore`, a different future."""
        self._lib.hunch_env_reseed(self._p, 0, seed & _U64)

    def memory(self) -> tuple:
        """This instant as a tuple you can use as a dictionary key: two
        instants with the same key have the same future. It holds the step
        number, the waiting agent, a hash of where in the step the decision is
        asked, and everything the world remembers. What the random numbers
        will be is not in it: two instants with one key are the same position,
        and chance may still take them apart."""
        buf = (ctypes.c_double * max(1, self.world.n_slots))()
        n = self._lib.hunch_env_memory(self._p, 0, buf)
        return (self.steps, self.agent, self._lib.hunch_env_key(self._p, 0)) + tuple(buf[:n])

    def playout(self) -> Dict[str, float]:
        """Finish the episode with every agent playing at random among its
        legal actions, all in C. Returns what each agent was paid from here
        on. `snapshot` first if you want to come back."""
        if self._lib.hunch_env_playout(self._p, 0, self._pay) < 0:
            self._fail()
        paid = {a: self._pay[i] for i, a in enumerate(self.world.agents)}
        for a, v in paid.items():
            self.returns[a] += v
        return paid


class VecEnv:
    """n environments stepped together, as numpy arrays.

        obs, legal = envs.reset(seed)
        obs, reward, over, legal = envs.step(actions)

    obs is (n, width) float64; legal is (n, actions) bool; reward is (n,) for
    the one agent you decide, or (n, agents) when you decide all of them
    (`envs.agent` then says whose turn each environment is at). over is 0 while
    an episode runs, 1 when it ended by done() and 2 at the horizon. An
    environment whose episode ended has already started its next: the obs and
    legal of that row are the new episode's first decision. The arrays returned
    are yours: each call returns new ones.
    """

    def __init__(self, world: World, n: int, outside: int, policy=None, seed: int = 0):
        import numpy as np
        if n < 1:
            raise ValueError(f"vec() needs at least one environment, got {n}")
        self.world, self.n, self._np = world, n, np
        self._env = Env(world, outside, policy, n=n)
        self._lib, self._p = world._lib, self._env._p
        names = self._env.outside
        if not names:
            raise ValueError("vec() needs at least one agent decided by you")
        self.agents = names
        self._one = world.agents.index(names[0]) if len(names) == 1 else None
        self.width = max(world.obs_dim[a] for a in names)
        self.n_actions = max(world.n_actions[a] for a in names)
        nag = len(world.agents)
        self._obs = np.zeros((n, self.width), np.float64)
        self._rew = np.zeros((n, nag), np.float64)
        self._over = np.zeros(n, np.uint8)
        self._legal = np.zeros(n, np.uint32)
        self._agent = np.zeros(n, np.int32)
        self._bits = np.arange(self.n_actions, dtype=np.uint32)
        self._seed = seed

    def _ptr(self, a):
        return a.ctypes.data_as(ctypes.c_void_p)

    def _mask(self):
        return ((self._legal[:, None] >> self._bits) & 1).astype(bool)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._seed = seed
        if self._lib.hunch_vec_reset(self._p, self._seed & _U64, self._ptr(self._obs),
                                     self.width, self._ptr(self._legal),
                                     self._ptr(self._agent)) < 0:
            self._env._fail()
        return self._obs.copy(), self._mask()

    def step(self, actions):
        np = self._np
        acts = np.ascontiguousarray(actions, dtype=np.int32)
        if acts.shape != (self.n,):
            raise ValueError(f"step needs {self.n} actions, got shape {acts.shape}")
        if self._lib.hunch_vec_step(self._p, self._ptr(acts), self._ptr(self._obs), self.width,
                                    self._ptr(self._rew), self._ptr(self._over),
                                    self._ptr(self._legal), self._ptr(self._agent)) < 0:
            self._env._fail()
        rew = self._rew[:, self._one] if self._one is not None else self._rew
        return self._obs.copy(), rew.copy(), self._over.copy(), self._mask()

    @property
    def agent(self):
        """Whose turn each environment is at, as an index into world.agents."""
        return self._agent.copy()

    @property
    def obs(self):
        """What each environment's waiting agent is shown now."""
        return self._obs.copy()

    def random_steps(self, rounds: int, seed: int = 1) -> int:
        """Uniform random play for `rounds` steps of every environment, all
        in C: how fast the world itself runs. Returns the decisions made."""
        made = self._lib.hunch_vec_random(self._p, rounds, seed)
        if made < 0:
            self._env._fail()
        return made
