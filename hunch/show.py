"""What the commands print: a trained policy's report, an episode, a replay.

The runtime answers in JSON. Everything here turns that into the lines a
person reads, using the names the world's source gives: agents, actions
(`hunch p(obs, [left, right])`) and the expressions of the observation.
"""

from __future__ import annotations

from typing import List, Optional


def num(x: float, places: int = 4) -> str:
    """A signed number without trailing noise: +0.9300, -3, +0.0556."""
    if abs(x) < 0.5 * 10 ** -places:
        return "0"
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e9:
        return f"{int(round(x)):+d}"
    return f"{x:+.{places}f}"


def plain(x: float) -> str:
    """An observation or state value: 3, 0.625, -1."""
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
        return str(int(round(x)))
    return f"{x:.4g}"


def pct(p: float) -> str:
    if p <= 0:
        return "0%"
    if p < 0.005:
        return "<1%"
    if p > 0.995 and p < 1:
        return ">99%"
    return f"{100 * p:.0f}%"


class Names:
    """The world's own words for agents, actions and observation slots."""

    def __init__(self, unit: dict):
        self.agents: List[str] = list(unit["agents"])
        self.acts = {}
        self.labels = {}
        for name in self.agents:
            n = unit["agent_acts"].get(name, 2)
            given = unit.get("agent_action_names", {}).get(name)
            self.acts[name] = list(given) if given else [str(i) for i in range(n)]
            self.labels[name] = list(unit.get("agent_obs_labels", {}).get(name, []))
        self.named = {name: bool(unit.get("agent_action_names", {}).get(name))
                      for name in self.agents}

    def agent(self, i: int) -> str:
        return self.agents[i]

    def action(self, agent: int, a: int) -> str:
        acts = self.acts[self.agents[agent]]
        if 0 <= a < len(acts):
            return acts[a] if self.named[self.agents[agent]] else f"action {a}"
        return f"action {a}"

    def short(self, agent: int, a: int) -> str:
        return self.acts[self.agents[agent]][a]

    def sees(self, agent: int) -> str:
        return ", ".join(self.labels[self.agents[agent]])

    def obs(self, obs: list) -> str:
        return "[" + ", ".join(plain(v) for v in obs) + "]"

    def obs_labelled(self, agent: int, obs: list, indent: str = "        ") -> str:
        labels = self.labels[self.agents[agent]]
        if len(labels) != len(obs):
            return self.obs(obs)
        parts = [f"{lab} = {plain(v)}" for lab, v in zip(labels, obs)]
        if sum(len(part) for part in parts) > 70:
            # too much for one line: one value a line, under the line that asked
            return "".join(f"\n{indent}{part}" for part in parts)
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# The report: what a policy collects and what it does
# ---------------------------------------------------------------------------

def why_not_exact(why: str, limit: int) -> str:
    if why.startswith("rand:"):
        _, line, col = why.split(":")
        if int(col) < 0:
            return (f"sampled, not exact: the randint at line {line} has more outcomes than "
                    "are followed one by one (65,536)")
        return (f"sampled, not exact: rand() at line {line} has more outcomes than can be "
                "listed (randint can be listed)")
    if why == "off":
        return "sampled, not exact: exact scoring was turned off"
    return f"sampled, not exact: the world has more than {limit:,} positions"


def how_trained(train: dict) -> str:  # noqa: C901
    algo = train["algo"]
    sec = f"{train['wall_sec']:.1f}s"
    if algo == "regret":
        method = train.get("method")
        if method == "lookahead":
            return (f"solved by looking ahead: each of {train['positions']:,} positions was "
                    f"worked back from the end, and no agent would choose differently if it "
                    f"saw the whole state ({sec})")
        if method == "search":
            return (f"solved by search over every outcome: {train['positions']:,} positions, "
                    f"{train['situations']:,} situations ({sec})")
        if method == "regret":
            size = (f"{train['positions']:,} positions, {train['situations']:,} situations, "
                    f"{train['iterations']:,} rounds ({sec})")
            if train.get("gap", 0.0) > train.get("target", 0.001) * 1.0000001:
                # It ran out of time or rounds. What it reached is measured below.
                return (f"regret matching over every outcome stopped short of its target "
                        f"({train['target']:g}): {size}")
            return f"solved by regret matching over every outcome: {size}"
        return (f"trained by regret matching on sampled episodes: "
                f"{int(train['steps_used']):,} steps, {train['situations']:,} situations met "
                f"({sec})")
    if algo == "random":
        return "not trained: every agent plays at random"
    return f"trained with {algo}: {int(train['steps_used']):,} steps ({sec})"


def report_lines(rep: dict, names: Names, rows: int = 6) -> List[str]:
    out: List[str] = []
    nag = len(names.agents)
    exact = rep["exact"]
    ret = rep["return"]

    if exact:
        out.append(f"scored exactly: every chance outcome and every action weighed, "
                   f"{rep['positions']:,} positions")
    else:
        out.append(f"{why_not_exact(rep['why'], rep.get('limit', 0))}; "
                   f"{rep['episodes']:,} episodes, ± is one standard error")
    out.append("")

    def val(i: int, key: str, sem: str) -> str:
        if exact or sem not in rep:
            return num(rep[key][i])
        return f"{num(rep[key][i])} ± {rep[sem][i]:.4f}"

    width = max(len(n) for n in names.agents)
    for i, name in enumerate(names.agents):
        line = f"  {name:<{width}}  collects {val(i, 'return', 'sem')} per episode"
        if nag == 1:
            line += f"    (random play: {val(i, 'random', 'random_sem')})"
        out.append(line)
    if nag > 1:
        out.append("  with everyone playing at random:  "
                   + ", ".join(f"{n} {val(i, 'random', 'random_sem')}"
                               for i, n in enumerate(names.agents)))
        out.append("  against opponents playing at random:  "
                   + ", ".join(f"{n} {val(i, 'vs_random', 'vs_random_sem')}"
                               for i, n in enumerate(names.agents)))

    if exact:
        best, sure = rep["best_reply"], rep["sure"]
        gains = [max(0.0, best[i] - ret[i]) for i in range(nag)]
        if nag == 1:
            if sure[0]:
                out.append(f"  the best any policy can collect is {num(best[0])}"
                           + ("" if gains[0] > 5e-5 else ": this one is optimal"))
            else:
                top = rep.get("seeing_all", best[0])
                if gains[0] > 5e-5:
                    out.append(f"  a better policy was found: it collects {num(best[0])}")
                else:
                    out.append("  no better policy was found, but that is a search, not a proof")
                out.append(f"  seeing the whole state would allow {num(top)}; "
                           f"{names.agents[0]} is shown less than that")
        else:
            total = sum(gains)
            each = ", ".join(f"{n} {gains[i]:.4f}" for i, n in enumerate(names.agents))
            if all(sure):
                if total < 5e-5:
                    out.append("  no agent can gain anything by changing its policy alone: "
                               "this is an equilibrium")
                else:
                    out.append(f"  the most each agent could gain by changing its policy alone: "
                               f"{each}   (NashConv {total:.4f})")
            else:
                out.append(f"  a better reply was searched for, and gains: {each}")
                out.append("    (an agent here is shown the same thing after different "
                           "histories, so the search is not a proof that nothing gains more)")
    elif rep.get("unseen"):
        out.append(f"  {rep['unseen']:,} of {rep['decisions']:,} decisions were in situations "
                   "the policy never met in training, and were played at random")

    # what it does
    by_agent = {i: [] for i in range(nag)}
    for row in rep["rows"]:
        by_agent[row["agent"]].append(row)
    for i, name in enumerate(names.agents):
        mine = by_agent[i]
        if not mine:
            continue
        out.append("")
        acts = names.acts[name]
        freq = rep["freq"][i]
        tot = sum(freq) or 1.0
        mix = ", ".join(f"{acts[a]} {pct(freq[a] / tot)}" for a in range(len(acts))
                        if freq[a] > 0)
        out.append(f"what {name} does  (over all its decisions: {mix})")
        shown = mine[:rows]
        if sum(r["share"] for r in shown) < 0.05 * tot:
            # Measurements, not a handful of cases: nearly every decision is
            # made somewhere new, and a table of six of them says nothing.
            out.append(f"  it is rarely shown the same thing twice ({rep['situations']:,}"
                       f"{'' if rep.get('listed_all', True) else '+'} different observations), "
                       "so there is no table;")
            out.append("  hunch run shows an episode, hunch why what each decision was worth")
            continue
        cells = [[plain(v) for v in r["obs"]] for r in shown]
        labels = names.labels[name]
        legend = []
        if len(labels) != len(shown[0]["obs"]) or sum(len(lab) for lab in labels) > 60:
            short = [f"obs[{k}]" for k in range(len(shown[0]["obs"]))]
            if len(labels) == len(short) and labels != short:
                legend = [f"  {short[k]} is {labels[k]}" for k in range(len(short))]
            labels = short
        widths = [max([len(labels[k])] + [len(c[k]) for c in cells])
                  for k in range(len(labels))]
        awidth = [max(len(a), 4) for a in acts]
        head = "  " + "  ".join(f"{labels[k]:>{widths[k]}}" for k in range(len(labels)))
        head += "   |  " + "  ".join(f"{acts[a]:>{awidth[a]}}" for a in range(len(acts)))
        head += "   |  times per episode"
        out.append(head)
        for r, c in zip(shown, cells):
            line = "  " + "  ".join(f"{c[k]:>{widths[k]}}" for k in range(len(labels)))
            probs = []
            for a in range(len(acts)):
                legal = (r["legal"] >> a) & 1
                probs.append(f"{(pct(r['probs'][a]) if legal else '-'):>{awidth[a]}}")
            line += "   |  " + "  ".join(probs)
            line += f"   |  {r['share']:.3g}"
            out.append(line)
        out.extend(legend)
        has = rep.get("situations_by_agent", [len(mine)] * nag)[i]
        if has > len(shown):
            out.append(f"  the {len(shown)} it meets most, of {has:,} situations "
                       f"(--rows N lists more)")
    return out


# ---------------------------------------------------------------------------
# An episode, step by step
# ---------------------------------------------------------------------------

def state_text(fields: list, state: list, limit: int = 12) -> str:
    parts = []
    for f in fields:
        vals = state[f["base"]:f["base"] + f["len"]]
        if f["len"] == 1:
            parts.append(f"{f['name']}={plain(vals[0])}")
        elif f["len"] <= limit:
            parts.append(f"{f['name']}=[{' '.join(plain(v) for v in vals)}]")
        else:
            parts.append(f"{f['name']}=[{f['len']} values]")
    return "  ".join(parts)


def state_changes(fields: list, before: list, after: list, limit: int = 6) -> str:
    parts = []
    for f in fields:
        for k in range(f["len"]):
            a, b = before[f["base"] + k], after[f["base"] + k]
            if a != b:
                name = f["name"] if f["len"] == 1 else f"{f['name']}[{k}]"
                parts.append(f"{name} {plain(a)}→{plain(b)}")
    if len(parts) > limit:
        parts = parts[:limit] + [f"and {len(parts) - limit} more"]
    return ", ".join(parts)


def decision_text(d: dict, names: Names, by: str = "") -> str:
    name = names.agent(d["agent"])
    act = names.action(d["agent"], d["action"])
    # The share is how often the policy plays this here; a move made by a
    # person or a script has none.
    how = by or pct(d["probs"][d["action"]])
    return f"{name} sees {names.obs(d['obs'])}, plays {act} ({how})"


def episode_lines(trace: dict, ep: dict, names: Names,
                  you: Optional[int] = None, scripted: Optional[tuple] = None) -> List[str]:
    """`you` is the agent a person played. `scripted` is (agent, n): that
    agent's first n decisions came from --actions."""
    out = []
    left = scripted[1] if scripted else 0

    def by(d: dict) -> str:
        nonlocal left
        if d["agent"] == you:
            return "you"
        if scripted and d["agent"] == scripted[0] and left > 0:
            left -= 1
            return "scripted"
        return ""

    fields = trace["fields"]
    out.append(f"start    {state_text(fields, ep['start'])}")
    prev = ep["start"]
    totals = {n: 0.0 for n in names.agents}
    for t, step in enumerate(ep["steps"], 1):
        head = f"step {t:<3} "
        pad = " " * len(head)
        lines = [decision_text(d, names, by(d)) for d in step["d"]]
        paid = [f"{n} {num(v)}" for n, v in step["r"].items() if v]
        for n, v in step["r"].items():
            totals[n] = totals.get(n, 0.0) + v
        change = state_changes(fields, prev, step["s"])
        if change:
            lines.append(change)
        if paid:
            lines.append("reward: " + ", ".join(paid))
        if not lines:
            lines = ["(nothing decided, nothing changed)"]
        out.append(head + lines[0])
        out.extend(pad + line for line in lines[1:])
        prev = step["s"]
    n = len(ep["steps"])
    paid_out = ", ".join(f"{name} collected {num(v)}" for name, v in totals.items())
    out.append(f"ended after {n} step{'s' if n != 1 else ''}" + (f":  {paid_out}" if paid_out else ""))
    return out


# ---------------------------------------------------------------------------
# Why: each decision replayed once per action
# ---------------------------------------------------------------------------

def situation_key(agent: int, obs: list, legal: int):
    return (agent, legal, tuple(float(f"{v:.9g}") for v in obs))


def why_lines(why: dict, names: Names, detail: bool, known: Optional[dict] = None) -> List[str]:
    """`known` maps a situation (see situation_key) to the exact value of each
    action there, over every instant in which the agent is shown the same."""
    out = []
    n = why["samples"]
    better = []
    swing = (0.0, None)
    any_known = False
    for d in why["decisions"]:
        agent, took = d["agent"], d["action"]
        legal = [a for a in range(len(d["value"])) if d["value"][a] is not None]
        spread = max(d["value"][a] for a in legal) - min(d["value"][a] for a in legal)
        if spread > swing[0]:
            swing = (spread, d)
        mask = sum(1 << a for a in legal)
        q = (known or {}).get(situation_key(agent, d["obs"], mask))
        if q is not None:
            # Exact, and from the agent's side: a mistake is an action that is
            # better whatever the hidden things turn out to be, on average.
            any_known = True
            wins = [a for a in legal if a != took and q[a] - q[took] > 1e-3]
            if wins:
                better.append((d, max(wins, key=lambda a: q[a]), "seen"))
        else:
            wins = [a for a in legal if a != took and d["diff"][a] > 2 * d["se"][a]
                    and d["diff"][a] > 1e-9]
            if wins:
                better.append((d, max(wins, key=lambda a: d["diff"][a]), "once"))
        name = names.agent(agent)
        if detail:
            out.append(f"step {d['step']}  {name} sees {names.obs_labelled(agent, d['obs'])}")
            out.append(f"        and plays {names.action(agent, took)} "
                       f"({pct(d['probs'][took])})")
            width = max(len(names.short(agent, a)) for a in legal)
            out.append("")
            out.append(f"  this episode, replayed from that instant {n} times per action:")
            for a in legal:
                mark = ">" if a == took else " "
                line = (f"      {mark} {names.short(agent, a):<{width}}  "
                        f"{num(d['value'][a]):>9}")
                if a != took:
                    diff = d["diff"][a]
                    word = "better" if diff > 0 else "worse"
                    line += (f"   {abs(diff):.4f} {word} than {names.short(agent, took)}"
                             f" ± {d['se'][a]:.4f}")
                else:
                    line += "   taken"
                out.append(line)
            if q is not None:
                out.append("")
                out.append(f"  every time {name} is shown this, whatever it cannot see (exact):")
                for a in legal:
                    mark = ">" if a == took else " "
                    line = (f"      {mark} {names.short(agent, a):<{width}}  "
                            f"{num(q[a]):>9}")
                    if a != took:
                        diff = q[a] - q[took]
                        if abs(diff) <= 1e-3:
                            line += f"   as good as {names.short(agent, took)}"
                        else:
                            word = "better" if diff > 0 else "worse"
                            line += f"   {abs(diff):.4f} {word} than {names.short(agent, took)}"
                    else:
                        line += "   taken"
                    out.append(line)
            out.append("")
        else:
            once, seen = [], []
            for a in legal:
                if a == took:
                    continue
                star = "*" if (a in wins and q is None) else ""
                err = f" ±{d['se'][a]:.2f}" if d["se"][a] >= 0.005 else ""
                once.append(f"{names.short(agent, a)} {num(d['diff'][a], 3)}{err}{star}")
                if q is not None:
                    star = "*" if a in wins else ""
                    seen.append(f"{names.short(agent, a)} {num(q[a] - q[took], 3)}{star}")
            line = f"step {d['step']:<3} {name} plays {names.short(agent, took)}"
            if q is not None:
                line += "\n" + " " * 9 + "whenever shown this:  " + "   ".join(seen)
                line += "\n" + " " * 9 + "this episode:         " + "   ".join(once)
            else:
                line += "\n" + " " * 9 + "instead:  " + "   ".join(once)
            out.append(line)
    if not why["decisions"]:
        out.append("no decision was made in that step")
        return out
    if not detail:
        out.append("")
        if any_known:
            out.append("each number is what another action collects, less what the action "
                       "taken collects.")
            out.append("  whenever shown this: over every instant in which the agent is shown "
                       "the same thing (exact).")
            out.append("    This is the test of the policy: it cannot act on what it cannot see.")
            out.append(f"  this episode: that exact instant replayed, hidden things as they were; "
                       f"the mean of {n}")
            out.append("    continuations, ± one standard error where what follows is not certain.")
        else:
            out.append("each number is what that action would have collected, less what the "
                       "action taken collects,")
            out.append(f"averaged over {n} continuations from that exact instant (the same ones "
                       "for every action),")
            out.append("± one standard error; anything the agent could not see is as it was in "
                       "this episode.")
    if better:
        def what(d, a, kind):
            return f"step {d['step']} ({names.short(d['agent'], a)})"
        steps = ", ".join(what(*b) for b in better[:8])
        if any(kind == "seen" for _, _, kind in better):
            out.append(f"* another action was better whenever the agent is shown that, at: "
                       f"{steps}")
        else:
            out.append(f"* another action was better by more than twice its standard error at: "
                       f"{steps}")
    elif any_known:
        out.append("no decision had a better action, going by what the agent is shown")
    else:
        out.append("no decision had another action better by more than twice its standard error")
    if swing[1] is not None and not detail:
        d = swing[1]
        out.append(f"the decision that mattered most: step {d['step']} "
                   f"(best and worst action {swing[0]:.4f} apart);  --step {d['step']} "
                   "shows it in full")
    return out


def pick_action(text: str, acts: List[str], legal: int) -> Optional[int]:
    """The action a person typed, by number or by name; None if it is not one."""
    text = text.strip().lower()
    if not text:
        return None
    choice = None
    if text.isdigit():
        choice = int(text)
    else:
        hits = [i for i, a in enumerate(acts) if a.lower() == text]
        if not hits:
            hits = [i for i, a in enumerate(acts)
                    if a.lower().startswith(text) and (legal >> i) & 1]
        if len(hits) == 1:
            choice = hits[0]
    if choice is None or choice < 0 or choice >= len(acts) or not (legal >> choice) & 1:
        return None
    return choice
