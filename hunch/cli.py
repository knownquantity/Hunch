"""Hunch command-line interface.

Everything here uses only the Python standard library, so `hunch` runs on a
clean machine with nothing but python3 and gcc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import codegen, show
from .lang import HunchError, parse

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "runtime" / "hunch_vm.c").exists():
    # A source checkout: build and cache next to the sources.
    RUNTIME = ROOT / "runtime" / "hunch_vm.c"
    HOME = ROOT
    WORLDS = ROOT / "worlds"
else:
    # Installed with pip: the runtime ships inside the package, and what is
    # built from it goes to the user's cache directory.
    RUNTIME = Path(__file__).resolve().parent / "_runtime" / "hunch_vm.c"
    HOME = Path(os.environ.get("HUNCH_HOME") or Path.home() / ".cache" / "hunch")
    WORLDS = Path(__file__).resolve().parent / "_worlds"
BUILD = HOME / "build"
VM_BIN = BUILD / "hunch-vm"
VM_LIB = BUILD / "libhunch.so"
CACHE = HOME / ".hunch-cache"


# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------

def _build(target: Path, flags: list, verbose: bool = False) -> Path:
    BUILD.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_mtime >= RUNTIME.stat().st_mtime:
        return target
    cc = os.environ.get("CC", "gcc")
    if shutil.which(cc) is None:
        raise SystemExit(
            "hunch: no C compiler found (need gcc or clang). "
            "Install one and try again.")
    # Built under another name and moved into place, so two processes that
    # start together never run half a file.
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    cmd = [cc, "-O3", "-march=native", "-std=c99", *flags, "-o", str(tmp), str(RUNTIME), "-lm"]
    if verbose:
        print(" ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit("hunch: failed to build the runtime\n" + proc.stderr)
    os.replace(tmp, target)
    return target


def ensure_vm(verbose: bool = False) -> Path:
    """The runtime as a program: what the commands run."""
    return _build(VM_BIN, [], verbose)


def ensure_lib(verbose: bool = False) -> Path:
    """The same file as a shared library: what `import hunch` steps worlds with."""
    return _build(VM_LIB, ["-fPIC", "-shared", "-DHUNCH_LIB"], verbose)


def resolve_world(name) -> Path:
    """The file a command was given: a path, or the name of a world that ships
    with Hunch (`hunch train kuhn`)."""
    path = Path(name)
    if not path.exists() and path.name == str(name) and (WORLDS / f"{name}.hunch").exists():
        return WORLDS / f"{name}.hunch"
    return path


def policy_beside(path: Path) -> Path:
    """Where `train` saves a world's policy and the other commands look for
    it: beside the world, or, for a world that ships inside an installed
    package, under HUNCH_HOME."""
    path = Path(path)
    if HOME != ROOT and path.resolve().parent == WORLDS.resolve():
        return HOME / "policies" / f"{path.stem}.hpol"
    return path.with_suffix(".hpol")


def parse_params(pairs) -> dict:
    """Turn ["name=value", ...] into {"name": value}."""
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise HunchError(f"--param needs name=value, got {pair!r}",
                             hint="e.g. --param delay=3")
        name, _, raw = pair.partition("=")
        try:
            out[name.strip()] = float(raw)
        except ValueError:
            raise HunchError(f"--param {name}={raw} is not a number")
    return out


def compile_file(path: Path, params: dict = None) -> tuple[Path, dict]:
    """Compile a .hunch file into a cached .hbc and return (path, unit)."""
    source = path.read_text()
    prog = parse(source, str(path))
    blob, unit = codegen.compile_program(prog, params)
    CACHE.mkdir(parents=True, exist_ok=True)
    # Named by its contents: two files called world.hunch in different folders, or
    # one file under different params, must never read each other's bytecode.
    digest = hashlib.sha1(blob).hexdigest()[:16]
    hbc = CACHE / f"{path.stem}-{digest}.hbc"
    # Written under another name and moved into place, so a run started at the
    # same moment on the same world never reads half a file.
    tmp = hbc.with_name(f"{hbc.name}.{os.getpid()}.tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, hbc)
    return hbc, unit


def _ranges(labels: list[str]) -> list[str]:
    """`board[0], board[1], board[2]` as `board[0..2]`: an array shown element
    by element, in order, reads as one range."""
    out: list[str] = []
    run: tuple[str, int, int] | None = None

    def close():
        if run:
            name, lo, hi = run
            out.append(f"{name}[{lo}]" if lo == hi else f"{name}[{lo}..{hi}]")

    for label in labels:
        m = re.fullmatch(r"(\w+)\[(\d+)\]", label)
        if m and run and run[0] == m.group(1) and int(m.group(2)) == run[2] + 1:
            run = (run[0], run[1], run[2] + 1)
            continue
        close()
        run = (m.group(1), int(m.group(2)), int(m.group(2))) if m else None
        if not m:
            out.append(label)
    close()
    return out


def describe(unit: dict) -> str:
    lines = [
        f"world      {unit['world']}",
        f"horizon    {unit['horizon']} steps",
        f"state      {unit['nslots']} slots, {len(unit['decls'])} declarations",
        f"bytecode   {len(unit['code'])} instructions, {len(unit['consts'])} constants",
    ]
    if unit["agents"]:
        for i, name in enumerate(unit["agents"]):
            lines.append(
                f"hunch #{i}   {name!r}: {unit['agent_obs'].get(name)} observation dims, "
                f"{unit['agent_acts'].get(name)} actions")
            labels = unit.get("agent_obs_labels", {}).get(name)
            if labels:
                lines.append("             sees     " + ", ".join(_ranges(labels)))
            acts = unit.get("agent_action_names", {}).get(name)
            if acts:
                lines.append("             actions  " + ", ".join(acts))
    else:
        lines.append("hunch      (none - this world has no decisions)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _vm(args_str: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    ensure_vm()
    return subprocess.run([str(VM_BIN)] + args_str, capture_output=capture, text=True)


def _open(args):
    """Compile the world named on the command line. (path, hbc, unit), or None
    after printing why not."""
    path = resolve_world(args.file)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return None
    try:
        hbc, unit = compile_file(path, parse_params(args.param))
    except HunchError as exc:
        print(str(exc), file=sys.stderr)
        return None
    return path, hbc, unit


def find_policy(args, path: Path):
    """The policy a command uses: the one named, or the one `hunch train` left
    beside the world. `--policy none` means every agent plays at random."""
    if args.policy:
        return None if args.policy == "none" else args.policy
    default = policy_beside(path)
    return str(default) if default.exists() else None


def search_header(policy) -> dict:
    """The fields of a policy file that is a search (`HUNCHSEARCH 1`: world,
    playouts, explore, hidden, checked; version 2 adds `rollout` and `guide`
    lines, for the network the search learned by self-play); {} for a table,
    networks or no file."""
    try:
        with open(policy) as f:
            lines = [f.readline().split() for _ in range(8)]
    except (OSError, TypeError, UnicodeDecodeError):
        return {}
    if not lines[0] or lines[0][0] != "HUNCHSEARCH":
        return {}
    return {words[0]: words[1:] for words in lines[1:] if words}


def policy_words(policy) -> str:
    if not policy:
        return "no policy, so every agent plays at random"
    head = search_header(policy)
    if not head:
        return f"policy {policy}"
    try:
        words = f"policy {policy} (a search, {int(head['playouts'][0]):,} playouts a decision"
    except (KeyError, IndexError, ValueError):
        return f"policy {policy} (a search)"
    if "guide" in head:
        words += ", guided by a network it trained by playing itself"
    if head.get("hidden") == ["found"]:
        words += "; it looks at the whole state, including what its agent is not shown"
    return words + ")"


def _vm_json(cmd: list[str], policy=None):
    """Run the runtime and read the JSON it prints. None after printing why not."""
    proc = _vm(cmd)
    if proc.returncode != 0:
        print(proc.stderr.strip(), file=sys.stderr)
        if policy and "could not read policy" in proc.stderr:
            print(f"  = help: {policy} does not fit this world. It may be from an older "
                  "version of it: train again, or pass --policy none", file=sys.stderr)
        return None
    return json.loads(proc.stdout)


def cmd_run(args) -> int:
    got = _open(args)
    if got is None:
        return 1
    path, hbc, unit = got
    policy = find_policy(args, path)
    cmd = ["trace", str(hbc), "--seed", str(args.seed), "--episodes", str(args.episodes)]
    if policy:
        cmd += ["--policy", policy]
    if args.forks:
        cmd.append("--forks")
    names = show.Names(unit)
    scripted = None
    if args.agent:
        if args.agent not in names.agents:
            print(f"hunch: no agent named '{args.agent}'; this world has "
                  + ", ".join(names.agents), file=sys.stderr)
            return 1
        # --actions takes numbers or the names the world gives its actions.
        acts = names.acts[args.agent]
        picked = []
        for word in filter(None, (w.strip() for w in (args.actions or "").split(","))):
            if word in acts:
                picked.append(acts.index(word))
            elif word.isdigit() and int(word) < len(acts):
                picked.append(int(word))
            else:
                print(f"hunch: '{word}' is not an action of {args.agent}; it has "
                      + ", ".join(f"{i} ({a})" if names.named[args.agent] else a
                                  for i, a in enumerate(acts)), file=sys.stderr)
                return 1
        cmd += ["--agent", args.agent, "--actions", ",".join(map(str, picked))]
        scripted = (names.agents.index(args.agent), len(picked))
    if args.json or args.forks or args.out:
        # The raw trace, for programs: the Studio reads this.
        if args.out:
            cmd += ["--out", args.out]
        proc = _vm(cmd, capture=(args.out is None))
        if proc.returncode != 0:
            print(proc.stderr or "", file=sys.stderr)
            return proc.returncode
        if args.out is None:
            print(proc.stdout)
        return 0
    trace = _vm_json(cmd, policy)
    if trace is None:
        return 1
    for k, ep in enumerate(trace["episodes"]):
        if k:
            print()
        print(f"{trace['world']}, seed {ep['seed']}, {policy_words(policy)}")
        print("\n".join(show.episode_lines(trace, ep, names, scripted=scripted)))
    return 0


def cmd_score(args) -> int:
    """What a policy collects and what it does, without training anything."""
    got = _open(args)
    if got is None:
        return 1
    path, hbc, unit = got
    policy = find_policy(args, path)
    rep = _report(hbc, policy, args)
    if rep is None:
        return 1
    if args.json:
        print(json.dumps(rep))
        return 0
    print(f"{unit['world']}: {policy_words(policy)}")
    print("\n".join(show.report_lines(rep, show.Names(unit), args.rows)))
    return 0


EXACT_NODES = 2_000_000     # the runtime's default for --exact-nodes


def _exact_limit(args) -> int:
    asked = getattr(args, "exact_nodes", None)
    return EXACT_NODES if asked is None else asked


def _too_big(hbc: Path, args) -> bool:
    """Following a world whole takes seconds before it turns out to be too big.
    Once one command has found that out, the others do not try again. The note
    sits beside the bytecode, which is named by the world's contents."""
    try:
        return int(hbc.with_suffix(".big").read_text()) >= _exact_limit(args) > 0
    except (OSError, ValueError):
        return False


def _note_too_big(hbc: Path, args) -> None:
    try:
        hbc.with_suffix(".big").write_text(str(_exact_limit(args)))
    except OSError:
        pass


def _report(hbc: Path, policy, args):
    cmd = ["report", str(hbc), "--seed", str(getattr(args, "seed", 1))]
    if policy:
        cmd += ["--policy", policy]
    if getattr(args, "episodes", None):
        cmd += ["--episodes", str(args.episodes)]
    known_big = _too_big(hbc, args)
    if known_big:
        cmd += ["--exact-nodes", "0"]
    elif getattr(args, "exact_nodes", None) is not None:
        cmd += ["--exact-nodes", str(args.exact_nodes)]
    cmd += ["--rows", str(max(getattr(args, "rows", 6), 6))]
    if getattr(args, "score_sec", None) is not None:
        cmd += ["--score-sec", str(args.score_sec)]
    rep = _vm_json(cmd, policy)
    if rep is None:
        return None
    if rep.get("why") == "search":
        pass                                        # sampled because of what the policy is
    elif known_big:
        rep["why"], rep["limit"] = "big", _exact_limit(args)
    elif not rep["exact"] and rep["why"] == "big":
        _note_too_big(hbc, args)
    return rep


def cmd_grade(args) -> int:
    """Grade an agent that is not a policy file: a program, or written answers."""
    from . import world as _world
    path = resolve_world(args.file)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 1
    try:
        w = _world.World(path, parse_params(args.param))
    except HunchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    agent = args.agent or (w.agents[0] if w.agents else None)
    if agent not in w.agents:
        print(f"hunch: no agent named '{agent}'; this world has " + ", ".join(w.agents),
              file=sys.stderr)
        return 1
    against = find_policy(args, path)
    asked = None
    try:
        if args.questions:
            for q in w.questions(agent, args.exact_nodes):
                print(json.dumps(q))
            return 0
        if args.answers:
            lines = Path(args.answers).read_text().splitlines()
            while lines and not lines[-1].strip():
                lines.pop()
            answers, with_ids = {}, None
            for n, line in enumerate(lines, 1):
                at = f"{args.answers} line {n}"
                if not line.strip():
                    raise ValueError(f"{at} is blank: every line is one answer")
                try:
                    row = json.loads(line)
                except ValueError:
                    row = None
                has_id = isinstance(row, dict) and "id" in row
                if with_ids is None:
                    with_ids = has_id
                if has_id != with_ids:
                    raise ValueError(f"{at}: some lines carry an id and some do not; "
                                     "use one form throughout")
                if not has_id:
                    answers[n - 1] = line           # question n - 1, read by parse_answer
                    continue
                given = [k for k in ("answer", "action", "probs") if k in row]
                qid = row["id"]
                if isinstance(qid, bool) or not isinstance(qid, int):
                    raise ValueError(f"{at}: the id must be a question number, got {qid!r}")
                if not given:
                    raise ValueError(f'{at}: an id but no "answer"')
                if qid in answers:
                    raise ValueError(f"{at}: a second answer to question {qid}")
                answers[qid] = row[given[0]]
            if not with_ids:                        # in order: one per question
                answers = [answers[k] for k in range(len(answers))]
            grade = w.grade_answers(answers, agent, against, args.exact_nodes)
        elif args.cmd:
            asked = w.program(agent, args.cmd, args.timeout)
            grade = w.grade({agent: asked}, against, episodes=args.episodes, seed=args.seed,
                            exact_nodes=args.exact_nodes)
        else:
            print("hunch: grade needs --cmd PROGRAM, --answers FILE or --questions",
                  file=sys.stderr)
            return 1
    except (ValueError, _world.HunchRuntimeError) as exc:
        print(f"hunch: {exc}", file=sys.stderr)
        return 1
    finally:
        if asked is not None:
            try:
                asked.close()
            except Exception:  # noqa: BLE001
                pass
    if args.json:
        print(json.dumps({
            "world": w.name, "agent": agent, "exact": grade.exact, "collects": grade.collects,
            "reference": grade.reference, "best_reply": grade.best_reply, "sure": grade.sure,
            "gives_away": grade.gives_away(agent), "sem": grade.sem, "episodes": grade.episodes,
            "mistakes": grade.mistakes}))
        return 0
    print(grade)
    if len(w.agents) > 1:
        print(f"  the other agents played: {policy_words(against)}")
    return 0


def cmd_why(args) -> int:
    """Replay each decision of one episode once per action."""
    got = _open(args)
    if got is None:
        return 1
    path, hbc, unit = got
    policy = find_policy(args, path)
    cmd = ["why", str(hbc), "--seed", str(args.seed), "--samples", str(args.samples)]
    if policy:
        cmd += ["--policy", policy]
    if args.step:
        cmd += ["--step", str(args.step)]
    why = _vm_json(cmd, policy)
    if why is None:
        return 1
    names = show.Names(unit)
    if not names.agents:
        print(f"hunch: {path} has no `hunch` blank, so no decision was made",
              file=sys.stderr)
        return 1
    if "search" in why:
        # A search has already compared the actions at each decision; nothing
        # is replayed, and there is no table of situations to look up.
        if args.json:
            print(json.dumps(why))
            return 0
        print(f"{unit['world']}, seed {args.seed}, {policy_words(policy)}")
        print("the episode ended with "
              + ", ".join(f"{n} {show.num(v)}" for n, v in zip(names.agents, why["returns"])))
        print()
        print("\n".join(show.why_search_lines(why, names, detail=bool(args.step))))
        return 0
    known = _situation_values(hbc, policy, args)
    if args.json:
        for d in why["decisions"]:
            legal = sum(1 << a for a, v in enumerate(d["value"]) if v is not None)
            d["shown_value"] = (known or {}).get(show.situation_key(d["agent"], d["obs"], legal))
        print(json.dumps(why))
        return 0
    print(f"{unit['world']}, seed {args.seed}, {policy_words(policy)}")
    print("the episode ended with "
          + ", ".join(f"{n} {show.num(v)}" for n, v in zip(names.agents, why["returns"])))
    print()
    print("\n".join(show.why_lines(why, names, detail=bool(args.step), known=known)))
    return 0


def _situation_values(hbc: Path, policy, args):
    """The exact value of each action in each situation, when the world is small
    enough to follow every outcome; None when it is not."""
    if _too_big(hbc, args):
        return None
    cmd = ["report", str(hbc), "--rows", "100000000", "--values"]
    if policy:
        cmd += ["--policy", policy]
    if getattr(args, "exact_nodes", None) is not None:
        cmd += ["--exact-nodes", str(args.exact_nodes)]
    proc = _vm(cmd)
    if proc.returncode != 0:
        return None
    rep = json.loads(proc.stdout)
    if not rep.get("exact"):
        if rep.get("why") == "big":
            _note_too_big(hbc, args)
        return None
    return {show.situation_key(r["agent"], r["obs"], r["legal"]): r["value"]
            for r in rep["rows"] if "value" in r}


def cmd_play(args) -> int:
    """Take one agent's seat and play against the policy."""
    got = _open(args)
    if got is None:
        return 1
    path, hbc, unit = got
    names = show.Names(unit)
    if not names.agents:
        print("hunch: this world has no decisions to play", file=sys.stderr)
        return 1
    seat = args.seat or names.agents[0]
    if seat not in names.agents:
        print(f"hunch: no agent named '{seat}'; this world has "
              + ", ".join(names.agents), file=sys.stderr)
        return 1
    me = names.agents.index(seat)
    policy = find_policy(args, path)
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    acts = names.acts[seat]
    others = [n for n in names.agents if n != seat]
    print(f"{unit['world']}: you are {seat}. "
          + (f"{', '.join(others)}: {policy_words(policy)}." if others else "")
          + f" Seed {seed}.")
    print("You are shown what your agent is shown, and nothing else. q quits.")
    taken: list[int] = []
    paid = 0.0
    while True:
        cmd = ["trace", str(hbc), "--seed", str(seed), "--agent", seat,
               "--actions", ",".join(map(str, taken)), "--ask"]
        if policy:
            cmd += ["--policy", policy]
        trace = _vm_json(cmd, policy)
        if trace is None:
            return 1
        ep = trace["episodes"][0]
        got_paid = sum(step["r"].get(seat, 0.0) for step in ep["steps"])
        if got_paid != paid:
            print(f"  you are paid {show.num(got_paid - paid)}")
            paid = got_paid
        ask = ep.get("awaiting")
        if ask is None:
            break
        print()
        print(f"step {ask['step']}: you see {names.obs_labelled(me, ask['obs'], '    ')}")
        legal = ask["legal"]
        menu = "  ".join(f"[{a}] {acts[a]}" if names.named[seat] else f"[{a}]"
                         for a in range(ask["n"]) if (legal >> a) & 1)
        choice = None
        while choice is None:
            try:
                text = input(f"  your move  {menu}  > ")
            except EOFError:
                print()
                return 0
            if text.strip().lower() in ("q", "quit", "exit"):
                return 0
            choice = show.pick_action(text, acts, legal)
            if choice is None:
                print("  not one of the moves you can make here")
        taken.append(choice)
    print()
    print("the episode, with everything shown:")
    print("\n".join("  " + line for line in show.episode_lines(trace, ep, names, you=me)))
    return 0


# The kinds of note the runtime records under --strict (see vm_report in
# runtime/hunch_vm.c), in the words `hunch test` uses for them.
NOTE_TEXT = {
    "index_read": "index out of range on read, clamped",
    "index_write": "index out of range on write, clamped",
    "div_zero": "division by zero, result is 0",
    "mod_zero": "modulo by zero, result is 0",
}


def cmd_test(args) -> int:
    """Compile a world and run it a few times with an untrained policy.

    The point is to catch a broken rule before training on it. A world that
    fails its own assertions exits here in a second instead of producing a
    plausible-looking learning curve.
    """
    path = resolve_world(args.file)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2
    if args.episodes < 1:
        print("hunch: test needs at least one episode", file=sys.stderr)
        return 2
    try:
        hbc, unit = compile_file(path, parse_params(args.param))
    except HunchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    # What the compiler made of the file: each agent, what it is shown and
    # what it can do. A blank that sees the wrong thing shows up here.
    print(describe(unit))
    cmd = ["trace", str(hbc), "--seed", str(args.seed),
           "--episodes", str(args.episodes), "--strict"]
    proc = _vm(cmd)
    if proc.returncode != 0:
        print(proc.stderr.strip(), file=sys.stderr)
        return proc.returncode
    report = json.loads(proc.stdout)
    totals = [sum(sum(step["r"].values()) for step in ep["steps"])
              for ep in report["episodes"]]
    mean = sum(totals) / len(totals) if totals else 0.0
    # --strict makes the runtime say what actually executed, as one JSON line
    # on stderr. Counting the asserts in the source would call an assertion
    # "held" when the episodes never reached it.
    try:
        strict = json.loads(proc.stderr.strip().splitlines()[-1])
        asserts, notes = strict["asserts"], strict["notes"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        print("hunch: the runtime returned no test report", file=sys.stderr)
        return 1
    if asserts:
        ran = [a for a in asserts if a["runs"] > 0]
        idle = sorted({a["line"] for a in asserts if a["runs"] == 0})
        noun = "assertion" if len(asserts) == 1 else "assertions"
        verdict = f"{len(ran)} of {len(asserts)} {noun} ran"
        if ran:
            verdict += " and held"
        if idle:
            where = ("line " if len(idle) == 1 else "lines ") + ", ".join(map(str, idle))
            verdict += f", {len(asserts) - len(ran)} never ran ({where})"
    else:
        # Do not imply a check that never ran.
        verdict = "no assertions in this world, so nothing was checked"
    whose = "mean reward" if len(unit["agents"]) < 2 else "mean reward over all agents"
    print(f"{path}: {len(totals)} episode(s), {verdict}, "
          f"{whose} {mean:+.3f}")
    # The language defines these (clamp the index, yield 0), so they are not
    # failures. They are reported because they usually mean a bug in the world.
    for note in sorted(notes, key=lambda n: (n["line"], n["col"], n["kind"])):
        what = NOTE_TEXT.get(note["kind"], note["kind"])
        times = "1 time" if note["count"] == 1 else f"{note['count']} times"
        print(f"warning: line {note['line']}, column {note['col']}: {what} ({times})")
    if strict.get("lost"):
        print(f"warning: {strict['lost']} more event(s) did not fit in the report, "
              "so the counts above are incomplete")
    return 0


def cmd_train(args) -> int:
    """Fill in the blanks, then say what they learned and what it is worth."""
    got = _open(args)
    if got is None:
        return 1
    path, hbc, unit = got
    if not unit["agents"]:
        print(f"hunch: {path} has no `hunch` blank, so there is nothing to train; "
              "`hunch run` plays it", file=sys.stderr)
        return 1
    out = args.out or str(policy_beside(path))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["train", str(hbc), "--algo", args.algo, "--seed", str(args.seed), "--out", out,
           "--eval-episodes", str(args.eval_episodes), "--json", "--progress"]
    known_big = _too_big(hbc, args)
    exact_nodes = 0 if known_big else args.exact_nodes
    for flag, value in (("--budget", args.budget), ("--lr", args.lr),
                        ("--entropy", args.entropy), ("--fork-points", args.fork_points),
                        ("--fork-ratio", args.fork_ratio), ("--rollout", args.rollout),
                        ("--target", args.target), ("--max-sec", args.max_sec),
                        ("--think", args.think), ("--playouts", args.playouts),
                        ("--games", args.games), ("--train-sec", args.train_sec),
                        ("--cores", args.cores), ("--exact-nodes", exact_nodes)):
        if value is not None:
            cmd += [flag, str(value)]
    quiet = args.json or not unit["agents"]
    first = unit["agents"][0] if unit["agents"] else ""
    ensure_vm()
    proc = subprocess.Popen([str(VM_BIN)] + cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    errors = []
    started = time.time()
    shown = 0.0
    for line in proc.stderr:
        parts = line.split()
        if len(parts) == 7 and parts[0] == "games":
            # self-play: a line every half minute or so
            sec = float(parts[2])
            if quiet or sec - shown < SELFPLAY_EVERY:
                continue
            shown = sec
            print(f"  {int(parts[1]):>8,} games of self-play  ({show.clock(sec)})", flush=True)
            continue
        if len(parts) == 3 and parts[0] in ("step", "gap"):
            # Progress is for runs long enough to wonder about.
            if quiet or time.time() - started < 1.0:
                continue
            if parts[0] == "step":
                print(f"  {int(float(parts[1])):>12,} steps   {first} collects "
                      f"{float(parts[2]):+.4f}", flush=True)
            else:
                what = ("an agent changing alone could gain" if len(unit["agents"]) > 1
                        else "a better policy was found, gaining")
                print(f"  {int(parts[1]):>12,} rounds  {what} {float(parts[2]):.4f}",
                      flush=True)
        else:
            errors.append(line)
    stdout = proc.stdout.read()
    if proc.wait() != 0:
        print("".join(errors).strip(), file=sys.stderr)
        return proc.returncode
    train = json.loads(stdout.strip().splitlines()[-1])
    if train.get("not_whole", "").startswith("big:"):
        _note_too_big(hbc, args)
    elif known_big and train.get("not_whole") == "off":
        # An earlier command found the world too big, so this one did not try.
        train["not_whole"] = f"big:{_exact_limit(args)}"
    policy = out if train["algo"] != "random" else None
    if train.get("method") == "mcts" and args.score_sec is None:
        # Scoring a search is playing it, and that is slow: a short sample
        # here, and `hunch score` for a longer one.
        args.score_sec = TRAIN_SCORE_SEC
    rep = _report(hbc, policy, args)
    if rep is None:
        return 1
    if args.json:
        print(json.dumps({"train": train, "report": rep, "policy": policy}))
        return 0
    print(f"{unit['world']}: {show.how_trained(train)}")
    print("\n".join(show.report_lines(rep, show.Names(unit), args.rows)))
    if policy:
        print()
        print(f"saved {policy}   (hunch run, why and play use it for {path})")
    return 0


ALGOS = ["auto", "regret", "mcts", "ppo", "fork", "reinforce", "random"]
TRAIN_SCORE_SEC = 5.0       # how long `train` plays a search to score it; `score` takes 15 s
SELFPLAY_EVERY = 30.0       # seconds between the progress lines of self-play


def cmd_studio(args) -> int:
    if not (ROOT / "studio").exists():
        print("hunch: the Studio runs from a source checkout: "
              "git clone https://github.com/knownquantity/Hunch", file=sys.stderr)
        return 1
    from . import studio_server
    return studio_server.main(["--port", str(args.port), "--host", args.host])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

POLICY_HELP = ("a policy file; default: the .hpol beside the world if `hunch train` left "
               "one; 'none' for random play")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hunch",
        description="Hunch: write the rules, leave the decisions blank.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("test", help="compile a world, show what each agent sees, and check its rules")
    sp.add_argument("file")
    sp.add_argument("--episodes", type=int, default=3)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--param", action="append", metavar="NAME=VALUE")
    sp.set_defaults(func=cmd_test)

    sp = sub.add_parser("train", help="learn the blanks")
    sp.add_argument("file")
    sp.add_argument("--param", action="append", metavar="NAME=VALUE")
    sp.add_argument("--algo", default="auto", choices=ALGOS,
                    help="auto (default): the exact answer when the world is small enough "
                         "to follow whole; else a search at play time (mcts) when sampling "
                         "finds nothing hidden from the agents; else ppo for one agent and "
                         "regret for several")
    sp.add_argument("--think", type=float, metavar="SECONDS",
                    help="mcts: playouts per decision are chosen to take about this long on "
                         "this machine (default 0.1), then saved in the policy")
    sp.add_argument("--playouts", type=int, help="mcts: playouts per decision, set directly")
    sp.add_argument("--train-sec", type=float, metavar="SECONDS",
                    help="mcts: how long the search plays itself to train the network that "
                         "guides it (default 600)")
    sp.add_argument("--games", type=int,
                    help="mcts: games of self-play, instead of a time (rounded up to rounds of "
                         "64); 0 for no network, a search with random playouts")
    sp.add_argument("--cores", type=int,
                    help="mcts: processes that play self-play games at once (default: every core)")
    sp.add_argument("--score-sec", type=float, metavar="SECONDS",
                    help="mcts: how long the search is played to score it (default 5)")
    sp.add_argument("--budget", type=int,
                    help="environment steps (default 200,000; 2,000,000 for sampled regret)")
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--out", help="where the policy goes (default: beside the world, .hpol)")
    sp.add_argument("--eval-episodes", type=int, default=128)
    sp.add_argument("--rows", type=int, default=6, help="situations listed per agent")
    sp.add_argument("--target", type=float,
                    help="exact regret stops when no agent can gain more than this alone "
                         "(default 0.001)")
    sp.add_argument("--max-sec", type=float, help="and after this long (default 60)")
    sp.add_argument("--exact-nodes", type=int,
                    help="largest world followed whole, in positions (default 2,000,000)")
    sp.add_argument("--lr", type=float)
    sp.add_argument("--entropy", type=float)
    sp.add_argument("--fork-points", type=int)
    sp.add_argument("--fork-ratio", type=float)
    sp.add_argument("--rollout", type=int)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("run", help="play an episode and print it step by step")
    sp.add_argument("file")
    sp.add_argument("--param", action="append", metavar="NAME=VALUE")
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--episodes", type=int, default=1)
    sp.add_argument("--policy", help=POLICY_HELP)
    sp.add_argument("--agent", help="script this agent's actions with --actions")
    sp.add_argument("--actions", help="comma separated actions for --agent")
    sp.add_argument("--json", action="store_true", help="print the raw trace as JSON")
    sp.add_argument("--forks", action="store_true",
                    help="with the JSON trace: a sampled value for every action at every decision")
    sp.add_argument("--out", help="write the JSON trace to this file")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("why", help="replay each decision of an episode once per action")
    sp.add_argument("file")
    sp.add_argument("--param", action="append", metavar="NAME=VALUE")
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--policy", help=POLICY_HELP)
    sp.add_argument("--step", type=int, help="show this step's decisions in full")
    sp.add_argument("--samples", type=int, default=200,
                    help="continuations tried per action (default 200)")
    sp.add_argument("--exact-nodes", type=int,
                    help="positions followed for the exact values (default 2,000,000; 0 for none)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_why)

    sp = sub.add_parser("play", help="take an agent's seat and play against the policy")
    sp.add_argument("file")
    sp.add_argument("--param", action="append", metavar="NAME=VALUE")
    sp.add_argument("--as", dest="seat", metavar="AGENT", help="the agent you play (default: the first)")
    sp.add_argument("--seed", type=int, help="default: a new one each game")
    sp.add_argument("--policy", help=POLICY_HELP)
    sp.set_defaults(func=cmd_play)

    sp = sub.add_parser("score", help="what a policy collects and what it does")
    sp.add_argument("file")
    sp.add_argument("--param", action="append", metavar="NAME=VALUE")
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--policy", help=POLICY_HELP)
    sp.add_argument("--rows", type=int, default=6, help="situations listed per agent")
    sp.add_argument("--episodes", type=int, help="episodes sampled when it cannot be exact")
    sp.add_argument("--score-sec", type=float, metavar="SECONDS",
                    help="a search policy is scored by playing it for about this long "
                         "(default 15; 0 for no limit, with --episodes)")
    sp.add_argument("--exact-nodes", type=int,
                    help="largest world scored exactly, in positions (default 2,000,000)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("grade", help="grade any outside agent: a program or written answers")
    sp.add_argument("file")
    sp.add_argument("--param", action="append", metavar="NAME=VALUE")
    sp.add_argument("--agent", help="the seat being graded (default: the first agent)")
    sp.add_argument("--cmd", help="a program asked one JSON question per line on stdin; it "
                    "prints one answer per line, flushed: an action name or JSON probabilities")
    sp.add_argument("--timeout", type=float, default=120.0,
                    help="seconds --cmd may take over one answer (default 120)")
    sp.add_argument("--questions", action="store_true",
                    help="print every decision the agent can face, one JSON question per line")
    sp.add_argument("--answers", help="a file of answers to --questions, one per line "
                    "(in order, or JSON with id and answer)")
    sp.add_argument("--policy", help="what the other agents play. " + POLICY_HELP)
    sp.add_argument("--episodes", type=int, default=2000,
                    help="episodes played when the world is too big to grade exactly")
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--exact-nodes", type=int,
                    help="largest world graded exactly, in positions (default 2,000,000)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_grade)

    sp = sub.add_parser("studio", help="serve the interactive Hunch Studio")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--host", default="127.0.0.1")
    sp.set_defaults(func=cmd_studio)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except HunchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0
