#!/usr/bin/env python3
"""How close is a trained policy to perfect play? Measure it on a board small
enough to solve.

    python3 examples/perfect_play.py              # about 13 minutes, 3.6 GB
    python3 examples/perfect_play.py 100 120      # 100 games a seat, 2 minutes of self-play

worlds/four.hunch is four-in-a-row on a 5 by 4 board. `hunch train` treats it
as too big to solve and gives it a search that learns by playing itself. Asked
to (`exact_nodes`), Hunch also solves it outright: with best play by both it
is a draw. So the search can be played against an opponent that never makes a
mistake, and every game it does not draw is a game it lost.

The perfect opponent here picks at random among the moves that keep the
game's value, so the games differ. A move's value is read off the solved
policy: play the move in a copy of the game, let the solved policy finish it
for both sides, and see how it ends.
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hunch                                         # noqa: E402

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TRAIN_SEC = float(sys.argv[2]) if len(sys.argv) > 2 else None    # None: train's own default

world = hunch.example("four")
out = Path(hunch.__file__).resolve().parent.parent / ".hunch-cache"
out.mkdir(exist_ok=True)

start = time.time()
solved = world.train(exact_nodes=6_000_000, out=out / "four.solved.hpol")
print(f"solved: {solved.value}   ({time.time() - start:.0f}s)", flush=True)
assert solved.proven, "the board was not solved exactly"

start = time.time()
learned = world.train(out=out / "four.learned.hpol", train_sec=TRAIN_SEC)
print(f"trained by self-play   ({time.time() - start:.0f}s)", flush=True)

study = world.env(agents=world.agents)               # a copy of the game, to try moves in


def value(history, move, seat):
    """What `seat` collects after `history` and then `move`, with the solved
    policy playing both sides from there."""
    study.reset(1)
    for m in history + [move]:
        study.act(m)
    while not study.over:
        study.act(solved.act(study.agent, study.obs, study.legal))
    return study.returns[seat]


def column(before, after):
    """The column of the one disc that `after` has and `before` does not."""
    (cell,) = [i for i in range(len(after)) if before[i] != after[i]]
    return cell % 5


rng = random.Random(7)
for mine in world.agents:
    theirs = next(a for a in world.agents if a != mine)
    env = world.env(agents=[theirs], policy=learned)  # the search plays `mine`
    won = drawn = lost = 0
    for seed in range(1, GAMES + 1):
        env.reset(seed)
        history, seen = [], [0] * 20
        while not env.over:
            board = [int(v) for v in env.state["board"]]
            if board != seen:                          # the search has moved
                history.append(column(seen, board))
            worth = {m: value(history, m, theirs) for m in env.legal}
            move = rng.choice([m for m in env.legal if worth[m] == max(worth.values())])
            history.append(move)
            env.act(move)
            seen = board[:]
            seen[sum(1 for r in range(4) if board[r * 5 + move]) * 5 + move] = 2 - (mine == "o")
        result = env.returns[mine]
        won, drawn, lost = won + (result > 0), drawn + (result == 0), lost + (result < 0)
    print(f"the trained search as {mine} against perfect play: "
          f"won {won}, drew {drawn}, lost {lost} of {GAMES}", flush=True)
