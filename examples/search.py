#!/usr/bin/env python3
"""Play a world too big to solve by forking it: Connect Four, no training.

    python3 examples/search.py

`hunch.Search` is Monte Carlo tree search written against the public Env
calls (snapshot, restore, reseed, act, playout). Nothing here knows the rules
of Connect Four: they are in worlds/connect4.hunch, and the same search plays
any world where nothing is hidden.
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hunch                                         # noqa: E402

world = hunch.example("connect4")


def match(playouts_a: int, playouts_b: int, games: int, seed: int):
    """A against B, swapping who moves first. 0 playouts: uniform random play.
    Returns A's wins, draws and losses."""
    a = hunch.Search(world, playouts_a, seed=1) if playouts_a else None
    b = hunch.Search(world, playouts_b, seed=2) if playouts_b else None
    rng, env, results = random.Random(seed), world.env(), []
    for game in range(games):
        seat_a = world.agents[game % 2]
        env.reset(seed + game)
        while not env.over:
            mover = a if env.agent == seat_a else b
            env.act(mover.choose(env) if mover else rng.choice(env.legal))
        results.append(env.returns[seat_a])
    return results.count(1), results.count(0), results.count(-1)


for a, b, games in ((300, 0, 30), (1500, 150, 20)):
    start = time.time()
    won, drawn, lost = match(a, b, games, seed=100)
    print(f"search with {a} playouts a move against "
          f"{f'search with {b}' if b else 'random play'}: "
          f"won {won}, drew {drawn}, lost {lost}   ({time.time() - start:.0f}s)")
