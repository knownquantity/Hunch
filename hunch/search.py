"""Search at play time, for worlds too big to solve whole.

`hunch train` follows every position of a world when it can. Connect Four has
about 4.5 trillion, so there the other use of a world applies: fork it at the
decision in front of you, try each action many times, and play the one that
did best. This is Monte Carlo tree search (UCT) written against nothing but
the public `Env` calls: snapshot, restore, reseed, act, memory and playout.

    world = hunch.World("worlds/connect4.hunch")
    env = world.env()
    env.reset(1)
    search = hunch.Search(world, playouts=2000)
    while not env.over:
        env.act(search.choose(env))

The search reads the world's whole state, so it is fair only where the agents
are shown everything (board games). In a world with hidden cards it would be
looking at them.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Optional

from .world import Env, World


class Search:
    """playouts   simulations per decision: more is stronger and slower
    explore    how readily a less-tried action is tried again
    """

    def __init__(self, world: World, playouts: int = 1000, explore: float = 1.0, seed: int = 1):
        self.world, self.playouts, self.explore = world, playouts, explore
        self._rng = random.Random(seed)
        self._env = world.env()                     # every agent decided here
        # What each action did when last searched: {action: (tries, mean value)}
        self.tried: Dict[int, tuple] = {}

    def choose(self, env: Env, playouts: Optional[int] = None) -> int:
        """The action to play at the decision `env` is waiting at. `env` is
        not changed."""
        if env.over:
            raise ValueError("the episode is over: there is nothing to choose")
        if env.horizon != self._env.horizon:        # search the episode `env` is playing
            self._env = self.world.env(horizon=env.horizon)
        root = env.snapshot()
        sim, rng = self._env, self._rng
        agents = self.world.agents
        tree: dict = {}             # memory -> [visits, agent, {action: [tries, total]}]
        low, high = math.inf, -math.inf
        for _ in range(playouts or self.playouts):
            sim.restore(root)
            sim.reseed(rng.getrandbits(63))
            path = []
            while not sim.over:
                key = sim.memory()
                node = tree.get(key)
                if node is None:                    # a new position: add it, then play out
                    tree[key] = node = [0, sim.agent, {a: [0, 0.0] for a in sim.legal}]
                    path.append((node, self._pick(node, low, high, rng)))
                    sim.act(path[-1][1])
                    sim.playout()
                    break
                action = self._pick(node, low, high, rng)
                path.append((node, action))
                sim.act(action)
            got = sim.returns
            for a in agents:
                low, high = min(low, got[a]), max(high, got[a])
            for node, action in path:
                node[0] += 1
                stat = node[2][action]
                stat[0] += 1
                stat[1] += got[node[1]]
        stats = tree[env.memory()][2]
        self.tried = {a: (n, w / n if n else 0.0) for a, (n, w) in stats.items()}
        return max(stats, key=lambda a: (stats[a][0], stats[a][1]))

    def _pick(self, node, low: float, high: float, rng) -> int:
        """UCB1 over the node's actions, with values scaled to what has been
        seen so far."""
        visits, _, stats = node
        fresh = [a for a, (n, _) in stats.items() if n == 0]
        if fresh:
            return rng.choice(fresh)
        span = (high - low) or 1.0
        log_n = math.log(visits + 1)
        best, best_score = None, -math.inf
        for a, (n, w) in stats.items():
            score = (w / n - low) / span + self.explore * math.sqrt(log_n / n)
            if score > best_score:
                best, best_score = a, score
        return best
