"""The cache-eviction world as a standard Gymnasium environment.

This is the Python side of the head-to-head in this directory. It is written the
way a competent user would write it: a Gymnasium Env subclass, numpy arrays,
same rules and same 17-feature observation as worlds/cache.hunch.

Nothing here is deliberately awkward. If this file is shorter than the Hunch
version, or clearer, that is the answer to the question the comparison asks.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


SLOTS = 4
REQUESTS = 60
HOT_A = 5          # keys 1..5 are hot for the first half
HOT_B = 5          # keys 6..10 take over halfway through
SWITCH_AT = 30
SCAN_PERIOD = 12
SCAN_START = 9     # requests 9..11 of every period are one-touch scan keys


class CacheEnv(gym.Env):
    """A four-slot cache that must choose its own eviction policy.

    A hit pays 1. On a miss with the cache full, the agent picks the slot to
    evict. The hot set moves halfway through, and a three-request scan of
    one-touch keys arrives every twelfth request, so neither LRU nor LFU is
    right.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.observation_space = spaces.Box(0.0, 10.0, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Discrete(SLOTS)
        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # -- world construction -------------------------------------------------
    def _make_requests(self) -> np.ndarray:
        req = np.empty(REQUESTS, dtype=np.int64)
        for i in range(REQUESTS):
            if i < SWITCH_AT:
                req[i] = 1 + int(self._rng.random() * self._rng.random() * HOT_A)
            else:
                req[i] = 6 + int(self._rng.random() * self._rng.random() * HOT_B)
            if i % SCAN_PERIOD >= SCAN_START:
                req[i] = 200 + i
        return req

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.req = self._make_requests()
        self.key = np.full(SLOTS, -1, dtype=np.int64)
        self.last = np.zeros(SLOTS, dtype=np.int64)
        self.freq = np.zeros(SLOTS, dtype=np.int64)
        self.cursor = 0
        self.hits = 0
        return self._obs(), {}

    # -- mechanics ----------------------------------------------------------
    def _obs(self) -> np.ndarray:
        age = self.cursor - self.last
        mean_age = age.mean()
        mean_freq = self.freq.mean()
        # after the final step the cursor is one past the end; the terminal
        # observation is not used, but it still has to be well defined
        want = self.req[min(self.cursor, REQUESTS - 1)]
        return np.array([
            *age / 60.0,
            *(age - mean_age) / 30.0,
            *self.freq / 20.0,
            *(self.freq - mean_freq) / 10.0,
            1.0 if want > 10 else 0.0,
        ], dtype=np.float32)

    def step(self, action: int):
        want = int(self.req[self.cursor])
        hit = np.flatnonzero(self.key == want)
        reward = 0.0

        if hit.size:
            slot = int(hit[-1])          # last match, as the Hunch world does
            self.hits += 1
            reward = 1.0
            self.freq[slot] += 1
            self.last[slot] = self.cursor
        else:
            empty = np.flatnonzero(self.key == -1)
            slot = int(empty[-1]) if empty.size else int(action)
            self.key[slot] = want
            self.last[slot] = self.cursor
            self.freq[slot] = 1

        self.cursor += 1
        terminated = self.cursor >= REQUESTS
        return self._obs(), reward, terminated, False, {"hits": self.hits}

    def hit_rate(self) -> float:
        return self.hits / REQUESTS


def make(seed: int | None = None) -> CacheEnv:
    return CacheEnv(seed=seed)
