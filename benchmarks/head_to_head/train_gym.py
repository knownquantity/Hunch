"""Train a policy for cache_gym.py and report its hit rate.

This is the second file the Python side needs. The Hunch side needs no second
file: `./bin/hunch train worlds/cache.hunch --budget 200000` is the whole
training program, because the trainer is part of the language runtime.
"""

from __future__ import annotations

import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from cache_gym import CacheEnv


BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
EPISODES = 96


def evaluate(model, seed: int = 987654321) -> float:
    env = CacheEnv(seed=seed)
    hits = []
    for e in range(EPISODES):
        obs, _ = env.reset(seed=seed + e * 7919)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, _, info = env.step(int(action))
        hits.append(info["hits"])
    return float(np.mean(hits))


def main() -> None:
    env = Monitor(CacheEnv(seed=1))
    model = PPO("MlpPolicy", env, verbose=0, seed=1,
                n_steps=2048, batch_size=256, learning_rate=3e-4)
    model.learn(total_timesteps=BUDGET)
    print(f"budget={BUDGET} mean_hits={evaluate(model):.2f} / 60")


if __name__ == "__main__":
    main()
