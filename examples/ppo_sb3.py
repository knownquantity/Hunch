#!/usr/bin/env python3
"""Train a Stable-Baselines3 PPO agent in a Hunch world, then grade it exactly.

    pip install stable-baselines3 gymnasium
    python3 examples/ppo_sb3.py

The world is Kuhn poker. PPO sits in alice's seat and plays against the solved
policy for bob, through the gymnasium environment `world.gym` returns. A
learning curve cannot say how good the result is; `world.grade` can. It asks
the trained network what it does in each of alice's six situations and weighs
every deal and every reply, so the numbers below have no sampling error:
what the agent collects, how much the best reply to it would take, and which
decisions cost it most.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                   # noqa: E402
from stable_baselines3 import PPO                    # noqa: E402

import hunch                                         # noqa: E402

world = hunch.example("kuhn")
solved = world.train()                               # CFR+ over the whole game, milliseconds
print(solved, "\n")

env = world.gym("alice", policy=solved)              # you are alice; bob plays the solved policy
model = PPO("MlpPolicy", env, n_steps=512, batch_size=128, ent_coef=0.01, seed=1, verbose=0)


def alice(obs, legal):
    """What the network does where alice is shown `obs`: one probability per action."""
    tensor, _ = model.policy.obs_to_tensor(np.asarray(obs, dtype=np.float32))
    return model.policy.get_distribution(tensor).distribution.probs[0].tolist()


print("before training")
print(world.grade({"alice": alice}, against=solved), "\n")
model.learn(total_timesteps=int(sys.argv[1]) if len(sys.argv) > 1 else 60_000)
print("after training")
print(world.grade({"alice": alice}, against=solved))
