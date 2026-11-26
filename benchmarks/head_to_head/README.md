# Head-to-head: Hunch versus Python, Gymnasium and Stable-Baselines3

The question: is a Hunch world actually a better way to write and train an
environment, or only a different one?

One world, built twice, by the same person, on the same machine. The Hunch
version is the shipped `worlds/cache.hunch`. The Python version is
`cache_gym.py` plus `train_gym.py` in this directory, written the way a
competent user would write it: a Gymnasium `Env` subclass, numpy state, the
same 17-feature observation, the same rules, Stable-Baselines3's PPO.

Neither side is a strawman. The Python version is clean and idiomatic. The
Hunch version is the one the project already ships.

## Results

| | Hunch | Python + Gymnasium + SB3 |
|---|---:|---:|
| Lines of code to define the world (+ trainer) | **93** | 119 |
| Files | **1** | 2 |
| Installs before it runs | **0** | 2 (gymnasium, stable-baselines3, which pulls torch) |
| Wall clock to a trained policy, 200k steps | **3.1 s** | 180.4 s |
| Hits per episode after 200k steps | 29.69 | **31.82** |
| Hits per episode after 2M steps (Hunch only) | 31.80 | not run |
| Wall clock to about 31 hits | **33 s** (2M steps) | 180 s (200k steps) |

Environment-only comparison for speed, 200 episodes: Hunch 0.54 s, Python
0.59 s. The simulation speed is the same. The difference is the trainer: Hunch's
is compiled C, SB3's is Python and torch with per-step overhead.

## What each number means

**Authoring: a modest win, not 10x.** 93 lines against 119, one file against
two. The Hunch side does not need a training script at all, which is the real
saving: the trainer is part of the language runtime. But a 1.3x line-count
difference is not a reason to learn a new language.

**Iteration speed: the strongest result.** 58x faster to a trained policy.
That is large enough to change how you work: you can try ten ideas in the time
one Python run takes. Most of it is that 200k steps of a 60-step environment
takes three seconds of compiled C and three minutes of Python.

**Sample efficiency: Hunch loses by about 10x.** SB3 reaches 31.82 hits in
200k steps; Hunch needs about 2M to reach 31.80. Its built-in PPO is not
competitive with a mature library per sample.

**Time to the same result: Hunch still wins by about 5x.** Those 2M steps take
33 seconds, against 180 seconds for SB3's 200k. Each step is roughly 60 times
cheaper, which more than pays for needing ten times as many. So the honest
statement is not "Hunch learns worse"; it is "Hunch needs more samples but gets
there faster". Sample efficiency is the metric that matters when a sample is
expensive, such as a real robot or a language-model call. For a simulated
world, wall clock is the metric, and Hunch wins it.

## What I tried to close the sample-efficiency gap

All at the same budget, on the cache world, three seeds each:

| change | hits at 200k |
|---|---:|
| default (learning rate 3e-3, 4 epochs) | 29.57 |
| learning rate 1e-3 | 27.92 |
| learning rate 3e-4 | 28.18 |
| 8 epochs | **30.25** |
| 16 epochs | 29.94 |

Eight epochs helps the cache, is neutral on the market (2.258 against 2.243)
and makes the dungeon much worse (0.395 against 0.757, two seeds collapsing
instead of one). So it is not a default worth changing.

Moving the gradient clip after the minibatch average, which is where a mature
implementation puts it, changed the results by exactly zero: the gradient norm
never reaches the threshold, so the clip never fires either way.

Conclusion: no single knob closes the gap. Matching SB3 would mean
reimplementing its update properly, not retuning this one. Given that Hunch is
already faster in wall clock, that work is not the best use of the next day.

**Inspection: Hunch is the only one that answers the question.** `hunch run
--forks` returns, for every decision, the measured value of each action the
agent could have taken. In the Gymnasium path there is no equivalent; you would
write a snapshot-and-replay harness yourself, with matched random numbers, and
that is exactly the machinery the language owns.

## Honest caveats

- Hyperparameters differ because each tool has its own defaults: Hunch's PPO
  uses a 3e-3 learning rate, SB3's uses 3e-4. Neither was tuned for this world,
  so the sample-efficiency gap is the gap between the two as shipped, not a
  controlled measurement of the algorithms.
- The two environments implement the same rules but are not bit-identical. The
  request generator and the step counter differ in small ways.
- One seed, one run. The Hunch side is deterministic; the SB3 side is not.
- Wall clock includes framework overhead, which is part of the workflow being
  measured, but it would look different on a GPU with a large network.

## Conclusion

The claim that survives is not "better RL results". It is **faster iteration
and better visibility**: one file, no setup, three seconds to a trained policy,
and per-action values at every decision for free. Where final performance per
sample matters, a mature library wins and this comparison says so.

## Reproduce

```bash
./bin/hunch train worlds/cache.hunch --algo ppo --budget 200000 --seed 1
pip install gymnasium stable-baselines3
cd benchmarks/head_to_head && python3 train_gym.py 200000
```
