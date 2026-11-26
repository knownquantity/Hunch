# The fork experiments: forking against PPO on equal budgets

This is the record of the experiments Hunch started from: training the blanks
with a neural network, and asking whether replaying a decision once per action
(`--algo fork`) beats PPO at the same number of environment steps. It mostly
does not, and the sections below say where and why. They are kept as written.

The numbers were measured at commit `06c2ab3`. The network learners have not
changed since: `ppo` and `fork` give bit-identical results at that commit and
now on the dungeon, the duel and the market (60,000 steps, seed 3). The tables
were not re-run in full.

One conclusion below was acted on. The rock-paper-scissors section ends by
saying that an unexploitable policy needs regret matching over the values a
fork already computes, not a better gradient. `hunch train` now does that for
worlds with more than one agent; the main README has the results.

Reproduce with:

```bash
python3 benchmarks/run_all.py             # the five-seed tables (~3 min)
python3 benchmarks/cache_baselines.py     # LRU / LFU / Belady comparison
python3 benchmarks/action_width.py        # the 4 to 32 action sweep
```

---

## Worlds

Four benchmark worlds (six games in `worlds/`), each trained from scratch on **identical environment-step budgets**
across **5 seeds**. Return is scored on a fixed stream of evaluation episodes
shared by every algorithm, so nobody wins on luck. `mean ± std` over seeds.

### Dungeon: 400k steps, 50-step horizon

Find a key, avoid a trap, open the exit. The payoff arrives a dozen decisions
after the ones that caused it.

| algorithm | return |
|---|---|
| random | −0.069 |
| REINFORCE | **1.1099 ± 0.0131** |
| PPO | 1.0482 ± 0.1457 |
| hunch fork | 0.9066 ± 0.4384 |

These numbers replaced an earlier set in which both learned methods sat at the
ceiling. That set was produced by a compiler bug: a helper call inside the
observation overwrote the two direction features, so the policy was navigating
by `x + 1` and `y` instead of the vector the source declares. With the
observation it actually writes, the world is harder, fork trails PPO by 13% and
has a seed spread eight times PPO's, and plain REINFORCE wins.

### Market maker: 600k steps, 48-tick episodes

Quote a market while carrying inventory. Reward is equity change minus a charge
for risk. Noise is the enemy: two branches of the same decision see different
coin flips unless you hold the randomness fixed.

| algorithm | return |
|---|---|
| random | 1.411 |
| REINFORCE | 0.841 ± 1.915 |
| PPO | **2.2520 ± 0.0546** |
| hunch fork | 2.2126 ± 0.1181 |

### Bluffing duel: 600k steps, pure self-play

Two agents, hidden cards, zero human data. Raw return is meaningless here
because both sides improve together, so the honest test is against opponents
that cannot adapt:

| algorithm | raw return | vs. always-fold | vs. always-call |
|---|---|---|---|
| random | 0.385 | n/a | n/a |
| REINFORCE | 0.092 | 0.694 | 0.258 |
| PPO | 0.094 | 0.296 | 0.172 |
| hunch fork | 0.165 | **0.489** | 0.254 |

Fork leads by 1.76x here, which is not a demonstrated difference: see the
scorecard below. The table above is also not a fair algorithm comparison,
because PPO was left at its defaults while fork's five knobs were tuned on
these worlds. Raising PPO's learning rate from 3e-3 to 1e-3, a single flag,
puts PPO above fork on the benchmark stream and on a fresh 32-stream
evaluation.

The policy shipped in `policies/duel.hpol` is not the balanced strategy this
section used to claim. Over 200 traced decisions it bets every hand and never
checks: 74 bets of 1, 126 bets of 2, zero checks. That scores +1.00 against a
player who always folds, because a folder forfeits, and about zero against a
caller, which is the showdown value of a random hand. It is a pure strategy and
maximally exploitable by a caller, which is the same failure the
rock-paper-scissors section describes. Its raw return is also below both
five-seed means. It is an artifact of a seed that found a corner, not an
example of self-play working.

### Cache eviction: 2M steps, a real systems decision

At equal budget the two methods are within 0.4% of each other: PPO 31.7979 ±
0.5589, fork 31.6771 ± 0.3051. The interesting comparison is not PPO at all but
the textbook policies:

A four-slot cache on a workload with hot keys, scan traffic, and a hot set that
moves halfway through. Replayed on the same 96 evaluation episodes:

| policy | hits / 60 | hit rate |
|---|---|---|
| LRU | 27.86 | 46.4% |
| LFU | 28.10 | 46.8% |
| **hunch (trained)** | **31.74** | **52.9%** |
| Belady OPT (clairvoyant) | 36.09 | 60.2% |

The agent was never told what LRU is. It beats both classic baselines by 3.6–3.9
hits per episode, a 14% relative improvement in hit rate, and gets 88% of the
way to an oracle that can see the future.

---

## What is actually new here

The mechanism is not. Restoring a simulator, replaying one branch per action
under shared randomness, and using the siblings as a baseline has been
published at least three times: VinePPO (ICML 2025) resets to intermediate
states for Monte Carlo value estimates; Branching Policy Optimization
(arXiv:2607.14171, July 2026) forks K actions from sandbox snapshots and proves
a variance reduction; Counterfactual Rollout Replay (arXiv:2609.33875, NeurIPS
2026) replaces an advantage with the terminal-return contrast of a replayed
branch. Common random numbers for policy comparison go back to PEGASUS (2000).
Counterfactual baselines for multi-agent credit assignment go to COMA (2018).
Holes in a program filled by a learner go to ALisp (2000), and a language for
describing worlds to train in is VGDL (2013). Emulators that snapshot a flat
state block with the PRNG inside it are ALE and MuJoCo.

What is left is narrower, and it is a claim about the language rather than the
algorithm. The decision point is a source-level hole, and the compiler owns the
state that gets replayed, so a counterfactual comparison needs no environment
API, no learned model and no noise inference. That is an integration property,
not a new estimator, and nothing here shows it beats PPO in general.

## Honest scorecard

PPO is an extremely strong baseline. On all three single-agent worlds fork now
trails it, by 0.4% on the cache, 2.6% on the market and 13% on the dungeon, at
equal budgets. If the claim were "fork beats PPO", this would be a failure, and
it is where the evidence points. Here is the rest of what the numbers say.

**On the duel, fork leads and the lead is not established.** Five seeds give
fork a mean of 0.165 against PPO's 0.094, a factor of 1.76. With this much
variance that is a Welch t of 1.39 and an exact permutation p of 0.19, so the
honest reading is that the experiment has not shown a difference. Against fixed
opponents fork beats a folder more often than PPO does (0.489 against 0.296) and
loses more to a caller (0.254 against 0.172), which is the shape of a more
aggressive policy rather than a better balanced one.

**Why that number changed.** An earlier revision of this document reported 0.275
for fork and called it nearly three times PPO. That measurement predates the fix
that made every decision consume exactly one random draw, which shifted every
stream in the runtime; the benchmark file was never regenerated, so a figure
produced by the old code stayed in the document after the code changed. Rerunning
the committed harness on the committed code gives 0.165. Nothing in the learner
was tuned to produce the new number, and the old one is not available to claim.

**The rework that got there.** The first version of this system had fork at
0.885 on the dungeon, 2.019 on the market, 30.0 on the cache and 0.133 on the
duel. Three changes moved it:

| change | why |
|---|---|
| anneal the fork budget (`--fork-ratio 1.5 --fork-decay 0.5`) | forking buys credit assignment, and it is worth most while the critic is still wrong. Spend it in the first half, then get out of the way. |
| bound the branch horizon (24 steps) and bootstrap from the critic | a branch that replays to the end of the episode costs more than it is worth. The exact return-to-go of the action actually taken is free, so only the alternatives are rolled. |
| inject the exact advantage into the main trajectory (`--fork-inject 1`) | two optimiser passes fighting over one policy is worse than one coherent update. The counterfactual advantage overwrites the sampled one at forked states. |

Those changes were measured before the random-draw fix described above, so the
figures they produced (1.120 / 2.182 / 31.44 / 0.275) are not comparable with the
current table. The direction survived the fix; the magnitudes did not, which is
itself a reason to distrust a result that rests on one seed.

**Fork is faster in wall-clock at equal step budgets**: 7.4s vs 8.7s on the
dungeon, 11.4s vs 13.2s on the market, because forked rollouts need no
backpropagation passes.

**The most useful thing fork produces may be the explanation.** The same
mechanism that computes counterfactual advantages is what powers the Studio's
"rewind moments": a replay of the exact instant, once per action, with the
measured value of each.

### Where the method actually pays: it is the opposite of what I expected

The obvious prediction is that wider action spaces favour forking. A sampled
advantage gives the learner one action out of A; a counterfactual gives all A
exactly. So fork should get *better* as A grows.

It does not. Hold the market's economics completely fixed and change only the
resolution of the (width, skew) quote grid:

| actions | PPO | fork | gap |
|---:|---:|---:|---:|
| 4 | 2.078 | 2.051 | −1.3% |
| 8 | 2.077 | 2.115 | **+1.8%** |
| 16 | 2.213 | 2.125 | −4.0% |
| 32 | 2.195 | 1.915 | −12.8% |

(`python3 benchmarks/action_width.py`, 5 seeds, identical budgets.)

The reason is arithmetic, not statistics. Forking a state costs **O(A)**, one
branch per action, so the budget buys fewer and fewer forks as A grows. At 32
actions a forked state costs 768 environment steps, which is a third of an
entire training iteration. The information per fork goes up linearly; the
number of forks you can afford goes down linearly; they cancel, and then the
fixed cost of the branches wins.

Put together with the duel result, that gives a fairly sharp picture of the
niche: **fork pays when the action space is small enough to enumerate cheaply
and the return is noisy for a reason the critic cannot see.** The duel is both
(A=2 and A=3, with an opponent). The market and the cache are neither.

### More adversarial worlds, and a theory that did not survive

The duel was a single data point, so I built more worlds in its shape: a
first-price auction where two bidders place sealed bids from a six-level grid
and both decide on the same tick. Then the same auction with the lot settled
one tick later, and three ticks later. Same rules, same bids, same values; only
the settlement delay moves.

The hypothesis: fork's edge should grow with that delay, because a payoff
further from the decision needs an estimate of what the delayed payoff was
worth, and an exact replay supplies one.

| settlement delay | PPO | fork | gap |
|---|---:|---:|---:|
| same tick | 0.813 | 0.848 | +4.4% |
| one tick later | 0.743 | 0.848 | +14.2% |
| three ticks later | 0.877 | 0.765 | **−12.7%** |

Five seeds each, 600k steps, shared evaluation stream.

**It is not a trend.** The middle row looked like one and is not: PPO's mean
there is dragged down by a single seed that collapsed to 0.000, and with that
seed removed PPO leads. At three ticks the ordering reverses outright.

The one consistent thing is that fork is *steadier*: its ranges are
0.65–1.17, 0.23–1.69 and 0.72–0.80 against PPO's 0.65–1.13, 0.00–1.02 and
0.56–1.16. Steadier is not better, and a method that is reliably mediocre is
not a result.

So the displacement story is dead too. The honest state is that the duel shows
no established advantage, that no mechanism has survived a second world, and
that three explanations have failed: "an adversary is enough" (fails on the
auction) and "the optimum must be mixed" (fails on rock-paper-scissors). The
dose-response experiment was supposed to test a third idea, that the delay
between a decision and its payoff is what matters, but it could not: the
runtime attaches a reward to the most recent decision by that agent, and the
settlement happens at the top of a step in which the agent bids again, so the
payoff landed on the new bid rather than the one that caused it. Those results
are withdrawn rather than reported as a failed hypothesis. I am not going to keep
inventing explanations until one sticks.

### The mixing test, and the end of the explanations

The last hypothesis was that fork helps when the optimum must be *mixed* rather
than chosen: bluffing at the right frequency is a mixed optimum, bidding on an
auction is a pure one. That would have explained all six worlds at once.

`worlds/rps.hunch` tests it in isolation. Three actions, no hidden information,
no delay, and the only thing to learn is how often to play each one. The
prediction was committed to git before the run.

| method | mean spread | worst seed |
|---|---:|---:|
| PPO | 1.864 | 1.989 |
| fork | 1.820 | 1.985 |

Spread is the gap between the best and worst score against the three fixed
opponents. A uniform policy scores near zero against all three, so a spread near
2 means the policy collapsed onto one move.

Neither method learned to mix. Both settled on a single action and stayed
there, which is maximally punishable, and fork's 1.820 against PPO's 1.864 is
noise.

So the third explanation is dead too, but for a sharper reason than this
section first gave: a variance-reduced estimate does not change the objective,
so it cannot change where the objective settles. Fork and PPO share a fixed
point, and they landed on it together.

The experiment also corrected something this document previously got wrong.
It claimed the implementation could not learn the simplest mixed strategy. It
can. It was tuned to a temperature at which it cannot converge in the time
available: the entropy coefficient was 0.01 against a run of roughly 10,500
Adam steps, while reaching the entropy-regularised fixed point takes on the
order of 1/(lr x entropy), about 33,000 steps at those settings. Raising the
coefficient to 0.3, changing nothing else, moves the same five seeds from a
spread of 1.87 to 0.155. A method that responds monotonically to a temperature
knob and lands near the equilibrium was misconfigured, not broken.

Two defects made the original run worse and are worth recording. The rps world
gave the policy a constant observation while the value-to-go changed every
tick, which inflates the advantage variance by a large factor. And fork
branches rolled a fixed 24 ticks past the end of the episode, which is fixed.

The general lesson still stands, and it is not this implementation's fault:
gradient descent-ascent does not converge in bilinear games, and entropy
regularisation only moves the fixed point to a quantal response equilibrium.
A genuinely unexploitable policy needs regret matching over the exact values
fork already computes, not a better gradient.

The duel is therefore reported as an experiment that did not establish a
difference. Three explanations were tried and all three failed. Inventing a
fourth would be fitting the story to the data rather than the other way round.

**What did not work.** Three things I tried and rejected, in case they are
useful to someone else: (a) keeping branch decisions as extra training data
made the dungeon collapse on one seed: the harvested batch is too correlated
with itself; (b) training only the critic from counterfactual targets, without
the policy gradient, was clearly worse (0.81 vs 1.12), so the counterfactual
*policy* gradient is doing the real work; (c) raising the market's price noise
to make it more realistic did not favour fork: both methods degraded and PPO
stayed ahead, so the self-play result is about opponents, not volatility.

This is a working system with an honest, specific result: comparable to a
strong PPO baseline on single-agent tasks, clearly better when there is an
adversary, and worse as the action space widens. The four worlds were chosen to
be different from each other, not to make any one method look good, and the
action-width sweep, which I expected to be fork's best case, turned out to be
its worst.

## How the rewind works

Every world's state lives in one flat block of memory: all its variables, its
random number generator, its step counter. Snapshotting the universe is a
single `memcpy` of about 2 KB. Nothing else in the runtime has to cooperate.

That single fact is what makes the rest cheap:

1. **Record.** While an episode runs, save the state block before every decision.
2. **Select.** Rank the recorded decisions by policy entropy plus surprise
   (`|reward − value|`), and take the ones that were actually hard.
3. **Fork.** For each selected instant, restore the block once per action,
   force that action, and roll forward. The random stream is part of the
   snapshot, so every branch sees *the same coin flips*. The noise cancels and
   the difference between branches is purely causal.
4. **Update.** You now have `Q(s, a)` for every action at that state. The policy
   gradient is exact: `∂J/∂z_k = p_k · A_k`. The action that was actually taken
   needs no rollout at all: its real continuation is already in the trajectory.
5. **Bootstrap.** A branch rolls a bounded horizon, then hands the tail to the
   critic, so a fork costs tens of steps instead of a whole episode.

Every step spent inside a branch counts against the same budget as a real
environment step. The comparison above is therefore like for like.
