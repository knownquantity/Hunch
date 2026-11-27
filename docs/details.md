# Hunch in detail

The [README](../README.md) is the short version. This page has the full walkthrough of the commands, what `train` does for each kind of world, the results checked against known answers, and every limit.

## The loop

```bash
./bin/hunch test  worlds/leduc.hunch     # run the rules, check the world's own asserts
./bin/hunch train worlds/leduc.hunch     # fill the blanks, and say what was found
./bin/hunch run   worlds/leduc.hunch     # one episode, step by step
./bin/hunch why   worlds/leduc.hunch     # what every other action would have been worth
./bin/hunch play  worlds/leduc.hunch     # take a seat and play against it
```

`train` saves the policy beside the world, and the other commands pick it up from there.

### `run`: an episode you can read

```text
$ ./bin/hunch run worlds/leduc.hunch --seed 3
Leduc, seed 3, policy worlds/leduc.hpol
start    card=[4 2]  shown=6  paid=[1 1]  round=1  raises=0  turn=0  history=0
step 1   alice sees [4, 6, 0], plays call (25%)
         turn 0→1, history 0→1
step 2   bob sees [2, 6, 1], plays raise (85%)
         paid[1] 1→3, raises 0→1, turn 1→0, history 1→5
step 3   alice sees [4, 6, 5], plays raise (69%)
         paid[0] 1→5, raises 1→2, turn 0→1, history 5→17
...
step 6   bob sees [2, 0, 157], plays call (>99%)
         history 157→472
         reward: alice +5, bob -5
ended after 6 steps:  alice collected +5, bob collected -5
```

### `why`: was that a mistake?

Bob raised with a queen into a king and lost five chips. `why` goes back to each decision and tries every other action, and it answers two different questions:

```text
$ ./bin/hunch why worlds/leduc.hunch --seed 3 --step 2
step 2  bob sees card[1] = 2, shown = 6, history = 1
        and plays raise (85%)

  this episode, replayed from that instant 200 times per action:
        call     -0.7200   2.2700 better than raise ± 0.4200
      > raise    -2.9900   taken

  every time bob is shown this, whatever it cannot see (exact):
        call     +0.3936   as good as raise
      > raise    +0.3942   taken
```

With alice holding the king, calling was better by 2.3 chips. But bob cannot see her card, and over every hand in which he sees what he saw here, the raise is worth as much as the call. It was bad luck, not a bad policy. The first block is a replay of the same instant under the same random numbers, which works in any world. The second needs a world small enough to follow whole.

### `play`: sit in

```text
$ ./bin/hunch play worlds/kuhn.hunch --as bob --seed 12
Kuhn: you are bob. alice: policy worlds/kuhn.hpol. Seed 12.
You are shown what your agent is shown, and nothing else. q quits.

step 2: you see card_b = 1, facing_bet = 0
  your move  [0] pass  [1] bet  >
```

## What `train` does

The runtime holds the whole world, its random number generator included, in one flat block of memory. Any instant can be copied and replayed with a different action or a different roll of the dice. `train` uses that to follow every outcome when the world is small enough, and picks its method from what the world is:

| the world | method | what you are told |
|---|---|---|
| small, and no agent would choose differently if it saw the whole state (board games, Nim, most one-agent worlds) | backward induction over every position | the value, and a policy that attains it, exactly |
| small, one agent | search over every outcome | the best any policy can collect, when that can be proven; otherwise the best found and the ceiling for an agent that saw everything |
| small, several agents | regret matching (CFR+) over every outcome | each agent's value, and how much each could gain by changing alone (NashConv) |
| too big to follow, and sampling finds nothing hidden from the agents | a search at each decision (`mcts`), guided by a network it trains by playing itself | the return against random play, sampled, and what the search weighed at each decision |
| too big to follow or uses `rand()`, something hidden, one agent | PPO with a 32×32 network | the return with a standard error, against random play |
| too big to follow, something hidden, several agents | regret matching on sampled episodes | the same, sampled |

"Small" is up to two million positions after merging the ones that share a state, and chance from `randint`, whose outcomes can be listed. `--algo` overrides the choice: `regret`, `ppo`, `mcts`, `fork`, `reinforce`, `random`.

Every result is scored the same way whichever method produced it. `hunch score world.hunch --policy p.hpol` prints the report again without training.

## Checked against known answers

`hunch train`, default settings, one CPU core:

| world | situations | `train` finds | NashConv | known value | time |
|---|---:|---|---:|---|---:|
| `kuhn.hunch` | 12 | −0.0556 to alice | 0.0007 | −1/18 = −0.0556 | 0.1 s |
| `leduc.hunch` | 936 | −0.0856 to alice | 0.0008 | −0.0856 (OpenSpiel `leduc_poker`) | 0.3 s |
| `liars_dice.hunch` | 24,576 | −0.0273 to alice | 0.0008 | −7/258 = −0.0271 | 10 s |
| `tictactoe.hunch` | 4,520 | 0, a draw | 0 | 0 | 0.2 s |
| `duel.hunch` | 75 | +0.0959 to alice | 0.0010 | none published | 0.1 s |

NashConv is the sum over agents of what each could gain by switching to its best reply; it is zero at an equilibrium. `--target` sets where the solver stops (default 0.001).

One-agent worlds: `dungeon.hunch` (key, trap, locked exit, 50 steps) comes out at +1.134 in three seconds. That one is reported as found, not proven, because the hero is shown less than the whole state; an agent that saw everything could collect +1.135. PPO at 400,000 steps averaged 1.05 over five seeds on a sampled evaluation ([`benchmarks/README.md`](../benchmarks/README.md)).

Worlds that draw real numbers with `rand()` are trained with PPO. `cache.hunch` is a four-slot cache on a workload with hot keys, scans, and a hot set that moves. The shipped policy, replayed on 96 evaluation episodes (`python3 benchmarks/cache_baselines.py`):

| policy | hits / 60 | hit rate |
|---|---:|---:|
| LRU | 27.86 | 46.4% |
| LFU | 28.10 | 46.8% |
| **hunch (trained)** | **31.77** | **53.0%** |
| Belady OPT (sees the future) | 36.09 | 60.2% |

## Writing a world

The complete reference is one page: [`llms.txt`](../llms.txt). It is written to be pasted into a model's context whole, and a model that has read it can write a new world in about thirty lines.

The parts that matter most:

- **`hunch name(obs, actions)`** is a decision. `obs` is everything that agent is shown, and nothing else reaches its policy. `actions` is a count, or a list of names that then work as constants: `if a == bet`.
- **`hunch name(obs, actions, legal)`** takes a third array marking which actions are allowed here. Illegal ones are never played, by any method.
- **`reward(x)`** or **`reward(name, x)`** pays an agent. **`done()`** ends the episode.
- **`assert cond;`** makes a world check its own rules. `hunch test` runs episodes with random play and reports which asserts ran, and any index that went out of range or division by zero.
- **`randint(lo, hi)`** is chance the runtime can list, so worlds that use it can be solved exactly. `rand()` is a real number, and makes the world sample-only.
- **Only `state` lasts between steps.** Every `let` starts each step at zero.

Use whole numbers in observations when the thing observed is a whole number. A table policy keys on exactly what the agent is shown, and the report prints it under the names in your source.

## Limits

- Exact answers stop at about two million positions. Past that, or with `rand()`, results are sampled and say so, with standard errors.
- Regret matching is known to approach an equilibrium only with two agents whose rewards sum to zero. In other worlds it still runs and NashConv is still measured, so the report says how close it came, but it may stop at `--max-sec` short of the target.
- Exact answers rest on one rule: two instants with the same `state` at the same step have the same future. The compiler and runtime enforce it (locals reset every step), and `tests/fuzz_exact.py` checks the printed claims against brute force on random worlds, but those worlds are tiny and this is one C file; treat a surprising number as a bug report.
- A best reply is a proof only when the agent can tell apart everything it has seen and done, or when each thing it is shown comes up in one position only. Otherwise the report says the search is not a proof.
- Sampled regret matching keeps one table row per thing an agent can be shown, so it needs observations that repeat. It is also much slower than the exact solver where both apply: on liar's dice, 20 million sampled steps (20 s) reach a NashConv of 0.05, and the exact solver reaches 0.0008 in 10 s. With real-valued observations and several agents, use `--algo ppo`, which makes no equilibrium claim.
- The network is a 32×32 MLP on one CPU core. This is for small worlds.
- `grade` is exact for the agent and the opponents you give it. What an agent "gives away" is read off a reference policy, so it is as sharp as that policy is close to an equilibrium (0.001 by default), and the report says how close. Where an episode can show an agent the same thing twice, a decision's cost is for changing one visit only.
- An exact `grade` asks your agent about every situation once, so it needs an agent whose answer depends only on what it is shown. A language model sampled at a temperature is graded on the answers it happened to give; ask it for probabilities, or ask several times and average, if you want its mixed strategy. Leduc has 468 situations a seat and liar's dice 12,288, which is that many model calls.
- A search policy is Monte Carlo tree search guided by a network, as in AlphaZero: PUCT over the network's policy, and where a playout reaches a position the search has not met, a value that is half the network's estimate and half one random playout from there (the mix AlphaGo used; on Connect Four the network's value alone did worse, and on the 5×4 board random playouts alone did worse). Self-play leaves the random playout out, which plays more games and learned no worse. `train` makes the network by self-play for `--train-sec` seconds (default 600) on every core (`--cores N`), in rounds of 64 games at 200 playouts a move; it learns where the search spent its playouts and how each game ended, from the most recent 100,000 decisions. The network is two layers of 64 units; each part of an observation that takes at most 8 values in random play (a board cell) is one input per value. The time is fixed, not the games, so a machine with fewer cores ends with a weaker policy; `--games N` makes the same policy on any number of cores. `--games 0` is the search without a network, UCT with random playouts: about 190,000 playouts a second on tic-tac-toe and 33,000 on Connect Four, on one core, and it loses to perfect play where the right move pays off many moves later. Measured results are in [Worlds too big to solve](../README.md#worlds-too-big-to-solve); only two-player board games have been measured. A search cannot be scored exactly, so its report is a few sampled episodes. The check that nothing is hidden is by sampling, which is evidence and not proof.
- `grade` against a search policy on a big world plays whole episodes with a search at every move; expect it to take an hour, not seconds.
- The Python package is tested on Linux with gcc. The build flags are the same ones clang takes, but macOS and Windows have not been run.
- The language has no modules, no strings, and no recursion, on purpose: a world that fits on a page is one a person or a model can get right.

## Studio

`./bin/hunch studio` serves a page that replays the dungeon, the market, the duel and the cache side by side, untrained against trained, with the decisions where forking mattered most, and retrains them live.

## Where this came from

Hunch began as an experiment in credit assignment: replay a decision once per action under the same random numbers, and train a network on the measured difference (`--algo fork`). On equal budgets that does not beat PPO, and [`benchmarks/README.md`](../benchmarks/README.md) keeps the full record, including the hypotheses that failed. What survived is the mechanism. The same replay is what follows every outcome of a small world, what `why` uses, and what lets a person sit in with `play`.

## Repository layout

```text
├── bin/hunch                     CLI entry point
├── hunch/                        the language, in Python
│   ├── lang.py                   tokenizer and parser, with line/col errors
│   ├── codegen.py                syntax tree to bytecode
│   ├── cli.py                    the commands
│   ├── show.py                   what the commands print
│   ├── world.py                  `import hunch`: World, Env, VecEnv, Policy, grade
│   ├── bindings.py               gymnasium and PettingZoo
│   ├── search.py                 tree search over forks
│   └── studio_server.py          HTTP and JSON API behind the Studio
├── runtime/hunch_vm.c            the whole engine: VM, exact graph, regret matching, search and its self-play, MLP, PPO, fork; also built as a library
├── worlds/                       one .hunch file each
├── examples/                     grade PPO, a program or a language model; search Connect Four
├── policies/                     shipped trained policies for the Studio
├── benchmarks/                   the fork experiments and the cache baselines
├── tests/                        test_compiler.py, test_bindings.py, and fuzz_exact.py: exact answers against brute force
├── studio/                       the browser playground, no build step
├── docs/                         this page, and Hunch from Python
└── llms.txt                      one-page reference for humans and models
```

```bash
python3 tests/test_compiler.py            # 52 tests, about twenty seconds
python3 tests/test_bindings.py            # the Python package; numpy, gymnasium and pettingzoo checks run when installed
python3 tests/fuzz_exact.py               # 200 random worlds: exact answers against brute force
```
