<p align="center">
  <img src="assets/cards/hunch-card-1200x630.png" alt="Hunch card, a 9 by 5 riso-printed grid of blue and purple squares on white, with the letter H formed in light yellow on the left">
</p>

<h1 align="center">
  Hunch
  <a href="https://github.com/knownquantity/Hunch/actions/workflows/tests.yml"><img src="https://github.com/knownquantity/Hunch/actions/workflows/tests.yml/badge.svg" alt="tests" height="22"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/knownquantity/Hunch" alt="license" height="22"></a>
</h1>

**Hunch** is a small language for reinforcement-learning environments and games. You write the rules of a world and what is rewarded, and leave a `hunch(...)` blank wherever a decision is made. `hunch train` fills the blanks, and then tells you what it found: what each agent collects, whether anything better exists, and what the policy does.

A world is one file. This is all of Kuhn poker, the smallest game in which bluffing is correct:

```text
world Kuhn
horizon 3

state {
  card_a: int = 0;
  card_b: int = 0;
  phase: int = 0;       // 0 alice opens, 1 bob faces a bet, 2 bob after a pass,
                        // 3 alice faces a bet
}

init {
  card_a = randint(0, 2);             // two different cards out of 0, 1, 2
  card_b = randint(0, 1);
  if card_b >= card_a { card_b = card_b + 1; }
}

step {
  let show = -1;                      // +1 if alice holds the higher card
  if card_a > card_b { show = 1; }
  let facing_bet = 0;
  if phase == 1 || phase == 3 { facing_bet = 1; }

  if phase == 0 {
    let a = hunch alice([card_a, facing_bet], [pass, bet]);
    if a == bet { phase = 1; } else { phase = 2; }
  } else if phase == 1 {
    let b = hunch bob([card_b, facing_bet], [pass, bet]);
    if b == pass { reward(alice, 1); reward(bob, -1); }
    else { reward(alice, 2 * show); reward(bob, -2 * show); }
    done();
  } else if phase == 2 {
    let b = hunch bob([card_b, facing_bet], [pass, bet]);
    if b == pass { reward(alice, show); reward(bob, -show); done(); }
    phase = 3;
  } else {
    let a = hunch alice([card_a, facing_bet], [pass, bet]);
    if a == pass { reward(alice, -1); reward(bob, 1); }
    else { reward(alice, 2 * show); reward(bob, -2 * show); }
    done();
  }
}
```

Each `hunch` names an agent, says what that agent is shown, and names its actions. Two names in one file are two agents that learn against each other.

```text
$ ./bin/hunch train worlds/kuhn.hunch
Kuhn: solved by regret matching over every outcome: 58 positions, 12 situations, 453 rounds (0.0s)
scored exactly: every chance outcome and every action weighed, 58 positions

  alice  collects -0.0556 per episode
  bob    collects +0.0556 per episode
  with everyone playing at random:  alice +0.1250, bob -0.1250
  against opponents playing at random:  alice +0.1205, bob +0.1666
  the most each agent could gain by changing its policy alone: alice 0.0006, bob 0.0002   (NashConv 0.0007)

what alice does  (over all its decisions: pass 71%, bet 29%)
  card_a  facing_bet   |  pass   bet   |  times per episode
       0           0   |   81%   19%   |  0.333
       1           0   |  >99%   <1%   |  0.333
       2           0   |   42%   58%   |  0.333
       1           1   |   47%   53%   |  0.222
       0           1   |  >99%   <1%   |  0.134
       2           1   |   <1%  >99%   |  0.023
...
saved worlds/kuhn.hpol   (hunch run, why and play use it for worlds/kuhn.hunch)
```

That is the textbook answer. The game is worth −1/18 to the first player. Alice bluffs with the lowest card 19% of the time, bets the highest three times as often (58%), and calls with the middle card a third of the time more than she bluffs (53%), which is one of the known family of equilibria. Nobody wrote "bluff" anywhere.

You need `python3` and a C compiler; the compiler is pure Python and the runtime is one C file. From a checkout, `./bin/hunch` works with nothing installed. To get the `hunch` command and `import hunch` anywhere:

```bash
pip install git+https://github.com/knownquantity/Hunch
hunch train kuhn          # the worlds in worlds/ ship with it, by name
```

The same file is also an environment. It steps from Python at about a million decisions a second, wraps as gymnasium or PettingZoo, and, because the game is solved, it can grade any agent you bring: a PPO network, a script, a language model. [From Python](#from-python) has that part.

---

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

---

## From Python

```python
import hunch

world  = hunch.World("worlds/kuhn.hunch")
solved = world.train()                      # the policy, with its report

env = world.env(agents=["alice"], policy=solved)    # you decide alice; bob plays the policy
env.reset(seed=1)
while not env.over:
    print(env.obs, env.legal)               # what alice is shown; the actions allowed
    env.act("bet")                          # by name or by number
print(env.returns)                          # {'alice': 1.0, 'bob': -1.0}
```

| call | what you get |
|---|---|
| `world.env(agents, policy)` | one environment. You decide `agents`; the rest play `policy`, or at random without one |
| `world.vec(n, agent, policy)` | `n` environments stepped together in C as numpy arrays, with a legal-action mask |
| `world.gym(agent, policy)` | a `gymnasium.Env` for one seat (`action_mask` in `info`; passes `check_env`) |
| `world.pettingzoo()` | a PettingZoo AEC environment over every seat (passes `api_test`) |
| `env.snapshot()`, `restore`, `reseed` | any instant, random numbers included: restore it and act differently for a fork, reseed for a different future |
| `world.grade(agents, against)` | what any agent collects, how much it gives away, and its costliest decisions |
| `hunch.Search(world, playouts)` | tree search at play time, for worlds too big to solve |

Speed on one core, 1,024 environments through `vec.step()`, not counting the time to choose the actions: Kuhn 1.5 million decisions a second, Leduc 1.1 million, tic-tac-toe 0.7 million, Connect Four 0.3 million.

### `grade`: an exact score for any agent

A learning curve against a fixed opponent cannot tell a strong agent from one that has learned that opponent's habits. When the world is small enough to follow whole, `grade` can. It asks your agent what it does in every situation its seat can face, then weighs every deal and every reply. Nothing is sampled.

[`examples/ppo_sb3.py`](examples/ppo_sb3.py) trains Stable-Baselines3 PPO as alice against the solved bob, through `world.gym`, and grades the network before and after:

```python
def alice(obs, legal):                      # any function: an action, or one probability per action
    tensor, _ = model.policy.obs_to_tensor(np.asarray(obs, dtype=np.float32))
    return model.policy.get_distribution(tensor).distribution.probs[0].tolist()

print(world.grade({"alice": alice}, against=solved))
```

```text
after training
Kuhn: graded alice (every outcome weighed, nothing sampled)
  alice collects -0.0574 per episode; the reference policy collects -0.0556 there
    the most alice could collect against the others is -0.0550: it falls 0.0024 short
    the best reply by bob collects +0.3252: alice gives away 0.2696
      (give or take 0.0007: how far the reference policy is from an equilibrium)
  decisions that cost most (per episode):
    0.0024  alice shown card_a = 1, facing_bet = 0: played pass 96%, bet 4%; pass was better
```

After 60,000 steps PPO collects within 0.002 of the most anyone could against this bob, and a bob who knew its policy would take 0.27 a hand from it. Both numbers are exact, and the reward curve shows only the first.

For a sharper reference than the default, train with `world.train(target=1e-6)`.

The same grade works for anything that can answer a question. A program is asked one JSON question per line and prints one action per line, flushed:

```bash
hunch grade worlds/kuhn.hunch --agent alice --cmd 'python3 examples/always_bet.py'
```

For a language model, the questions are prompts. Each holds the rules (the comment at the top of the world's file), what the agent is shown, and the actions it may take:

```bash
hunch grade worlds/kuhn.hunch --agent alice --questions > questions.jsonl
# ... ask your model each "prompt"; write one answer per line ...
hunch grade worlds/kuhn.hunch --agent alice --answers answers.jsonl
```

An answer is an action name, JSON probabilities such as `{"pass": 0.7, "bet": 0.3}`, or a plain sentence that names exactly one action. Anything it cannot read for certain (two actions named, "I will not bet", a blank line, a missing answer) stops the grade and names the question; nothing is guessed. [`examples/grade_llm.py`](examples/grade_llm.py) does the round trip against any OpenAI-compatible endpoint.

A worked case: [`worlds/dice3.hunch`](worlds/dice3.hunch) is liar's dice with one three-sided die each, 96 situations a seat, and a rules comment written to be read by a model. A Claude model was given the 96 prompts for each seat and told to answer as a player, by reasoning and without a solver; its answers are in [`examples/answers/`](examples/answers).

```text
$ hunch train worlds/dice3.hunch
$ hunch grade worlds/dice3.hunch --agent alice --answers examples/answers/dice3_alice.jsonl
Dice3: graded alice (every outcome weighed, nothing sampled)
  alice collects +0.1112 per episode; the reference policy collects +0.1108 there
    the most alice could collect against the others is +0.1112: it falls 0.0000 short
    the best reply by bob collects -0.1111: alice gives away 0.0000
      (give or take 0.0007: how far the reference policy is from an equilibrium)
  no decision costs as much as 0.0001 against these opponents
```

The model said its strategy wins five games in nine and cannot be pushed below that. The grade agrees: the best reply to it collects exactly −1/9. For contrast, an agent that never bluffs (bids only what its own die makes true, and otherwise calls) collects −0.2776 in the same seat and gives away 0.8886, and the report lists the opening bids that cost it most. That is one model on one small game, with no check on how it reasoned beyond its instructions; the point is that the grade needs no opponent pool, no Elo and no sampling.

A world too big to follow whole is graded by playing episodes instead, and the report then says "sampled" and gives a standard error.

### `Search`: worlds too big to solve

[`worlds/connect4.hunch`](worlds/connect4.hunch) is standard Connect Four in about 80 lines, with about 4.5 trillion positions. `train` cannot follow that, and sampling learns little in a minute. `hunch.Search` forks the world at the decision in front of it instead: Monte Carlo tree search in under a hundred lines of Python over `snapshot`, `restore`, `reseed` and `playout`, with no knowledge of the game.

```text
$ python3 examples/search.py
search with 300 playouts a move against random play: won 30, drew 0, lost 0   (5s)
search with 1500 playouts a move against search with 150: won 17, drew 1, lost 2   (42s)
```

On tic-tac-toe, where the answer is known, search with 2,500 playouts a move never lost to the solved policy in 40 games as the second player; with 600 it lost 6. The search reads the whole state, so it is fair only where nothing is hidden.

## What `train` does

The runtime holds the whole world, its random number generator included, in one flat block of memory. Any instant can be copied and replayed with a different action or a different roll of the dice. `train` uses that to follow every outcome when the world is small enough, and picks its method from what the world is:

| the world | method | what you are told |
|---|---|---|
| small, and no agent would choose differently if it saw the whole state (board games, Nim, most one-agent worlds) | backward induction over every position | the value, and a policy that attains it, exactly |
| small, one agent | search over every outcome | the best any policy can collect, when that can be proven; otherwise the best found and the ceiling for an agent that saw everything |
| small, several agents | regret matching (CFR+) over every outcome | each agent's value, and how much each could gain by changing alone (NashConv) |
| too big to follow, or uses `rand()`, one agent | PPO with a 32×32 network | the return with a standard error, against random play |
| too big to follow, several agents | regret matching on sampled episodes | the same, sampled |

"Small" is up to two million positions after merging the ones that share a state, and chance from `randint`, whose outcomes can be listed. `--algo` overrides the choice: `regret`, `ppo`, `fork`, `reinforce`, `random`.

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

One-agent worlds: `dungeon.hunch` (key, trap, locked exit, 50 steps) comes out at +1.134 in three seconds. That one is reported as found, not proven, because the hero is shown less than the whole state; an agent that saw everything could collect +1.135. PPO at 400,000 steps averaged 1.05 over five seeds on a sampled evaluation ([`benchmarks/README.md`](benchmarks/README.md)).

Worlds that draw real numbers with `rand()` are trained with PPO. `cache.hunch` is a four-slot cache on a workload with hot keys, scans, and a hot set that moves. The shipped policy, replayed on 96 evaluation episodes (`python3 benchmarks/cache_baselines.py`):

| policy | hits / 60 | hit rate |
|---|---:|---:|
| LRU | 27.86 | 46.4% |
| LFU | 28.10 | 46.8% |
| **hunch (trained)** | **31.77** | **53.0%** |
| Belady OPT (sees the future) | 36.09 | 60.2% |

## Writing a world

The complete reference is one page: [`llms.txt`](llms.txt). It is written to be pasted into a model's context whole, and a model that has read it can write a new world in about thirty lines.

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
- `Search` is plain UCT with random playouts, written for clarity in Python: about 19,000 playouts a second on tic-tac-toe. It is a strong baseline for a small board game, not a tuned engine.
- The Python package is tested on Linux with gcc. The build flags are the same ones clang takes, but macOS and Windows have not been run.
- The language has no modules, no strings, and no recursion, on purpose: a world that fits on a page is one a person or a model can get right.

## Studio

`./bin/hunch studio` serves a page that replays the dungeon, the market, the duel and the cache side by side, untrained against trained, with the decisions where forking mattered most, and retrains them live.

## Where this came from

Hunch began as an experiment in credit assignment: replay a decision once per action under the same random numbers, and train a network on the measured difference (`--algo fork`). On equal budgets that does not beat PPO, and [`benchmarks/README.md`](benchmarks/README.md) keeps the full record, including the hypotheses that failed. What survived is the mechanism. The same replay is what follows every outcome of a small world, what `why` uses, and what lets a person sit in with `play`.

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
├── runtime/hunch_vm.c            the whole engine: VM, exact graph, regret matching, MLP, PPO, fork; also built as a library
├── worlds/                       one .hunch file each
├── examples/                     grade PPO, a program or a language model; search Connect Four
├── policies/                     shipped trained policies for the Studio
├── benchmarks/                   the fork experiments and the cache baselines
├── tests/                        test_compiler.py, test_bindings.py, and fuzz_exact.py: exact answers against brute force
├── studio/                       the browser playground, no build step
└── llms.txt                      one-page reference for humans and models
```

```bash
python3 tests/test_compiler.py            # 48 tests, a few seconds
python3 tests/test_bindings.py            # the Python package; numpy, gymnasium and pettingzoo checks run when installed
python3 tests/fuzz_exact.py               # 200 random worlds: exact answers against brute force
```
