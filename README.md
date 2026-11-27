<p align="center">
  <img src="assets/cards/hunch-card-1200x630.png" alt="Hunch card, a 9 by 5 riso-printed grid of blue and purple squares on white, with the letter H formed in light yellow on the left">
</p>

<h1 align="center">
  Hunch
  <a href="https://github.com/knownquantity/Hunch/actions/workflows/tests.yml"><img src="https://github.com/knownquantity/Hunch/actions/workflows/tests.yml/badge.svg?branch=main" alt="tests" height="22"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/knownquantity/Hunch" alt="license" height="22"></a>
</h1>

**Hunch** is a self-play-native programming language for RL environments. You write the rules of a world and what is rewarded, and leave a `hunch(...)` blank wherever a decision is made. `hunch train` fills the blanks, and then tells you what it found: what each agent collects, whether anything better exists, and what the policy does.

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

## Install

You need `python3` and a C compiler. From a checkout, `./bin/hunch` works with nothing installed. To get the `hunch` command and `import hunch` anywhere:

```bash
pip install hunch-lang
hunch train kuhn          # the worlds in worlds/ ship with it, by name
```

## Five commands

```bash
./bin/hunch test  worlds/leduc.hunch     # compile it, show what each agent sees, check the world's own asserts
./bin/hunch train worlds/leduc.hunch     # fill the blanks, and say what was found
./bin/hunch run   worlds/leduc.hunch     # one episode, step by step
./bin/hunch why   worlds/leduc.hunch     # what every other action would have been worth
./bin/hunch play  worlds/leduc.hunch     # take a seat and play against it
```

`train` saves the policy beside the world, and the others pick it up from there.

`why` is the one to try. In one hand of Leduc poker, bob raised with a queen into a king and lost five chips. Was that a mistake?

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

With alice holding the king, calling was better by 2.3 chips. But bob cannot see her card, and over every hand in which he sees what he saw here, the raise is worth as much as the call. It was bad luck, not a bad policy.

Hunch can answer that because the runtime keeps the whole world, its random numbers included, in one block of memory. Any instant can be copied and played again with a different action. That copy is a fork, and everything here is built on it.

## Worlds too big to solve

[`worlds/connect4.hunch`](worlds/connect4.hunch) is standard Connect Four in about 80 lines, with about 4.5 trillion positions. Nothing can follow all of that, so `train` gives you a policy that thinks at each move, and teaches it by having it play itself:

```text
$ ./bin/hunch train worlds/connect4.hunch
     1,920 games of self-play  (30 s)
     3,648 games of self-play  (1:01)
     ...
ConnectFour: too big to follow whole (more than 2,000,000 positions); in 40,000 sampled positions
nothing an agent was shown hid anything from it [...], so the policy is a search: it learned by
playing itself, and a network trained on those games guides it (612.6s)
  self-play: 26,112 games, 706,046 decisions, in 10:00 on 4 cores (--train-sec S sets how long)
  at each decision it forks the world and tries every action, 2,700 playouts a decision [...];
  where a playout reaches a position it has not met, the value is half the network's estimate
  and half one random playout from it
...
$ ./bin/hunch play worlds/connect4.hunch          # sit down against it
```

At each decision the policy forks the world and tries every action thousands of times (Monte Carlo tree search), then plays the action tried most. A small network steers it, in the manner of AlphaZero: it says which moves are worth trying first and who is winning in a position, and that estimate is averaged with one random playout where the search leaves the positions it has met. `train` makes the network by self-play: the search plays games against itself, the network learns from them where the search spent its effort and who went on to win, and the next games are played with the better network. It knows nothing about Connect Four beyond the 80 lines. Self-play runs for ten minutes by default, on every core (`--train-sec` and `--cores` change that).

Measured after the default ten minutes of self-play on 4 cores, playing on one core at the tenth of a second a move that `train` picks. The opponent on Connect Four is the same search without a network (random playouts), also at a tenth of a second:

| opponent | as first player | as second player |
|---|---|---|
| Connect Four, the search without a network, from the usual empty board | won 46, drew 1, lost 3 of 50 | won 48, lost 2 of 50 |
| the same, from 50 random openings of four discs | won 27, drew 2, lost 21 of 50 | won 37, drew 2, lost 11 of 50 |
| four-in-a-row on a 5×4 board ([`worlds/four.hunch`](worlds/four.hunch)), perfect play (the best result is a draw) | drew 100 of 100 | drew 100 of 100 |
| four-in-a-row on a 5×5 board, perfect play (a draw) | drew 100 of 100 | drew 96, lost 4 of 100 |

The perfect player picks at random among the moves that keep the game's exact value, so the games differ. The 5×4 row is one command to check: [`python3 examples/perfect_play.py`](examples/perfect_play.py) has Hunch solve that board outright, train on it the ordinary way, and play the two against each other (about a quarter of an hour, 3.6 GB). The 5×5 board is past what Hunch solves; its perfect player was a separate solver that is not in this repository. Without the network, at the same time a move, the search drew 25 of 100 as the first player on the 5×4 board and lost the rest, to a trap that closes fifteen moves after its mistake; on the 5×5 board it drew 74 and 67. `--games 0` gives you that search, with no training.

Strength grew with self-play and then levelled off. From the random openings on Connect Four, against the search without a network, in runs at 3,000 playouts a move (about 0.105 s): 64 games of self-play won 38 of 100, one minute on 4 cores 58, ten minutes on one core 64, ten minutes on 4 cores 69, and thirty minutes on 4 cores 65. A second training run with another seed won 75 of 100 from the empty board and 63 from the openings. Nothing here is proven beyond the games played, and Connect Four has not been measured against perfect play.

## What `train` does

| the world | method | what you are told |
|---|---|---|
| small enough to follow whole: up to two million positions | solved. Where no agent is kept from seeing something, by backward induction or a search over every outcome, which give the exact answer. Where agents hide things from each other, by regret matching (CFR+), which closes in on an equilibrium and stops within 0.001 of one | every number is computed over all outcomes, not sampled: what each agent collects, and how much any agent could still gain by changing its policy alone |
| too big, and nothing is hidden from the agents | a search at each decision, guided by a network it trains by playing itself | sampled returns, and what the search weighed at each decision |
| too big or real-valued chance, with something hidden | PPO for one agent, sampled regret matching for several | sampled returns with standard errors |

`hunch train`, default settings, one core, against answers that are known:

| world | `train` finds | known value | time |
|---|---|---|---:|
| `kuhn.hunch` | −0.0556 to alice | −1/18 = −0.0556 | 0.1 s |
| `leduc.hunch` | −0.0856 to alice | −0.0856 (OpenSpiel `leduc_poker`) | 0.3 s |
| `liars_dice.hunch` | −0.0273 to alice | −7/258 = −0.0271 | 10 s |
| `tictactoe.hunch` | 0, a draw | 0 | 0.2 s |

## From Python

The same file is an environment. It steps at about a million decisions a second, and wraps as gymnasium or PettingZoo.

```python
import hunch

world  = hunch.World("worlds/kuhn.hunch")
solved = world.train()                              # the policy, with its report

env = world.env(agents=["alice"], policy=solved)    # you decide alice; bob plays the policy
env.reset(seed=1)
while not env.over:
    env.act("bet")
print(env.returns)                                  # {'alice': 1.0, 'bob': -1.0}

def always_bet(obs, legal):                         # any function, network or model
    return "bet"

print(world.grade({"alice": always_bet}, against=solved))
```

`grade` is for evaluation. Because the game is solved, it can score any agent you bring, a PPO network, a script or a language model, by weighing every deal and every reply with nothing sampled. It reports what the agent collects, how much a best reply would take from it, and which decisions cost it most. In [`examples/ppo_sb3.py`](examples/ppo_sb3.py), PPO after 60,000 steps collects within 0.002 of the best possible against the solved opponent, and an opponent who knew its policy would take 0.27 a hand from it. The reward curve shows only the first number. Neither idea is new: [Timbers and others](https://arxiv.org/abs/2004.09677) showed that trained agents can be this exploitable, and [Regazzoni and others](https://neurips.cc/virtual/2026/loc/sydney/178174) scored language models this way on Kuhn poker. `grade` makes it one call on any small world you write. A language model is graded on the one answer it gives in each situation; to grade the mix it really plays, have it answer with probabilities, or ask several times and average.

[Hunch from Python](docs/python.md) has the full interface and the grading of a language model.

## Writing a world

The whole language is one page: [`llms.txt`](llms.txt). It is written to be pasted into a model's context. In one test, a model given only that page wrote Nim and a hidden-mine game that trained correctly, the first on its first try and the second on its second; that is one test, not a rate. The language has no modules, no strings and no recursion, on purpose: a world that fits on a page is one a person or a model can get right.

## Next to other tools

No algorithm here is new, and [OpenSpiel](https://github.com/google-deepmind/open_spiel) has all of them and more. There, a game is a class written against an API (Kuhn poker is 227 lines of Python) and you choose and script the algorithm. Here, Kuhn poker is the 78 lines above and `train` chooses. Use OpenSpiel for its hundred games, exact equilibria, more than two players and GPU training; use Hunch when the game is your own, small, and you want the answer today. [Hunch next to other tools](docs/compare.md) covers Gambit, Rulebook, Ludii, Pgx and the published work this builds on.

## Limits

- This is a tool for small games and for learning what a policy does. Two million positions holds Leduc poker and one-die liar's dice, not the games poker or chess research is about.

- Exact answers stop at about two million positions. Past that, or with `rand()`, results are sampled and say so.
- The search reads the whole state, so `train` picks it only when sampling finds nothing hidden, which is evidence and not proof. Its self-play takes ten minutes by default and gets what it can in that time: a machine with fewer cores plays fewer games and ends weaker. It has been measured on two-player board games only.
- Big worlds with hidden information get PPO with a small network or sampled regret matching, on one core. Expect much less from them than from the exact solver: on liar's dice, 20 seconds of sampling gets within 0.05 of an equilibrium and the exact solver within 0.0008.
- Regret matching is known to approach an equilibrium only with two agents whose rewards sum to zero.
- Tested on Linux with gcc. macOS and Windows have not been run.

[Hunch in detail](docs/details.md) has every limit, the full walkthrough of the commands, and where the project came from.

## Tests

```bash
python3 tests/test_compiler.py            # the language, the commands, the solvers
python3 tests/test_bindings.py            # the Python package
python3 tests/fuzz_exact.py               # 200 random worlds: exact answers against brute force
```
