<p align="center">
  <img src="assets/cards/hunch-card-1200x630.png" alt="Hunch card, a 9 by 5 riso-printed grid of blue and purple squares on white, with the letter H formed in light yellow on the left">
</p>

<h1 align="center">
  Hunch
  <a href="https://github.com/knownquantity/Hunch/actions/workflows/tests.yml"><img src="https://github.com/knownquantity/Hunch/actions/workflows/tests.yml/badge.svg?branch=main" alt="tests" height="22"></a>
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

## Install

You need `python3` and a C compiler. From a checkout, `./bin/hunch` works with nothing installed. To get the `hunch` command and `import hunch` anywhere:

```bash
pip install git+https://github.com/knownquantity/Hunch
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

[`worlds/connect4.hunch`](worlds/connect4.hunch) is standard Connect Four in about 80 lines, with about 4.5 trillion positions. Nothing can follow all of that, so `train` gives you a policy that thinks at each move instead of one worked out ahead of time:

```text
$ ./bin/hunch train worlds/connect4.hunch
ConnectFour: too big to follow whole (more than 2,000,000 positions); in 40,000 sampled positions
nothing an agent was shown hid anything from it, so the policy is a search: at each decision it
forks the world and tries every action, 3,300 playouts a decision
...
$ ./bin/hunch play worlds/connect4.hunch          # sit down against it
```

At each decision the policy forks the world, plays every action out thousands of times (Monte Carlo tree search with random playouts), and plays the action that was tried most. It knows nothing about Connect Four beyond the 80 lines. Measured on one core at the default tenth of a second a move:

| opponent | as first player | as second player |
|---|---|---|
| Connect Four, random play | won 30 of 30 | won 30 of 30 |
| tic-tac-toe, the solved policy (the best result is a draw) | drew 60 of 60 | drew 60 of 60 |
| four-in-a-row on a 5×4 board, the solved policy (the best result is a draw) | drew 31 of 100, lost 69 | drew 100 of 100 |

The last row is the limit. Against perfect play the search loses most games as the first player, to a trap that closes fifteen moves after its mistake. `--think 1` gives it a second a move, and it then drew 22 of 30. It is a strong player to practise or evaluate against, not a solver.

## What `train` does

| the world | method | what you are told |
|---|---|---|
| small enough to follow whole: up to two million positions | solved, by backward induction, a search over every outcome, or regret matching (CFR+), depending on what the agents are shown | exact values, and how much any agent could gain by changing its policy alone |
| too big, and nothing is hidden from the agents | a search at each decision | sampled returns, and what the search weighed at each decision |
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

`grade` is for evaluation. Because the game is solved, it can score any agent you bring, a PPO network, a script or a language model, by weighing every deal and every reply with nothing sampled. It reports what the agent collects, how much a best reply would take from it, and which decisions cost it most. In [`examples/ppo_sb3.py`](examples/ppo_sb3.py), PPO after 60,000 steps collects within 0.002 of the best possible against the solved opponent, and an opponent who knew its policy would take 0.27 a hand from it. The reward curve shows only the first number.

[Hunch from Python](docs/python.md) has the full interface and the grading of a language model.

## Writing a world

The whole language is one page: [`llms.txt`](llms.txt). It is written to be pasted into a model's context, and a model that has read it can write a new world in about thirty lines. The language has no modules, no strings and no recursion, on purpose: a world that fits on a page is one a person or a model can get right.

## Limits

- Exact answers stop at about two million positions. Past that, or with `rand()`, results are sampled and say so.
- The search beats random play and loses to perfect play where the right move pays off many moves later. It reads the whole state, so `train` picks it only when sampling finds nothing hidden, which is evidence and not proof.
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
