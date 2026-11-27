# Hunch next to other tools

Nothing Hunch computes is a new algorithm. Regret matching, best replies, tree search guided by a network and PPO are all published, and [OpenSpiel](https://github.com/google-deepmind/open_spiel) has every one of them and more. What Hunch adds is the front: the rules are one short file, one command picks the method, and the report says how far the result is from the best possible.

This page was written on 2026-11-27 from each project's own repository and documentation. Line counts are whole files, comments included.

## If you already use OpenSpiel

OpenSpiel is the tool to compare with, and for most research it is the right one. It has about a hundred games, CFR and its variants, an exact equilibrium by linear programming, best response, exploitability, AlphaZero with a GPU, games with simultaneous moves, more than two players and rewards that do not sum to zero, and years of use in published work. Hunch checks its own Leduc value against it.

A game in OpenSpiel is a class written against its API: Kuhn poker is 227 lines of Python (`open_spiel/python/games/kuhn_poker.py`) and tic-tac-toe 304. You then choose an algorithm and write the script that runs it and measures the result. In Hunch, Kuhn poker is the 78-line file `worlds/kuhn.hunch` and tic-tac-toe 59 lines, and `hunch train` does the rest.

Reach for Hunch when the game is your own, small, and you want an answer today. Reach for OpenSpiel when you need an algorithm Hunch does not have, a game it already ships, or results other papers can be compared with.

## What each tool gives you

| | the game is written as | hidden information and chance | an equilibrium, and how far a policy is from one | trains a player | Python environment |
|---|---|---|---|---|---|
| **Hunch** | a small language, one file | yes | yes, for small games (CFR+, and exact best replies) | search with a small network on CPU; PPO | gymnasium, PettingZoo |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | a Python or C++ class | yes | yes, by many methods, some exact | AlphaZero, PPO, Deep CFR and more | its own, and wrappers |
| [Gambit](https://github.com/gambitproject/gambit) | the game tree, node by node | yes | yes, exact, including several players and rewards that do not sum to zero | no | no |
| [Rulebook](https://github.com/rl-language/rlc) | a compiled language with types and modules | yes | no | yes, one command (needs PyTorch) | yes |
| [Ludii](https://github.com/Ludeme/Ludii) | a language of board-game concepts | yes | no | search players | a separate project |
| [Ludax](https://github.com/gdrtodd/ludax) | a board-game language compiled for GPUs | no | no | AlphaZero on GPU | JAX |
| [Pgx](https://github.com/sotetsuk/pgx) | JAX code | in some of its games | no | AlphaZero and PPO examples on GPU | PettingZoo |
| [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) | a Python class (tic-tac-toe is 328 lines) | per environment | no | no, it is an interface | it is the interface |

## What the others do better

- **Bigger and stronger.** Pgx, Ludax and LightZero train on GPUs and reach games like Go and chess. Hunch's network is two small layers on a CPU, measured only on Connect Four and smaller boards.
- **Shorter board games.** Ludii has "three in a line" built in, so its tic-tac-toe rules are about a dozen lines against Hunch's 59, and it ships more than two thousand games.
- **Truly exact equilibria.** Gambit and OpenSpiel's linear program give the equilibrium itself. Hunch's regret matching stops when no agent could gain more than 0.001 by changing its policy, and prints the gap it reached.
- **More kinds of game.** Hunch's equilibrium claim holds for two agents whose rewards sum to zero. Gambit handles the rest.
- **A fuller language.** Rulebook has types, modules, editor support and bindings for four host languages.

## What is published already

- **Rules in a small language, out comes the solution of a hidden-information game.** Gala did this in 1995 (Koller and Pfeffer, [IJCAI 1995](https://ijcai.org/proceedings/1995/0153.pdf)).
- **Scoring a language model by what a best reply would take from it.** Regazzoni, Caillaud and Leitão did this for eight commercial models on Kuhn poker ("Strength Is Not Soundness", [NeurIPS 2026 workshop](https://neurips.cc/virtual/2026/loc/sydney/178174)). `hunch grade` does the same for any small world you write. The larger game benchmarks for language models (GTBench, TextArena, Kaggle Game Arena, GameBench, gg-bench) rank models by wins or ratings against other players, and their repositories contain no exploitability measure.
- **A reward curve that looks finished can hide an exploitable policy.** Timbers and others showed it for trained agents on Leduc and liar's dice ([arXiv 2004.09677](https://arxiv.org/abs/2004.09677)). [`examples/ppo_sb3.py`](../examples/ppo_sb3.py) is a demonstration of that result, not a finding.
- **A model writing a game from a description.** It has been measured in Python against OpenSpiel's API ([Code World Models](https://arxiv.org/abs/2510.04542)) and in Ludii's and Ludax's languages. Hunch's evidence is one test: a model given only [`llms.txt`](../llms.txt) wrote two working worlds.

## What we did not find in one tool

- A short rules file from which the tool itself chooses between solving and learning, and reports the distance from an equilibrium, with no algorithm named by the user. The parts exist separately; this joins them.
- `why` as a command: one decision of one played episode replayed once per action under the same random numbers, beside the exact value of each action over everything the agent cannot see. OpenSpiel has the second half as a function. Replaying under the same random numbers is a standard technique; the claim is the tool.
- Training by self-play with nothing to install beyond `python3` and a C compiler.

The search behind this list was not exhaustive. If you know a tool that does one of these, please open an issue.
