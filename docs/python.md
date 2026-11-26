# Hunch from Python

A world file is also an environment. This page covers `import hunch`: stepping a world, the gymnasium and PettingZoo wrappers, forks, and grading any agent you bring. The language itself is in [`llms.txt`](../llms.txt); the commands are in the [README](../README.md).

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
| `hunch.Search(world, playouts)` | tree search from your own Python, over `snapshot` and `restore` |

Speed on one core, 1,024 environments through `vec.step()`, not counting the time to choose the actions: Kuhn 1.5 million decisions a second, Leduc 1.1 million, tic-tac-toe 0.7 million, Connect Four 0.3 million.

### `grade`: an exact score for any agent

A learning curve against a fixed opponent cannot tell a strong agent from one that has learned that opponent's habits. When the world is small enough to follow whole, `grade` can. It asks your agent what it does in every situation its seat can face, then weighs every deal and every reply. Nothing is sampled.

[`examples/ppo_sb3.py`](../examples/ppo_sb3.py) trains Stable-Baselines3 PPO as alice against the solved bob, through `world.gym`, and grades the network before and after:

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

An answer is an action name, JSON probabilities such as `{"pass": 0.7, "bet": 0.3}`, or a plain sentence that names exactly one action. Anything it cannot read for certain (two actions named, "I will not bet", a blank line, a missing answer) stops the grade and names the question; nothing is guessed. [`examples/grade_llm.py`](../examples/grade_llm.py) does the round trip against any OpenAI-compatible endpoint.

A worked case: [`worlds/dice3.hunch`](../worlds/dice3.hunch) is liar's dice with one three-sided die each, 96 situations a seat, and a rules comment written to be read by a model. A Claude model was given the 96 prompts for each seat and told to answer as a player, by reasoning and without a solver; its answers are in [`examples/answers/`](../examples/answers).

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
