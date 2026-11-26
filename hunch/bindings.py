"""A Hunch world in the libraries people already train with.

    world.gym("alice", policy=solved)     # gymnasium.Env: you are alice, bob plays `solved`
    world.pettingzoo()                    # pettingzoo.AECEnv: every agent on its turn

Each library is imported inside the function that needs it, so `import hunch`
works with none of them installed.

A world shows an agent something only when it asks that agent to decide, so
nothing is observed after the last decision of an episode. Where a library
wants an observation there anyway, it is given the one the agent saw last.
"""

from __future__ import annotations

from .world import HunchRuntimeError, World

_U63 = (1 << 63) - 1


def _need(module: str):
    try:
        return __import__(module)
    except ImportError:
        raise ImportError(f"this needs {module}, which is not installed: "
                          f"pip install {module}") from None


def _begin(env, seed, rng) -> None:
    """Start the episode `seed` names, or one drawn from `rng` when it is None.
    An episode that ends before the outside agents decide anything is skipped."""
    for _ in range(10000):
        if seed is None:
            seed = int(rng.integers(_U63))
        if env.reset(seed):
            return
        seed = None
    raise HunchRuntimeError(f"{env.world.name}: no episode reaches a decision")


def _mask(np, legal, n: int):
    """Legal actions as an int8 array, the form gymnasium's Discrete.sample takes."""
    mask = np.zeros(n, np.int8)
    mask[legal] = 1
    return mask


def make_gym(world: World, agent=None, policy=None, render_mode=None):
    """A gymnasium.Env in which you decide for `agent` (default: the only
    agent, or the first) and every other agent plays `policy`, or uniformly at
    random without one.

    The observation is a float32 Box, the action Discrete, and
    info["action_mask"] says which actions are legal now. reset(seed=s) plays
    the episode `hunch run --seed s` plays. An illegal action raises
    HunchRuntimeError. The reward is what `agent` was paid since its last
    decision, including what other agents' moves paid it.
    """
    np, gymnasium = _need("numpy"), _need("gymnasium")

    if not world.agents:
        raise ValueError(f"{world.name} has no agent")
    name = agent or world.agents[0]
    if name not in world.agents:
        raise ValueError(f"no agent named '{name}'; this world has " + ", ".join(world.agents))
    if render_mode not in (None, "ansi"):
        raise ValueError("render_mode must be None or 'ansi'")
    n_obs, n_act = world.obs_dim[name], world.n_actions[name]

    class HunchEnv(gymnasium.Env):
        metadata = {"render_modes": ["ansi"]}

        def __init__(self):
            self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
            self.action_space = gymnasium.spaces.Discrete(n_act)
            self.render_mode = render_mode
            self.world = world
            self.agent = name
            self._env = world.env(agents=[name], policy=policy)
            self._obs = np.zeros(n_obs, np.float32)
            self._live = False
            self._owed = 0.0

        def _info(self) -> dict:
            return {"action_mask": _mask(np, self._env.legal, n_act)}

        def action_masks(self):
            """The legal actions as booleans (the name sb3-contrib's MaskablePPO looks for)."""
            return self._info()["action_mask"].astype(bool)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            _begin(self._env, seed, self.np_random)
            self._obs[:] = self._env.obs
            self._live = True
            # Anything paid before the first decision arrives with the first step.
            self._owed = self._env.last_paid[name]
            return self._obs.copy(), self._info()

        def step(self, action):
            if not self._live:
                raise gymnasium.error.ResetNeeded("the episode is over or was never started; "
                                                  "call reset()")
            env = self._env
            paid = env.act(int(action))
            if env.over:
                self._live = False
            else:
                self._obs[:] = env.obs
            reward, self._owed = float(paid[name] + self._owed), 0.0
            return self._obs.copy(), reward, env.terminated, env.truncated, self._info()

        def render(self):
            if self.render_mode != "ansi":
                return None
            env = self._env
            state = ", ".join(f"{k}={v}" for k, v in env.state.items())
            if env.over:
                return f"over; {state}"
            return f"{name} sees {env.describe()}; {state}"

    return HunchEnv()


def make_pettingzoo(world: World, agents=None, policy=None, render_mode=None):
    """A PettingZoo AECEnv over `agents` (default: all); any others play
    `policy`. Agents keep their Hunch names and move in the order the world
    asks them to, which need not alternate.

    infos[agent]["action_mask"] says what the agent to move may play; it is all
    zero for everyone else. observe(agent) is what that agent saw at its latest
    decision (zeros before its first). Every agent's pay is added to its
    running total on the step that pays it, so an agent paid on another's move
    finds it in last() when its turn comes or when the episode ends.
    """
    np = _need("numpy")
    spaces = _need("gymnasium").spaces
    _need("pettingzoo")
    from gymnasium.utils import seeding
    from pettingzoo import AECEnv

    names = list(agents or world.agents)
    if not names:
        raise ValueError(f"{world.name} has no agent")
    for a in names:
        if a not in world.agents:
            raise ValueError(f"no agent named '{a}'; this world has " + ", ".join(world.agents))
    if render_mode is not None:
        raise ValueError("render_mode must be None")

    class HunchAEC(AECEnv):
        metadata = {"name": f"hunch_{world.name.lower()}", "render_modes": [],
                    "is_parallelizable": False}

        def __init__(self):
            super().__init__()
            self.possible_agents = list(names)
            self.agents = []
            self.observation_spaces = {a: spaces.Box(-np.inf, np.inf, (world.obs_dim[a],),
                                                     np.float32) for a in names}
            self.action_spaces = {a: spaces.Discrete(world.n_actions[a]) for a in names}
            self.render_mode = None
            self.world = world
            self._env = world.env(agents=names, policy=policy)
            self._rng = None

        def observation_space(self, agent):
            return self.observation_spaces[agent]

        def action_space(self, agent):
            return self.action_spaces[agent]

        def observe(self, agent):
            return self._seen[agent].copy()

        def _show(self) -> None:
            """Select the agent the world is waiting on, or end everyone's episode."""
            env = self._env
            self.infos = {a: {"action_mask": np.zeros(world.n_actions[a], np.int8)}
                          for a in self.agents}
            if env.over:
                self.terminations = {a: env.terminated for a in self.agents}
                self.truncations = {a: env.truncated for a in self.agents}
                self.agent_selection = self.agents[0]
                return
            agent = env.agent
            self._seen[agent] = np.array(env.obs, np.float32)
            self.infos[agent]["action_mask"] = _mask(np, env.legal, world.n_actions[agent])
            self.agent_selection = agent

        def reset(self, seed=None, options=None):
            if seed is not None or self._rng is None:
                self._rng, _ = seeding.np_random(seed)
            _begin(self._env, seed, self._rng)
            self.agents = list(self.possible_agents)
            self.rewards = {a: 0.0 for a in self.agents}
            self._cumulative_rewards = {a: 0.0 for a in self.agents}
            # Anything paid before the first decision arrives with the first step.
            self._owed = dict(self._env.last_paid)
            self.terminations = {a: False for a in self.agents}
            self.truncations = {a: False for a in self.agents}
            self._seen = {a: np.zeros(world.obs_dim[a], np.float32) for a in self.agents}
            self._show()

        def step(self, action):
            agent = self.agent_selection
            if self.terminations[agent] or self.truncations[agent]:
                self._was_dead_step(action)
                return
            self._cumulative_rewards[agent] = 0.0
            paid = self._env.act(int(action))
            self.rewards = {a: paid[a] + self._owed.get(a, 0.0) for a in self.rewards}
            self._owed = {}
            self._accumulate_rewards()
            self._show()

    return HunchAEC()
