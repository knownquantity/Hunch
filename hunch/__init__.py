"""Hunch: write the rules of a world, leave the decisions blank."""

__version__ = "0.4.0"

from .world import Env, Grade, HunchRuntimeError, Policy, Trained, VecEnv, World  # noqa: E402,F401
from .search import Search  # noqa: E402,F401


def examples() -> list:
    """The names of the worlds that ship with Hunch."""
    from . import cli
    return sorted(p.stem for p in cli.WORLDS.glob("*.hunch"))


def example(name: str) -> World:
    """One of the worlds that ship with Hunch, by name: example("kuhn")."""
    from . import cli
    path = cli.WORLDS / f"{name}.hunch"
    if not path.exists():
        raise ValueError(f"no example world named '{name}'; there are: " + ", ".join(examples()))
    return World(path)
