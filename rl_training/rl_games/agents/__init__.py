"""Custom rl_games training algorithms.

Unlike network builders (which live in a process-global registry), rl_games resolves
algorithms per Runner instance, so registration takes the runner as an argument:

    runner = Runner(algo_observer=IsaacAlgoObserver())
    register_algos(runner)
"""

from rl_games.algos_torch import players

from .a2c_teacher_agent import A2CTeacherAgent
from .a2c_caps_agent import A2CCapsAgent

__all__ = ["A2CTeacherAgent", "A2CCapsAgent", "register_algos"]


def register_algos(runner) -> None:
    """Register this repo's algorithms on `runner`, selectable via the YAML's `algo.name`.

    a2c_teacher — standard PPO plus the annealed ANN->SNN distillation tail. Distillation
    only affects training, so it plays back with the stock continuous PPO player.

    a2c_caps — standard PPO plus a CAPS-style spatial-smoothness penalty on the actor
    (agents/a2c_caps_agent.py). Training-only; plays back with the stock continuous PPO
    player.
    """
    runner.algo_factory.register_builder(
        "a2c_teacher", lambda **kwargs: A2CTeacherAgent(**kwargs)
    )
    runner.player_factory.register_builder(
        "a2c_teacher", lambda **kwargs: players.PpoPlayerContinuous(**kwargs)
    )
    runner.algo_factory.register_builder(
        "a2c_caps", lambda **kwargs: A2CCapsAgent(**kwargs)
    )
    runner.player_factory.register_builder(
        "a2c_caps", lambda **kwargs: players.PpoPlayerContinuous(**kwargs)
    )
