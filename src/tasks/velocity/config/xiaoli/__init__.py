from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    xiaoli_flat_env_cfg,
    xiaoli_rough_env_cfg,
)
from .rl_cfg import xiaoli_ppo_runner_cfg

register_mjlab_task(
    task_id="Xiaoli-Velocity-Rough",
    env_cfg=xiaoli_rough_env_cfg(),
    play_env_cfg=xiaoli_rough_env_cfg(play=True),
    rl_cfg=xiaoli_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Xiaoli-Velocity-Flat",
    env_cfg=xiaoli_flat_env_cfg(),
    play_env_cfg=xiaoli_flat_env_cfg(play=True),
    rl_cfg=xiaoli_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
