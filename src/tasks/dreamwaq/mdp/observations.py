from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase


def cenet_features(env: ManagerBasedRlEnv, history_length: int) -> torch.Tensor:
    """Extract CENet features (estimated velocity + context vector) for DreamWaq.

    Args:
        env: The RL environment.
        history_length: Number of past observation frames to use.

    Returns:
        Tensor of shape (num_envs, 3 + 16) containing estimated velocity (3D) and
        context vector (16D).
    """
    # Access the runner via env (runner should be attached as an attribute)
    if hasattr(env.unwrapped, 'runner') and env.unwrapped.runner is not None:
        runner = env.unwrapped.runner
        # Use pre‑computed CENet outputs stored in the runner
        # These are set by runner.cenet.before_action before each action
        est_vel = runner.current_est_vel
        context_vec = runner.current_context_vec
        if est_vel is None or context_vec is None:
            # Fallback: compute from history buffer (should not happen during training)
            env_ids = torch.arange(env.num_envs, device=env.device)
            hist_obs = runner.get_history_observations(env_ids, history_length)
            batch_size = hist_obs.shape[0]
            hist_obs_flat = hist_obs.reshape(batch_size, -1)
            with torch.no_grad():
                _, est_vel, _, _, context_vec = runner.cenet(hist_obs_flat)
        # Concatenate features
        features = torch.cat([est_vel, context_vec], dim=-1)
        return features
    else:
        # No runner attached (play mode with dummy agents)
        # Return zero features with correct shape
        num_envs = env.num_envs
        device = env.device
        return torch.zeros((num_envs, 3 + 16), device=device)