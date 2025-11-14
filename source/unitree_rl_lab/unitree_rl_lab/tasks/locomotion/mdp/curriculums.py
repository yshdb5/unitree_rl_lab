from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)

def scalar_schedule(env, target_attr, start, end, num_steps):
    """
    Linearly interpolates a parameter across training.
    """
    progress = env.training_step / max(1, num_steps)
    progress = torch.clam(progress, 0.0, 1.0)
    value = start + progress * (end - start)

    # Apply nested attribute update
    obj = env
    for key in target_attr[:-1]:
        obj = getattr(obj, key)
    setattr(obj, target_attr[-1], value)

    return value

def progressive_penalty_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_weight: float = 0.0,
    end_weight: float = -3.0,
    num_steps: int = 1_000_000,
) -> float:
    """Augmente progressivement le poids d'une pénalité de récompense.

    Args:
        env: L'environnement RL
        env_ids: IDs des environnements (non utilisé mais requis)
        term_name: Nom du terme de récompense à modifier
        start_weight: Poids initial (ex: 0.0 = pas de pénalité)
        end_weight: Poids final (ex: -3.0 = forte pénalité)
        num_steps: Nombre de pas pour atteindre le poids final

    Returns:
        Le poids actuel basé sur la progression
    """
    # Calculer la progression linéaire
    progress = min(env.common_step_counter / num_steps, 1.0)
    current_weight = start_weight + (end_weight - start_weight) * progress

    # Mettre à jour dynamiquement le poids dans le reward manager
    if term_name in env.reward_manager._term_cfgs:
        env.reward_manager._term_cfgs[term_name].weight = current_weight

    return current_weight


def check_undesired_contacts(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_times = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    is_contact = contact_times > 0.0

    return torch.any(is_contact, dim=1)

