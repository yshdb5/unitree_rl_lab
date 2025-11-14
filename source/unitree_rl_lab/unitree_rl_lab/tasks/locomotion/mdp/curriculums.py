from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

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

def adaptive_penalty_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    target_term_name: str,
    monitor_reward_term: str,
    reward_threshold: float,
    start_weight: float = 0.0,
    end_weight: float = -3.0,
    num_steps_to_ramp: int = 1_000_000,
) -> float:
    """
    Augmente le poids d'une pénalité, MAIS seulement APRÈS qu'un autre
    terme de récompense (monitor_reward_term) a dépassé un seuil.
    Ceci correspond à la stratégie en deux phases du papier.
    """

    # 1. Initialiser l'état du curriculum (la 'phase') si non existant
    if not hasattr(env, "_curriculum_phase_trigger"):
        # 0 = Phase 1 (avant le seuil, pénalité = start_weight)
        # 1 = Phase 2 (après le seuil, rampe en cours)
        # 2 = Phase 2 (rampe terminée, pénalité = end_weight)
        env._curriculum_phase_trigger = 0
        env._curriculum_phase_start_step = 0
        print(f"*** CURRICULUM: Phase 1. '{target_term_name}' poids = {start_weight:.2f} ***")
        print(f"*** CURRICULUM: En attente de '{monitor_reward_term}' > {reward_threshold:.2f} ***")

    current_weight = start_weight

    # 2. Vérifier si le seuil de récompense est atteint (Phase 1)
    if env._curriculum_phase_trigger == 0:
        # Vérifier la récompense moyenne (logique basée sur lin_vel_cmd_levels)
        # Fait la vérification à la fin d'un épisode pour avoir des données complètes
        if env.common_step_counter % env.max_episode_length == 0 and len(env_ids) > 0:
            # Calcule la récompense moyenne par seconde
            avg_reward = torch.mean(env.reward_manager._episode_sums[monitor_reward_term][env_ids]) / env.max_episode_length_s

            # Le seuil est-il atteint ?
            if avg_reward > reward_threshold:
                print(f"*** CURRICULUM: '{monitor_reward_term}' a atteint {avg_reward:.2f} (seuil: {reward_threshold}).")
                print(f"*** CURRICULUM: Phase 2. Démarrage de la rampe pour '{target_term_name}'. ***")
                env._curriculum_phase_trigger = 1
                env._curriculum_phase_start_step = env.common_step_counter
                current_weight = start_weight
    
    # 3. Appliquer la rampe (Phase 2)
    elif env._curriculum_phase_trigger == 1: # Rampe en cours
        progress = min((env.common_step_counter - env._curriculum_phase_start_step) / num_steps_to_ramp, 1.0)
        current_weight = start_weight + (end_weight - start_weight) * progress
        if progress >= 1.0:
            env._curriculum_phase_trigger = 2 # Rampe terminée
            print(f"*** CURRICULUM: Rampe terminée. '{target_term_name}' poids = {end_weight:.2f} ***")
    
    else: # Phase 2 (rampe terminée)
        current_weight = end_weight
    
    # 4. Mettre à jour le poids dans le reward manager
    if target_term_name in env.reward_manager._term_cfgs:
        env.reward_manager._term_cfgs[target_term_name].weight = current_weight

    return current_weight
