from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import math

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time(env, sensor_cfg: SceneEntityCfg, command_name: str, threshold: float):
    """
    Reward airtime: encourages pushing off ground into a jump.
    `command_name` is required by the manager API, but we ignore it for backflip.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Enable ContactSensor.track_air_time=True in scene cfg")

    # airtime per selected foot, then clip & average
    air_times = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]  # [N, num_feet]
    reward = torch.mean(torch.clamp(air_times, min=0.0, max=threshold), dim=1)  # [N]
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


def _axis_angle_from_up(env, axis: str):
    """Angle (0..π) about a target body axis from the body 'up' vector."""
    up_b = env.scene["robot"].data.projected_gravity_b  # [N,3]
    z = torch.clamp(up_b[:, 2], -1.0, 1.0)
    if axis == "pitch":
        num = torch.abs(up_b[:, 0])
    elif axis == "roll":
        num = torch.abs(up_b[:, 1])
    else:
        num = torch.sqrt(up_b[:, 0] ** 2 + up_b[:, 1] ** 2)

    return torch.atan2(num, z)

def yaw_rate_penalty_air(env, sensor_cfg: SceneEntityCfg,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Penalize |yaw rate| only while airborne."""
    asset = env.scene[asset_cfg.name]
    wz = torch.abs(asset.data.root_ang_vel_b[:, 2])
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_f = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    airborne = (foot_f < 1.0).all(dim=1)
    return wz * airborne.float()

def non_target_axis_leak_air(env, sensor_cfg: SceneEntityCfg, target_axis: str):
    """Discourage roll if we want pitch (and vice-versa), only in air."""
    other = "roll" if target_axis == "pitch" else "pitch"
    leak = _axis_angle_from_up(env, other) / math.pi
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_f = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    airborne = (foot_f < 1.0).all(dim=1)
    return leak * airborne.float()

def target_axis_rate_air(env, sensor_cfg: SceneEntityCfg, axis: str,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Encourage angular speed about the target axis while airborne."""
    asset = env.scene[asset_cfg.name]
    if axis == "pitch":
        ang = -asset.data.root_ang_vel_b[:, 1]
    else:
        ang = -asset.data.root_ang_vel_b[:, 0]

    ang = torch.clamp(ang, min=0.0)     # roll uses x, pitch uses y
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_f = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    airborne = (foot_f < 1.0).all(dim=1)
    return ang * airborne.float()

def backflip_progress(env, sensor_cfg: SceneEntityCfg, axis: str,
                      full_rotation_rad: float,
                    min_airtime_for_progress: float = 0.1,
                      upright_bonus: float = 0.0,  # disabled
                      air_only: bool = True,
                      landing_window_s: float = 0.6):
    """Axis-aware progress in [0,1]; only counts in air if air_only=True."""
    angle = _axis_angle_from_up(env, axis)

    progress = torch.clamp(1.0 - (angle / math.pi), 0.0, 1.0)
    # ----------------------

    if air_only:
        contact_sensor = env.scene.sensors[sensor_cfg.name]

        foot_f = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)

        airborne = (foot_f < 1.0).all(dim=1)  # <-- 'foot_f' est maintenant défini
        last_air = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids].min(dim=1).values
        is_real_jump = (last_air >= min_airtime_for_progress)

        progress = progress * airborne.float() * is_real_jump.float()

    return progress

def successful_backflip(env, sensor_cfg: SceneEntityCfg, upright_tol_rad: float, axis: str,
                        min_airtime_s: float, post_land_stable_s: float, full_rotation_rad: float):
    """
    Success: (i) went upside-down about target axis while airborne,
             (ii) then landed and stayed upright on feet for a short window.
    """
    # Per-episode state
    if not hasattr(env, "_flip_state"):
        env._flip_state = {
            "seen_upside_down": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        }
    # Reset flag on episode reset
    at_reset = (env.episode_length_buf == 0)
    if at_reset.any():
        env._flip_state["seen_upside_down"][at_reset] = False

    # Sensors
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_f = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    airborne = (foot_f < 1.0).all(dim=1)
    on_feet_now = (foot_f > 1.0).all(dim=1)

    angle = _axis_angle_from_up(env, axis)  # [0, π]
    env._flip_state["seen_upside_down"] |= airborne & (angle <= 0.1 * math.pi)

    # Jumped & stable landing window
    last_air = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids].min(dim=1).values
    jumped = last_air >= min_airtime_s
    last_contact = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids].min(dim=1).values
    landed_stable = last_contact >= post_land_stable_s

    # Upright now
    up_b = env.scene["robot"].data.projected_gravity_b
    upright_now = up_b[:, 2] >= torch.cos(torch.tensor(upright_tol_rad, device=env.device))

    success = jumped & env._flip_state["seen_upside_down"] & on_feet_now & landed_stable & upright_now
    env._flip_state["seen_upside_down"][success] = False  # one-shot
    return success

def upward_vel_air_airborne(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Reward upward base velocity ONLY when all feet are in the air.
    """
    # upward velocity
    asset = env.scene[asset_cfg.name]
    zvel = torch.clamp(asset.data.root_lin_vel_w[:, 2], min=0.0)

    # airborne mask
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_f = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    airborne = (foot_f < 1.0).all(dim=1)    # [N]

    return zvel * airborne.float()

def post_flip_land_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, min_airtime_s: float) -> torch.Tensor:
    """Rewards all feet being in contact AFTER a jump."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Check for landing on feet (all feet have contact)
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] # [N, num_feet]
    on_feet = (foot_forces.abs() > 1.0).all(dim=1) # [N]

    # Check that a jump just happened (minimum air time was met)
    last_air = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]      # [N, 4]
    jumped = (last_air.min(dim=1).values >= min_airtime_s)

    return (on_feet & jumped).float()

def ang_vel_x_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize angular velocity in the x-direction (roll)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 0])


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize angular velocity in the z-direction (yaw)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])