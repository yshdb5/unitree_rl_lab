from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the flat terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.5, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. Reward angular velocity perpendicular to sagittal plane (pitch)
    reward_pitch_velocity = RewTerm(func=mdp.backflip_pitch_velocity, weight=7.0)

    # 2. Penalize other angular velocities (roll, yaw)
    penalize_roll_yaw_velocity = RewTerm(func=mdp.backflip_roll_yaw_velocity, weight=-1.0)

    # 3. Penalize use of joint torque
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-5e-6)

    # 4. Penalize early termination (weight starts at 0 and is increased by curriculum)
    early_termination = RewTerm(func=mdp.early_termination_penalty, weight=0.0)

    # 5. Penalize not being upright (to encourage landing on feet)
    penalize_not_upright = RewTerm(func=mdp.upward, weight=-3.0)

    # 6. Reward height to encourage jumping
    jump_height = RewTerm(func=mdp.reward_height, weight=4.0)
    
    # 7. Encourage symmetric hind-leg push (fix rear right lag)
    leg_symmetry = RewTerm(
        func=mdp.leg_action_symmetry,
        weight=2.0,
        params={
            "right_leg_ids": [7, 8],   # RR_thigh, RR_calf
            "left_leg_ids": [10, 11],  # RL_thigh, RL_calf
        },
    )
    
    # 8. Extra reward for achieving full 360° rotation (pitch ≈ -2π)
    full_flip = RewTerm(func=mdp.full_flip_completion, weight=6.0)

    # Regularization
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    successful_flip = DoneTerm(
        func=mdp.full_flip_done,
        params={
            "angle_tol": 0.4,
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    landing_penalty = CurrTerm(
        func=mdp.backflip_landing_curriculum,
        params={
            "reward_term_name": "early_termination",
            "progress_reward_name": "reward_pitch_velocity",
            "progress_threshold": 2.0,
            "max_penalty_weight": -20.0,
        },
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the backflip environment."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 4.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Configuration for the backflip environment for play."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # set number of environments to a small number for play
        self.scene.num_envs = 32
        # disable curriculum
        self.curriculum: CurriculumCfg = None
        # disable events other than reset
        self.events = EventCfg()
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {},
            },
        )
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (1.0, 1.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        # observation noise is disabled by default in play mode