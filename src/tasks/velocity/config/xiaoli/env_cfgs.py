"""Xiaoli velocity environment configurations."""

from src.assets.robots.xiaoli.xiaoli_constants import (
    XIAOLI_ACTION_SCALE,
    get_xiaoli_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def xiaoli_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Xiaoli rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500

    cfg.scene.entities = {"robot": get_xiaoli_robot_cfg()}

    # Set raycast sensor frame to Xiaoli base body.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = "base"

    foot_names = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="body", pattern=foot_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # Xiaoli's lower-leg meshes can legitimately brush terrain in resting poses.
    # Treat only base-ground contact as illegal to avoid constant reset loops.
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="body",
            entity="robot",
            pattern="base",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        nonfoot_ground_cfg,
    )

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = XIAOLI_ACTION_SCALE

    cfg.viewer.body_name = "base"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    # Xiaoli XML does not expose imu_lin_vel/imu_ang_vel builtins.
    cfg.observations["actor"].terms["base_lin_vel"].func = mdp.base_lin_vel
    cfg.observations["actor"].terms["base_lin_vel"].params = {}
    cfg.observations["actor"].terms["base_ang_vel"].func = mdp.base_ang_vel
    cfg.observations["actor"].terms["base_ang_vel"].params = {}
    cfg.observations["critic"].terms["base_lin_vel"].func = mdp.base_lin_vel
    cfg.observations["critic"].terms["base_lin_vel"].params = {}
    cfg.observations["critic"].terms["base_ang_vel"].func = mdp.base_ang_vel
    cfg.observations["critic"].terms["base_ang_vel"].params = {}

    cfg.events.pop("foot_friction", None)
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

    cfg.rewards["pose"].params["std_standing"] = {
        r".*_(hip|thigh)_joint.*": 0.05,
        r".*_calf_joint.*": 0.1,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r".*_(hip|thigh)_joint.*": 0.3,
        r".*_calf_joint.*": 0.6,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*_(hip|thigh)_joint.*": 0.3,
        r".*_calf_joint.*": 0.6,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)

    # Xiaoli XML does not define explicit foot sites, so drop site-based terms.
    cfg.observations["critic"].terms.pop("foot_height", None)
    cfg.rewards.pop("foot_clearance", None)
    cfg.rewards.pop("foot_swing_height", None)
    cfg.rewards.pop("foot_slip", None)
    cfg.rewards.pop("angular_momentum", None)

    cfg.rewards["body_ang_vel"].weight = 0.0
    cfg.rewards["air_time"].weight = 0.0

    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
    )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def xiaoli_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Xiaoli flat terrain velocity configuration."""
    cfg = xiaoli_rough_env_cfg(play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove raycast sensor and height scan (no terrain to scan).
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Disable terrain curriculum (not present in play mode since rough clears all).
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg
