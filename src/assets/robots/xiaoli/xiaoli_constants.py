"""Xiaoli constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

_MJLAB_XIAOLI_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "xiaoli" / "xmls" / "xiaoli.xml"
)
_LOCAL_XIAOLI_XML: Path = Path(__file__).resolve().parent / "xmls" / "xiaoli.xml"

# Prefer mjlab's asset path when available, and fall back to the local copy
# bundled in this repository for branches that do not ship Xiaoli assets.
XIAOLI_XML: Path = (
    _MJLAB_XIAOLI_XML if _MJLAB_XIAOLI_XML.exists() else _LOCAL_XIAOLI_XML
)
assert XIAOLI_XML.exists(), f"Xiaoli XML not found: {XIAOLI_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, XIAOLI_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(XIAOLI_XML))
    spec.assets = get_assets(spec.meshdir)
    # Strip the XML <motor> actuators; BuiltinPositionActuatorCfg adds its own.
    for act in list(spec.actuators):
        spec.delete(act)
    return spec


##
# Actuator config.
##

# PD gains and effort limits.
_XIAOLI_HIP_STIFFNESS = 40.0
_XIAOLI_KNEE_STIFFNESS = 60.0

# Hip and thigh joints share the same stiffness/effort_limit.
XIAOLI_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_(hip|thigh)_joint",),
    stiffness=_XIAOLI_HIP_STIFFNESS,
    damping=2.0,
    effort_limit=23.7,
)

XIAOLI_CALF_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_calf_joint",),
    stiffness=_XIAOLI_KNEE_STIFFNESS,
    damping=2.0,
    effort_limit=45.43,
)

XIAOLI_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XIAOLI_HIP_ACTUATOR_CFG, XIAOLI_CALF_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.9,
)

##
# Initial state.
##

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.42),
    joint_pos={
        ".*_hip_joint": 0.0,
        ".*_thigh_joint": 0.9,
        ".*_calf_joint": -1.8,
    },
    joint_vel={".*": 0.0},
)


##
# Final config.
##


def get_xiaoli_robot_cfg() -> EntityCfg:
    """Get a fresh Xiaoli robot configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_spec,
        articulation=XIAOLI_ARTICULATION,
    )


# Conservative defaults for position-action scaling used by some task configs.
XIAOLI_ACTION_SCALE: dict[str, float] = {
    ".*_hip_joint": 0.25 * 23.7 / _XIAOLI_HIP_STIFFNESS,
    ".*_thigh_joint": 0.25 * 23.7 / _XIAOLI_HIP_STIFFNESS,
    ".*_calf_joint": 0.25 * 45.43 / _XIAOLI_KNEE_STIFFNESS,
}


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_xiaoli_robot_cfg())

    viewer.launch(robot.spec.compile())
