#!/usr/bin/env python3

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


JOINT_NAMES = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)


def _resolve_mode(base_mode: str, t: float, cycle_seconds: float) -> str:
  if base_mode != "cycle":
    return base_mode
  phase = int(t / cycle_seconds) % 2
  return "stand" if phase == 0 else "gait"


def _get_targets(q_home: np.ndarray, mode: str, t: float, gait_hz: float) -> np.ndarray:
  if mode == "stand":
    return q_home

  # FL/RR in-phase, FR/RL out-of-phase for a small trot-like perturbation.
  phase = np.array(
    [
      0.0,
      0.0,
      0.0,
      math.pi,
      math.pi,
      math.pi,
      math.pi,
      math.pi,
      math.pi,
      0.0,
      0.0,
      0.0,
    ],
    dtype=np.float64,
  )
  amp = np.array(
    [
      0.06,
      0.10,
      0.14,
      0.06,
      0.10,
      0.14,
      0.06,
      0.10,
      0.14,
      0.06,
      0.10,
      0.14,
    ],
    dtype=np.float64,
  )
  sign = np.array(
    [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
    dtype=np.float64,
  )
  s = np.sin(2.0 * math.pi * gait_hz * t + phase)
  return q_home + amp * sign * s


def main() -> None:
  parser = argparse.ArgumentParser(
    description="View Xiaoli in stand/small-gait mode for manual validation."
  )
  parser.add_argument(
    "--mode",
    choices=("stand", "gait", "cycle"),
    default="cycle",
    help="Control mode in viewer. cycle alternates stand and gait.",
  )
  parser.add_argument("--kp", type=float, default=60.0)
  parser.add_argument("--kd", type=float, default=2.0)
  parser.add_argument("--gait-hz", type=float, default=1.2)
  parser.add_argument("--cycle-seconds", type=float, default=4.0)
  args = parser.parse_args()

  model_path = Path(__file__).resolve().parent / "xmls" / "scene.xml"
  model = mujoco.MjModel.from_xml_path(str(model_path))
  data = mujoco.MjData(model)

  if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
  else:
    mujoco.mj_resetData(model, data)

  joint_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES
  ]
  missing = [name for name, jid in zip(JOINT_NAMES, joint_ids, strict=True) if jid < 0]
  if missing:
    raise ValueError(f"Missing joints in model: {missing}")

  qadr = np.array([int(model.jnt_qposadr[jid]) for jid in joint_ids], dtype=np.int32)
  vadr = np.array([int(model.jnt_dofadr[jid]) for jid in joint_ids], dtype=np.int32)
  q_home = np.array([data.qpos[i] for i in qadr], dtype=np.float64)
  joint_id_to_idx = {jid: i for i, jid in enumerate(joint_ids)}

  print(f"Loaded: {model_path}")
  print(
    "Viewer mode:",
    args.mode,
    "| tips: --mode stand (静站), --mode gait (小步态), --mode cycle (切换)",
  )

  with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.perf_counter()
    while viewer.is_running():
      tic = time.perf_counter()
      sim_t = tic - t0
      mode = _resolve_mode(args.mode, sim_t, args.cycle_seconds)

      q = np.array([data.qpos[i] for i in qadr], dtype=np.float64)
      qd = np.array([data.qvel[i] for i in vadr], dtype=np.float64)
      q_target = _get_targets(q_home, mode, sim_t, args.gait_hz)
      tau = args.kp * (q_target - q) - args.kd * qd

      data.ctrl[:] = 0.0
      for a in range(model.nu):
        jid = int(model.actuator_trnid[a, 0])
        idx = joint_id_to_idx.get(jid)
        if idx is None:
          continue
        ctrl = float(tau[idx])
        lo = float(model.actuator_ctrlrange[a, 0])
        hi = float(model.actuator_ctrlrange[a, 1])
        data.ctrl[a] = min(max(ctrl, lo), hi)

      mujoco.mj_step(model, data)
      viewer.sync()

      elapsed = time.perf_counter() - tic
      sleep_for = model.opt.timestep - elapsed
      if sleep_for > 0.0:
        time.sleep(sleep_for)


if __name__ == "__main__":
  main()
