"""Top-level Model Predictive Control (MPC) module re-exporting all MPC classes & YAML factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import math

from non_holonomic_vehicle_planning.config import load_yaml_dict, parse_vehicle_params
from non_holonomic_vehicle_planning.control.kinematic_model import KinematicState, Node, calc_linear_discrete_model
from non_holonomic_vehicle_planning.control.mpc import (
    MPCConfig,
    MPCController,
    PATH,
    ReferencePath,
    calc_speed_profile,
    pi_2_pi,
    smooth_reference_yaw,
)
from non_holonomic_vehicle_planning.vehicle import VehicleParams


def parse_mpc_config(data: Dict[str, Any]) -> MPCConfig:
    """Construct ``MPCConfig`` from a YAML ``mpc`` dictionary."""
    m = data.get("mpc", data)
    q_diag = m.get("Q_diag", [4.5, 4.5, 2.5, 4.0])
    qf_diag = m.get("Qf_diag", [6.0, 6.0, 2.0, 4.5])
    r_diag = m.get("R_diag", [0.01, 0.15])
    rd_diag = m.get("Rd_diag", [0.01, 0.15])
    return MPCConfig(
        horizon=int(m.get("horizon", 14)),
        max_lateral_accel=float(m["max_lateral_accel_m_s2"]) if m.get("max_lateral_accel_m_s2") is not None else None,
        Q=np.diag([float(x) for x in q_diag]),
        Qf=np.diag([float(x) for x in qf_diag]),
        R=np.diag([float(x) for x in r_diag]),
        Rd=np.diag([float(x) for x in rd_diag]),
        dt=float(m.get("dt", 0.2)),
        target_speed=float(m.get("target_speed_kmh", 11.0)) / 3.6,
        dist_stop=float(m.get("dist_stop", 0.65)),
        speed_stop=float(m.get("speed_stop", 0.15)),
        heading_stop=math.radians(float(m.get("heading_stop_deg",5.))),
        time_max=float(m.get("time_max", 120.0)),
        iter_max=int(m.get("iter_max", 4)),
        du_res=float(m.get("du_res", 0.08)),
        steer_du_res=float(m.get("steer_du_res",.005)),
        search_window=int(m.get("search_window", 18)),
    )


def create_mpc_from_yaml(
    config_path: Optional[str | Path] = None,
    vehicle: Optional[VehicleParams] = None,
) -> MPCController:
    """Instantiate an ``MPCController`` directly from ``configs/default.yaml`` or a custom YAML file."""
    raw = load_yaml_dict(config_path)
    mpc_cfg = parse_mpc_config(raw)
    veh = vehicle or parse_vehicle_params(raw)
    return MPCController(config=mpc_cfg, vehicle=veh)


__all__ = [
    "KinematicState",
    "MPCConfig",
    "MPCController",
    "Node",
    "PATH",
    "ReferencePath",
    "calc_linear_discrete_model",
    "calc_speed_profile",
    "create_mpc_from_yaml",
    "parse_mpc_config",
    "pi_2_pi",
    "smooth_reference_yaw",
]

