"""YAML Configuration Loader and Typed Settings Schema."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from non_holonomic_vehicle_planning.vehicle import VehicleParams


def get_project_root() -> Path:
    """Return the root directory of the repository."""
    return Path(__file__).resolve().parents[2]


def get_configs_dir() -> Path:
    """Return the path to the ``configs/`` directory."""
    return get_project_root() / "configs"


@dataclass
class InflationConfig:
    """Parameters for static and dynamic obstacle inflation (Minkowski safety buffers)."""

    enabled: bool = True
    hard_margin_m: float = 0.30
    soft_margin_m: float = 1.20
    use_multi_disk_footprint: bool = True
    num_footprint_disks: int = 3
    pedestrian_radius_m: float = 0.40
    pedestrian_safety_margin_m: float = 1.15
    dynamic_vehicle_margin_m: float = 0.65
    velocity_lookahead_time_s: float = 0.65
    soft_cost_weight: float = 6.0


@dataclass
class RealTimePlannerConfig:
    """Parameters for real-time local trajectory replanning around moving obstacles."""

    enabled: bool = True
    lookahead_distance_m: float = 18.0
    prediction_horizon_s: float = 4.5
    lateral_offsets_m: List[float] = field(
        default_factory=lambda: [-4.5, -3.5, -2.5, -1.6, 1.6, 2.5, 3.5, 4.5]
    )
    replan_clearance_trigger_m: float = 0.45
    yield_stop_buffer_m: float = 2.5


@dataclass
class VisualizationConfig:
    """Parameters for 2D planar vehicle plots and GIF animations."""

    fps: int = 15
    playback_speed: float = 1.2
    dpi: int = 90
    closeup_radius_m: float = 7.0
    max_frames: int = 140
    show_inflation_zones: bool = True
    show_dynamic_predictions: bool = True
    show_mpc_horizon: bool = True
    show_steer_rays: bool = True


@dataclass
class AppConfig:
    """Complete application configuration loaded from YAML."""

    vehicle: VehicleParams
    inflation: InflationConfig
    realtime: RealTimePlannerConfig
    visualization: VisualizationConfig
    raw: Dict[str, Any] = field(default_factory=dict)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_yaml_dict(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load configuration dictionary from ``configs/default.yaml``, optionally merged with ``path``."""
    default_path = get_configs_dir() / "default.yaml"
    base_data: Dict[str, Any] = {}
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            base_data = yaml.safe_load(f) or {}

    if path is not None:
        custom_path = Path(path)
        if not custom_path.exists() and (get_configs_dir() / custom_path).exists():
            custom_path = get_configs_dir() / custom_path
        with open(custom_path, "r", encoding="utf-8") as f:
            override_data = yaml.safe_load(f) or {}
        base_data = _deep_merge(base_data, override_data)

    return base_data


def parse_vehicle_params(data: Dict[str, Any]) -> VehicleParams:
    """Construct ``VehicleParams`` from a YAML ``vehicle`` section."""
    v = data.get("vehicle", data)
    return VehicleParams(
        wheelbase=float(v.get("wheelbase", 2.9)),
        width=float(v.get("width", 2.0)),
        rear_overhang=float(v.get("rear_overhang", 0.8)),
        front_overhang=float(v.get("front_overhang", 0.8)),
        wheel_track=float(v.get("wheel_track", 1.50)),
        tire_radius=float(v.get("tire_radius", 0.40)),
        tire_width=float(v.get("tire_width", 0.30)),
        max_steer=math.radians(float(v.get("max_steer_deg", 45.0))),
        max_steer_rate=math.radians(float(v.get("max_steer_rate_deg_s", 50.0))),
        max_steer_accel=math.radians(float(v.get("max_steer_accel_deg_s2", 180.0))),
        steer_tau=float(v.get("steer_tau_s", 0.16)),
        accel_tau=float(v.get("accel_tau_s", 0.10)),
        ackermann_steering=bool(v.get("ackermann_steering", True)),
        max_speed=float(v.get("max_speed_kmh", 55.0)) / 3.6,
        min_speed=float(v.get("min_speed_kmh", -20.0)) / 3.6,
        max_accel=float(v.get("max_accel", 1.8)),
        explicit_max_curvature=float(v["max_curvature"]) if "max_curvature" in v else None,
    )


def parse_inflation_config(data: Dict[str, Any]) -> InflationConfig:
    """Construct ``InflationConfig`` from a YAML ``inflation`` section."""
    inf = data.get("inflation", data)
    return InflationConfig(
        enabled=bool(inf.get("enabled", True)),
        hard_margin_m=float(inf.get("hard_margin_m", 0.30)),
        soft_margin_m=float(inf.get("soft_margin_m", 1.20)),
        use_multi_disk_footprint=bool(inf.get("use_multi_disk_footprint", True)),
        num_footprint_disks=int(inf.get("num_footprint_disks", 3)),
        pedestrian_radius_m=float(inf.get("pedestrian_radius_m", 0.40)),
        pedestrian_safety_margin_m=float(inf.get("pedestrian_safety_margin_m", 1.15)),
        dynamic_vehicle_margin_m=float(inf.get("dynamic_vehicle_margin_m", 0.65)),
        velocity_lookahead_time_s=float(inf.get("velocity_lookahead_time_s", 0.65)),
        soft_cost_weight=float(inf.get("soft_cost_weight", 6.0)),
    )


def parse_realtime_config(data: Dict[str, Any]) -> RealTimePlannerConfig:
    """Construct ``RealTimePlannerConfig`` from a YAML ``planner.realtime`` section."""
    pl = data.get("planner", {})
    rt = pl.get("realtime", data.get("realtime", {}))
    return RealTimePlannerConfig(
        enabled=bool(rt.get("enabled", True)),
        lookahead_distance_m=float(rt.get("lookahead_distance_m", 18.0)),
        prediction_horizon_s=float(rt.get("prediction_horizon_s", 4.5)),
        lateral_offsets_m=[float(x) for x in rt.get("lateral_offsets_m", [-4.5, -3.5, -2.5, -1.6, 1.6, 2.5, 3.5, 4.5])],
        replan_clearance_trigger_m=float(rt.get("replan_clearance_trigger_m", 0.45)),
        yield_stop_buffer_m=float(rt.get("yield_stop_buffer_m", 2.5)),
    )


def parse_visualization_config(data: Dict[str, Any]) -> VisualizationConfig:
    """Construct ``VisualizationConfig`` from a YAML ``visualization`` section."""
    vis = data.get("visualization", data)
    return VisualizationConfig(
        fps=int(vis.get("fps", 15)),
        playback_speed=float(vis.get("playback_speed", 1.2)),
        dpi=int(vis.get("dpi", 90)),
        closeup_radius_m=float(vis.get("closeup_radius_m", 7.0)),
        max_frames=int(vis.get("max_frames", 140)),
        show_inflation_zones=bool(vis.get("show_inflation_zones", True)),
        show_dynamic_predictions=bool(vis.get("show_dynamic_predictions", True)),
        show_mpc_horizon=bool(vis.get("show_mpc_horizon", True)),
        show_steer_rays=bool(vis.get("show_steer_rays", True)),
    )


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    """Load complete ``AppConfig`` from ``configs/default.yaml`` and optional override file."""
    raw = load_yaml_dict(path)
    return AppConfig(
        vehicle=parse_vehicle_params(raw),
        inflation=parse_inflation_config(raw),
        realtime=parse_realtime_config(raw),
        visualization=parse_visualization_config(raw),
        raw=raw,
    )

