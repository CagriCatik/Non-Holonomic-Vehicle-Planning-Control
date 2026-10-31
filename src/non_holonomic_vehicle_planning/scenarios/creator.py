"""Scenario Creator (Szenarien-Ersteller) with Fluent Builder API & YAML Serialization.

Allows creating static and dynamic traffic scenarios (buildings, parked cars, moving
vehicles, crossing pedestrians, cyclists, and multi-layer inflation parameters) either
programmatically in Python or declaratively via YAML files in ``configs/scenarios/``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon
import yaml

from non_holonomic_vehicle_planning.config import (
    AppConfig,
    InflationConfig,
    get_configs_dir,
    load_config,
    load_yaml_dict,
    parse_inflation_config,
    parse_vehicle_params,
)
from non_holonomic_vehicle_planning.dynamic_obstacles import DynamicObstacle, MotionMode
from non_holonomic_vehicle_planning.obstacles import ObstacleBuilder
from non_holonomic_vehicle_planning.scenario import Scenario
from non_holonomic_vehicle_planning.vehicle import VehicleParams


class ScenarioCreator:
    """Fluent builder and YAML loader/exporter for static and dynamic planning scenarios."""

    def __init__(
        self,
        name: str = "Custom_Scenario",
        width: float = 50.0,
        height: float = 50.0,
        boundary_collision: bool = True,
        config_path: Optional[str | Path] = None,
    ) -> None:
        self.name = name
        self.width = float(width)
        self.height = float(height)
        self.boundary_collision = bool(boundary_collision)

        app_cfg: AppConfig = load_config(config_path)
        self.vehicle: VehicleParams = app_cfg.vehicle
        self.inflation: InflationConfig = app_cfg.inflation
        self.raw_config: Dict[str, Any] = app_cfg.raw

        self._obstacle_builder = ObstacleBuilder()
        self._rect_specs: List[Dict[str, Any]] = []
        self._poly_specs: List[List[List[float]]] = []
        self._dynamic_obstacles: List[DynamicObstacle] = []

        self.start_xy: Tuple[float, float] = (5.0, 5.0)
        self.start_yaw_deg: float = 0.0
        self.goal_xy: Tuple[float, float] = (40.0, 40.0)
        self.goal_yaw_deg: float = 0.0

    def set_environment(
        self,
        width: float,
        height: float,
        boundary_collision: bool = True,
    ) -> "ScenarioCreator":
        """Configure environment dimensions [m] and boundary collision check."""
        self.width = float(width)
        self.height = float(height)
        self.boundary_collision = bool(boundary_collision)
        return self

    def set_vehicle(
        self,
        wheelbase: float = 2.9,
        width: float = 2.0,
        rear_overhang: float = 0.8,
        front_overhang: float = 0.8,
        max_curvature: float = 1.0 / 4.39,
        max_steer_deg: float = 45.0,
    ) -> "ScenarioCreator":
        """Configure 2D planar vehicle parameters."""
        self.vehicle = VehicleParams(
            wheelbase=wheelbase,
            width=width,
            rear_overhang=rear_overhang,
            front_overhang=front_overhang,
            wheel_track=0.75 * width,
            max_steer=math.radians(max_steer_deg),
            explicit_max_curvature=max_curvature,
        )
        return self

    def set_inflation(
        self,
        enabled: bool = True,
        hard_margin_m: float = 0.30,
        soft_margin_m: float = 1.20,
        pedestrian_safety_margin_m: float = 1.15,
        dynamic_vehicle_margin_m: float = 0.65,
        velocity_lookahead_time_s: float = 0.65,
    ) -> "ScenarioCreator":
        """Configure static and dynamic obstacle inflation margins."""
        self.inflation = InflationConfig(
            enabled=enabled,
            hard_margin_m=hard_margin_m,
            soft_margin_m=soft_margin_m,
            pedestrian_safety_margin_m=pedestrian_safety_margin_m,
            dynamic_vehicle_margin_m=dynamic_vehicle_margin_m,
            velocity_lookahead_time_s=velocity_lookahead_time_s,
        )
        return self

    def set_start_goal(
        self,
        start: Tuple[float, float],
        start_yaw_deg: float,
        goal: Tuple[float, float],
        goal_yaw_deg: float,
    ) -> "ScenarioCreator":
        """Set ego vehicle start and goal poses (x [m], y [m], yaw [deg])."""
        self.start_xy = (float(start[0]), float(start[1]))
        self.start_yaw_deg = float(start_yaw_deg)
        self.goal_xy = (float(goal[0]), float(goal[1]))
        self.goal_yaw_deg = float(goal_yaw_deg)
        return self

    def add_rectangle_obstacle(
        self,
        width: float,
        height: float,
        bottom_left: Tuple[float, float],
    ) -> "ScenarioCreator":
        """Add a static rectangular obstacle."""
        self._obstacle_builder.add_rectangle(width, height, bottom_left)
        self._rect_specs.append({
            "width": float(width),
            "height": float(height),
            "bottom_left": [float(bottom_left[0]), float(bottom_left[1])],
        })
        return self

    def add_polygon_obstacle(
        self,
        coords: Sequence[Tuple[float, float]],
    ) -> "ScenarioCreator":
        """Add an arbitrary static polygon obstacle."""
        self._obstacle_builder.add_polygon(coords)
        self._poly_specs.append([[float(x), float(y)] for x, y in coords])
        return self

    def add_parked_vehicle(
        self,
        center: Tuple[float, float],
        yaw_deg: float = 90.0,
        length: float = 4.5,
        width: float = 2.0,
    ) -> "ScenarioCreator":
        """Add a static parked vehicle obstacle at arbitrary position and yaw orientation."""
        hl, hw = 0.5 * length, 0.5 * width
        local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=float)
        rad = math.radians(yaw_deg)
        c, s = math.cos(rad), math.sin(rad)
        rot = np.array([[c, s], [-s, c]], dtype=float)
        world = local @ rot + np.array([center[0], center[1]], dtype=float)
        coords = [(float(pt[0]), float(pt[1])) for pt in world]
        return self.add_polygon_obstacle(coords)

    def add_moving_vehicle(
        self,
        id: str,
        start: Tuple[float, float],
        yaw_deg: float = 0.0,
        speed: float = 2.0,
        motion_mode: MotionMode = "constant_velocity",
        waypoints: Optional[Sequence[Tuple[float, float]]] = None,
        length: float = 4.5,
        width: float = 2.0,
        safety_margin: Optional[float] = None,
        start_delay_s: float = 0.0,
    ) -> "ScenarioCreator":
        """Add a moving vehicle obstacle to the scenario."""
        wps = [(float(p[0]), float(p[1])) for p in waypoints] if waypoints else []
        self._dynamic_obstacles.append(
            DynamicObstacle(
                id=id,
                kind="vehicle",
                start=(float(start[0]), float(start[1])),
                yaw_deg=float(yaw_deg),
                speed=float(speed),
                motion_mode=motion_mode,
                waypoints=wps,
                length=float(length),
                width=float(width),
                safety_margin=safety_margin,
                start_delay_s=start_delay_s,
            )
        )
        return self

    def add_pedestrian(
        self,
        id: str,
        start: Tuple[float, float],
        goal: Optional[Tuple[float, float]] = None,
        speed: float = 1.25,
        periodic: bool = True,
        yaw_deg: float = 0.0,
        radius: float = 0.40,
        safety_margin: Optional[float] = None,
        start_delay_s: float = 0.0,
    ) -> "ScenarioCreator":
        """Add a walking pedestrian (Fußgänger) on or crossing the roadway."""
        if goal is not None:
            wps = [(float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))]
            mode: MotionMode = "periodic_ping_pong" if periodic else "waypoints"
            calc_yaw = math.degrees(math.atan2(goal[1] - start[1], goal[0] - start[0]))
        else:
            wps = []
            mode = "constant_velocity"
            calc_yaw = yaw_deg

        self._dynamic_obstacles.append(
            DynamicObstacle(
                id=id,
                kind="pedestrian",
                start=(float(start[0]), float(start[1])),
                yaw_deg=float(calc_yaw),
                speed=float(speed),
                motion_mode=mode,
                waypoints=wps,
                radius=float(radius),
                safety_margin=safety_margin,
                start_delay_s=start_delay_s,
            )
        )
        return self

    def build(self) -> Scenario:
        """Build and return the configured ``Scenario`` instance."""
        sc = Scenario(
            name=self.name,
            env_width=self.width,
            env_height=self.height,
            boundary_collision=self.boundary_collision,
            vehicle=self.vehicle,
            inflation_config=self.inflation,
        )
        sc.set_obstacles(self._obstacle_builder.build())
        sc.set_dynamic_obstacles(self._dynamic_obstacles)
        sc.set_start_goal(self.start_xy, self.start_yaw_deg, self.goal_xy, self.goal_yaw_deg)
        return sc

    def to_dict(self) -> Dict[str, Any]:
        """Export scenario specification as a YAML-serializable dictionary."""
        return {
            "scenario": {
                "name": self.name,
                "width": self.width,
                "height": self.height,
                "boundary_collision": self.boundary_collision,
                "start": {"x": self.start_xy[0], "y": self.start_xy[1], "yaw_deg": self.start_yaw_deg},
                "goal": {"x": self.goal_xy[0], "y": self.goal_xy[1], "yaw_deg": self.goal_yaw_deg},
                "static_obstacles": {
                    "rectangles": self._rect_specs,
                    "polygons": self._poly_specs,
                },
                "dynamic_obstacles": [d.to_dict() for d in self._dynamic_obstacles],
            },
            "inflation": {
                "enabled": self.inflation.enabled,
                "hard_margin_m": self.inflation.hard_margin_m,
                "soft_margin_m": self.inflation.soft_margin_m,
                "use_multi_disk_footprint": self.inflation.use_multi_disk_footprint,
                "num_footprint_disks": self.inflation.num_footprint_disks,
                "pedestrian_radius_m": self.inflation.pedestrian_radius_m,
                "pedestrian_safety_margin_m": self.inflation.pedestrian_safety_margin_m,
                "dynamic_vehicle_margin_m": self.inflation.dynamic_vehicle_margin_m,
                "velocity_lookahead_time_s": self.inflation.velocity_lookahead_time_s,
                "soft_cost_weight": self.inflation.soft_cost_weight,
            },
        }

    def to_yaml(self, path: str | Path) -> Path:
        """Write scenario definition to a YAML configuration file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioCreator":
        """Load a ``ScenarioCreator`` from a YAML file (merged with ``configs/default.yaml``)."""
        p = Path(path)
        if not p.exists():
            alt = get_configs_dir() / "scenarios" / p.name
            if not alt.suffix:
                alt = alt.with_suffix(".yaml")
            if alt.exists():
                p = alt

        data = load_yaml_dict(p)
        sc_data = data.get("scenario", {})
        creator = cls(
            name=str(sc_data.get("name", p.stem)),
            width=float(sc_data.get("width", 50.0)),
            height=float(sc_data.get("height", 50.0)),
            boundary_collision=bool(sc_data.get("boundary_collision", False)),
        )
        creator.raw_config = data
        creator.vehicle = parse_vehicle_params(data)
        creator.inflation = parse_inflation_config(data)

        st = sc_data.get("start", {"x": 5.0, "y": 5.0, "yaw_deg": 0.0})
        gl = sc_data.get("goal", {"x": 40.0, "y": 40.0, "yaw_deg": 0.0})
        creator.set_start_goal(
            (float(st["x"]), float(st["y"])),
            float(st.get("yaw_deg", 0.0)),
            (float(gl["x"]), float(gl["y"])),
            float(gl.get("yaw_deg", 0.0)),
        )

        static_obs = sc_data.get("static_obstacles", {})
        for rect in static_obs.get("rectangles", []):
            bl = rect["bottom_left"]
            creator.add_rectangle_obstacle(
                float(rect["width"]),
                float(rect["height"]),
                (float(bl[0]), float(bl[1])),
            )
        for poly in static_obs.get("polygons", []):
            creator.add_polygon_obstacle([(float(pt[0]), float(pt[1])) for pt in poly])

        for dyn_dict in sc_data.get("dynamic_obstacles", []):
            creator._dynamic_obstacles.append(DynamicObstacle.from_dict(dyn_dict))

        return creator

