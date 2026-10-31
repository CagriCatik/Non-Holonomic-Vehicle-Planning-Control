"""Scenario definition, multi-layer obstacle inflation, dynamic obstacles, and trajectory I/O."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import shapely
import shapely.geometry
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from non_holonomic_vehicle_planning.config import InflationConfig
from non_holonomic_vehicle_planning.dynamic_obstacles import DynamicObstacle
from non_holonomic_vehicle_planning.inflation import ObstacleInflater
from non_holonomic_vehicle_planning.planning.base import PlannedPath, TrajectorySegment, sanitize_csv_trajectory
from non_holonomic_vehicle_planning.vehicle import VehicleParams, wrap_angle


class Scenario:
    """Represents a 2D motion planning environment with static & dynamic obstacles, inflation, and poses."""

    def __init__(
        self,
        name: str,
        env_width: float,
        env_height: float,
        boundary_collision: bool = False,
        vehicle: Optional[VehicleParams] = None,
        inflation_config: Optional[InflationConfig] = None,
    ) -> None:
        self.name = name
        self.width = float(env_width)
        self.height = float(env_height)
        self.boundary_collision = bool(boundary_collision)

        if self.boundary_collision:
            bbox_coords = [
                (0.0, 0.0),
                (0.0, self.height),
                (self.width, self.height),
                (self.width, 0.0),
                (0.0, 0.0),
            ]
            self.boundary: Optional[LineString] = LineString(bbox_coords)
        else:
            self.boundary = None

        self.vehicle: VehicleParams = vehicle or VehicleParams()
        self.inflation_config: InflationConfig = inflation_config or InflationConfig()
        self.inflater: ObstacleInflater = ObstacleInflater(self.vehicle, self.inflation_config)

        self.vehicle_length: float = self.vehicle.length
        self.vehicle_width: float = self.vehicle.width
        self.max_curvature: List[float] = self.vehicle.curvature_candidates

        self.obstacles: List[Polygon] = []
        self.buffered_obstacles: List[Polygon] = []
        self.soft_buffered_obstacles: List[Polygon] = []
        self._buffered_union: Optional[shapely.geometry.base.BaseGeometry] = None
        self._soft_union: Optional[shapely.geometry.base.BaseGeometry] = None
        self._raw_union: Optional[shapely.geometry.base.BaseGeometry] = None

        self.dynamic_obstacles: List[DynamicObstacle] = []

        self.start: Optional[Tuple[Point, float]] = None  # (Point(x, y), yaw_deg)
        self.goal: Optional[Tuple[Point, float]] = None   # (Point(x, y), yaw_deg)

        self.planned_path: Optional[PlannedPath] = None
        self.path: List[LineString] = []
        self.total_tree: List[LineString] = []
        self.tree_segments: List[TrajectorySegment] = []
        self.tree_growth_history: List[object] = []

    @property
    def start_pose_rad(self) -> Tuple[float, float, float]:
        """Return (x, y, yaw_rad) for the start pose."""
        if self.start is None:
            raise ValueError("Start pose has not been set on Scenario.")
        return float(self.start[0].x), float(self.start[0].y), wrap_angle(math.radians(self.start[1]))

    @property
    def goal_pose_rad(self) -> Tuple[float, float, float]:
        """Return (x, y, yaw_rad) for the goal pose."""
        if self.goal is None:
            raise ValueError("Goal pose has not been set on Scenario.")
        return float(self.goal[0].x), float(self.goal[0].y), wrap_angle(math.radians(self.goal[1]))

    def set_start_goal(
        self,
        start: Tuple[float, float],
        yaw_start: float,
        goal: Tuple[float, float],
        yaw_goal: float,
    ) -> "Scenario":
        """Set start and goal poses. Yaw angles are specified in degrees."""
        if isinstance(start, Point) or isinstance(goal, Point):
            raise TypeError("start and goal must be (x, y) coordinate tuples.")
        self.start = (Point(float(start[0]), float(start[1])), float(yaw_start))
        self.goal = (Point(float(goal[0]), float(goal[1])), float(yaw_goal))
        return self

    def set_obstacles(self, obstacles: Sequence[Polygon]) -> "Scenario":
        """Set static obstacle polygons and precompute hard & soft inflation layers."""
        if not all(isinstance(obs, Polygon) for obs in obstacles):
            raise TypeError("All obstacles must be shapely.geometry.Polygon instances.")
        self.obstacles = list(obstacles)
        self._rebuild_buffers()
        return self

    def set_dynamic_obstacles(self, dynamic_obstacles: Sequence[DynamicObstacle]) -> "Scenario":
        """Set moving obstacles (vehicles, pedestrians, cyclists) in the scenario."""
        self.dynamic_obstacles = list(dynamic_obstacles)
        return self

    def add_dynamic_obstacle(self, obs: DynamicObstacle) -> "Scenario":
        """Append a moving obstacle to the scenario."""
        self.dynamic_obstacles.append(obs)
        return self

    def set_inflation(self, config: InflationConfig) -> "Scenario":
        """Configure obstacle inflation settings and rebuild static inflation layers."""
        self.inflation_config = config
        self.inflater = ObstacleInflater(self.vehicle, self.inflation_config)
        self._rebuild_buffers()
        return self

    def set_vehicle(
        self,
        max_curvature: float,
        length: float = 4.5,
        width: float = 2.0,
    ) -> "Scenario":
        """Configure vehicle dimensions and maximum curvature."""
        self.vehicle_length = float(length)
        self.vehicle_width = float(width)
        rear_overhang = min(0.8, 0.18 * self.vehicle_length)
        front_overhang = min(0.8, 0.18 * self.vehicle_length)
        wheelbase = max(1.5, self.vehicle_length - rear_overhang - front_overhang)
        self.vehicle = VehicleParams(
            wheelbase=wheelbase,
            width=self.vehicle_width,
            rear_overhang=rear_overhang,
            front_overhang=front_overhang,
            wheel_track=0.75 * self.vehicle_width,
            explicit_max_curvature=float(max_curvature),
        )
        self.max_curvature = np.linspace(min(0.1,float(max_curvature)),float(max_curvature),6).tolist()
        self.inflater = ObstacleInflater(self.vehicle, self.inflation_config)
        self._rebuild_buffers()
        return self

    def _rebuild_buffers(self) -> None:
        if not self.obstacles:
            self.buffered_obstacles = []
            self.soft_buffered_obstacles = []
            self._buffered_union = None
            self._soft_union = None
            self._raw_union = None
            return

        hard_polys, soft_polys, hard_u, soft_u = self.inflater.build_static_layers(self.obstacles)
        self.buffered_obstacles = hard_polys
        self.soft_buffered_obstacles = soft_polys
        self._buffered_union = hard_u
        self._soft_union = soft_u
        self._raw_union = unary_union(self.obstacles)

    def collision_free(
        self,
        geom_or_segment: Union[shapely.geometry.base.BaseGeometry, TrajectorySegment],
    ) -> bool:
        """Check whether a geometry or TrajectorySegment is collision-free against inflated static obstacles."""
        if isinstance(geom_or_segment, TrajectorySegment):
            offsets = self.inflater.disk_longitudinal_offsets
            px = geom_or_segment.x[:,None]+np.cos(geom_or_segment.yaw[:,None])*offsets
            py = geom_or_segment.y[:,None]+np.sin(geom_or_segment.yaw[:,None])*offsets
            if self._buffered_union is not None and np.any(shapely.intersects_xy(self._buffered_union,px,py)):
                return False
            if self.boundary is not None:
                points = shapely.points(px,py)
                if np.any(shapely.distance(points,self.boundary) <= self.inflater.disk_radius):
                    return False
            return True

        if not hasattr(geom_or_segment, "geom_type"):
            raise AttributeError("Object is not a Shapely geometry or TrajectorySegment.")

        if self._buffered_union is not None and geom_or_segment.intersects(self._buffered_union):
            return False
        if self.boundary is not None and geom_or_segment.intersects(self.boundary):
            return False
        return True

    def segment_inflation_cost(self, segment: TrajectorySegment) -> float:
        """Compute soft inflation clearance cost integrated along ``segment``."""
        if self._raw_union is None or not self.inflation_config.enabled:
            return 0.0
        if self._soft_union is not None and not segment.linestring.intersects(self._soft_union):
            return 0.0
        step = max(1, len(segment.x) // 6)
        cost = 0.0
        for idx in range(0, len(segment.x), step):
            cost += self.inflater.clearance_cost(
                float(segment.x[idx]),
                float(segment.y[idx]),
                self._raw_union,
            )
        return cost

    def dynamic_collision_free_at(self, x: float, y: float, yaw: float, t: float) -> bool:
        """Check whether the ego vehicle at (x, y, yaw) at time ``t`` avoids all inflated dynamic obstacles."""
        if not self.dynamic_obstacles:
            return True
        disks = self.inflater.pose_disks_world(x, y, yaw)
        r_ego = self.inflater.disk_radius
        for dyn in self.dynamic_obstacles:
            inf_poly = dyn.inflated_polygon_at(t, self.inflation_config, ego_disk_radius=r_ego)
            for dx, dy in disks:
                if inf_poly.contains(Point(float(dx), float(dy))):
                    return False
        return True

    def min_dynamic_clearance_at(self, x: float, y: float, yaw: float, t: float) -> float:
        """Compute minimum surface-to-surface distance [m] between ego vehicle and all dynamic obstacles at ``t``."""
        if not self.dynamic_obstacles:
            return float("inf")
        ego_fp = self.vehicle.footprint_polygon(x, y, yaw, margin=0.0)
        min_d = float("inf")
        for dyn in self.dynamic_obstacles:
            phys_poly = dyn.physical_polygon_at(t)
            d = float(ego_fp.distance(phys_poly))
            if d < min_d:
                min_d = d
        return min_d

    def set_planned_path(self, planned_path: PlannedPath) -> None:
        """Store the structured PlannedPath and synchronize legacy LineString list."""
        self.planned_path = planned_path
        if planned_path.segments:
            self.path = [seg.linestring for seg in reversed(planned_path.segments)]
        elif len(planned_path.x) >= 2:
            self.path = [planned_path.linestring]
        else:
            self.path = []

    def set_path(self, path: Sequence[Union[LineString, TrajectorySegment]]) -> None:
        """Set path from a list of segments or LineStrings (legacy compatibility)."""
        if not path:
            self.path = []
            self.planned_path = None
            return
        if isinstance(path[0], TrajectorySegment):
            segs_rev = list(path)
            segs_fwd = list(reversed(segs_rev))
            self.planned_path = PlannedPath.from_segments(segs_fwd)
            self.path = [s.linestring for s in segs_rev]
        else:
            self.path = [s for s in path]  # type: ignore[misc]
            xs: List[float] = []
            ys: List[float] = []
            for seg in reversed(self.path):
                for cx, cy in seg.coords:
                    if not xs or math.hypot(cx - xs[-1], cy - ys[-1]) > 1e-4:
                        xs.append(float(cx))
                        ys.append(float(cy))
            yaw_start = self.start[1] if self.start is not None else 0.0
            self.planned_path = sanitize_csv_trajectory(
                xs, ys, [yaw_start] * len(xs), [0.0] * len(xs)
            )

    def set_totaltree(self, tree: Sequence[Union[LineString, TrajectorySegment]]) -> None:
        """Store all explored RRT tree edges."""
        self.total_tree = [
            edge.linestring if isinstance(edge, TrajectorySegment) else edge
            for edge in tree
        ]

    def return_path_coords(self) -> List[Tuple[float, float, float, float]]:
        """Return list of (x, y, yaw_deg_0_360, reversing_float) tuples along the path."""
        if self.planned_path is None:
            return []
        coords: List[Tuple[float, float, float, float]] = []
        for x, y, yaw_rad, rev in zip(
            self.planned_path.x,
            self.planned_path.y,
            self.planned_path.yaw,
            self.planned_path.reversing,
        ):
            yaw_deg = (math.degrees(float(yaw_rad)) + 360.0) % 360.0
            coords.append((round(float(x), 4), round(float(y), 4), round(yaw_deg, 4), 1.0 if rev else 0.0))
        return coords

    def return_env_coords(self) -> Tuple[List[List[Tuple[float, float]]], Point, Point]:
        obstacle_coords = [list(obs.exterior.coords) for obs in self.obstacles]
        if self.start is None or self.goal is None:
            raise ValueError("Start and goal must be set before calling return_env_coords.")
        return obstacle_coords, self.start[0], self.goal[0]

    def return_scenariosize(self) -> Tuple[float, float]:
        return (self.width, self.height)

    def write_csv(self, name: str, directory: str | Path = "./Saved_scenarios", overwrite: bool = True) -> Optional[Path]:
        """Write planned trajectory to CSV without interactive blocking prompts."""
        coords = self.return_path_coords()
        if not coords:
            return None
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.name}_Path_{name}"
        if out_path.exists() and not overwrite:
            return out_path
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["", "x_coord", "y_coord", "yaw", "reversing"])
            for idx, (x, y, yaw, rev) in enumerate(coords):
                writer.writerow([idx, x, y, yaw, rev])
        return out_path

    def read_csv(
        self,
        name: str,
        set_path: bool = True,
        directory: str | Path = "./Saved_scenarios",
    ) -> Tuple[List[float], List[float], List[float], List[bool]]:
        """Read a saved trajectory CSV, automatically sanitizing duplicate points and yaw spikes."""
        in_path = Path(directory) / f"{self.name}_Path_{name}"
        if not in_path.exists() and Path(name).exists():
            in_path = Path(name)

        raw_x: List[float] = []
        raw_y: List[float] = []
        raw_yaw: List[float] = []
        raw_rev: List[bool] = []

        with open(in_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_x.append(float(row["x_coord"]))
                raw_y.append(float(row["y_coord"]))
                raw_yaw.append(float(row["yaw"]))
                rev_val = str(row["reversing"]).strip().lower()
                raw_rev.append(rev_val in ("true", "1", "1.0"))

        clean_path = sanitize_csv_trajectory(raw_x, raw_y, raw_yaw, raw_rev, resample_step=0.2)
        if set_path:
            self.set_planned_path(clean_path)

        x_out = clean_path.x.tolist()
        y_out = clean_path.y.tolist()
        yaw_deg_out = [((math.degrees(y) + 360.0) % 360.0) for y in clean_path.yaw]
        rev_out = [bool(r) for r in clean_path.reversing]
        return x_out, y_out, yaw_deg_out, rev_out

    def plot_scenario(
        self,
        plot_all_trees: bool = False,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> None:
        """Render the scenario, obstacles, inflation zones, RRT tree, and planned path."""
        from non_holonomic_vehicle_planning.visualization.plots import plot_planning_result

        plot_planning_result(
            self,
            plot_tree=plot_all_trees,
            output_path=save_path,
            show=show,
        )
