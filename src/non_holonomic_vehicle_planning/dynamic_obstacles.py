"""Dynamic Moving Obstacles (Moving Vehicles, Pedestrians, Cyclists) and Spatio-Temporal Inflation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from non_holonomic_vehicle_planning.config import InflationConfig
from non_holonomic_vehicle_planning.vehicle import VehicleParams, wrap_angle

ObstacleKind = Literal["vehicle", "pedestrian", "cyclist"]
MotionMode = Literal["constant_velocity", "periodic_ping_pong", "waypoints"]


@dataclass
class DynamicObstacle:
    """Represents a moving object (other vehicle, pedestrian, or cyclist) on the roadway."""

    id: str
    kind: ObstacleKind = "pedestrian"
    start: Tuple[float, float] = (0.0, 0.0)
    yaw_deg: float = 0.0
    speed: float = 1.2                      # [m/s]
    motion_mode: MotionMode = "constant_velocity"
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    length: float = 4.5                     # [m] used when kind == "vehicle"
    width: float = 2.0                      # [m] used when kind == "vehicle"
    radius: float = 0.42                    # [m] used when kind == "pedestrian"
    safety_margin: Optional[float] = None   # [m] optional per-object inflation margin override
    start_delay_s: float = 0.0              # [s] wait time before starting motion
    dwell_time_s: float = 6.5               # [s] sidewalk pause time for periodic_ping_pong pedestrians

    def __post_init__(self) -> None:
        self.start = (float(self.start[0]), float(self.start[1]))
        if self.waypoints:
            self.waypoints = [(float(p[0]), float(p[1])) for p in self.waypoints]

    @property
    def vehicle_params(self) -> VehicleParams:
        """Return 2D planar VehicleParams when this dynamic obstacle is a vehicle."""
        wb = max(1.4, 0.64 * self.length)
        overhang = 0.5 * (self.length - wb)
        return VehicleParams(
            wheelbase=wb,
            pose_reference="center",
            width=self.width,
            rear_overhang=overhang,
            front_overhang=overhang,
            wheel_track=0.75 * self.width,
            tire_radius=0.38,
            tire_width=0.28,
        )

    def pose_at(self, t: float) -> Tuple[float, float, float, float, float]:
        """Compute (x, y, yaw_rad, speed_mps, steer_rad) at simulation time ``t`` [s]."""
        eff_t = max(0.0, float(t) - self.start_delay_s)
        if eff_t <= 0.0 or abs(self.speed) < 1e-6:
            return self.start[0], self.start[1], wrap_angle(math.radians(self.yaw_deg)), 0.0, 0.0

        if self.motion_mode == "periodic_ping_pong" and len(self.waypoints) >= 2:
            p0 = np.asarray(self.waypoints[0], dtype=float)
            p1 = np.asarray(self.waypoints[1], dtype=float)
            seg_vec = p1 - p0
            seg_len = float(np.hypot(seg_vec[0], seg_vec[1]))
            if seg_len < 1e-4:
                return float(p0[0]), float(p0[1]), wrap_angle(math.radians(self.yaw_deg)), 0.0, 0.0
            cross_duration = seg_len / max(abs(self.speed), 1e-4)
            half_period = cross_duration + max(0.0, self.dwell_time_s)
            cycle = int(eff_t // half_period)
            t_in_half = eff_t - cycle * half_period
            if cycle % 2 == 0:
                yaw = math.atan2(seg_vec[1], seg_vec[0])
                if t_in_half >= cross_duration:
                    return float(p1[0]), float(p1[1]), wrap_angle(yaw), 0.0, 0.0
                frac = t_in_half / cross_duration
                pos = p0 + frac * seg_vec
                return float(pos[0]), float(pos[1]), wrap_angle(yaw), abs(self.speed), 0.0
            else:
                yaw = math.atan2(-seg_vec[1], -seg_vec[0])
                if t_in_half >= cross_duration:
                    return float(p0[0]), float(p0[1]), wrap_angle(yaw), 0.0, 0.0
                frac = t_in_half / cross_duration
                pos = p1 - frac * seg_vec
                return float(pos[0]), float(pos[1]), wrap_angle(yaw), abs(self.speed), 0.0

        if self.motion_mode == "waypoints" and len(self.waypoints) >= 2:
            pts = np.asarray(self.waypoints, dtype=float)
            diffs = np.diff(pts, axis=0)
            seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
            s_cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
            total_s = float(s_cum[-1])
            dist = min(abs(self.speed) * eff_t, total_s)
            if dist >= total_s - 1e-4:
                last_yaw = math.atan2(diffs[-1, 1], diffs[-1, 0])
                return float(pts[-1, 0]), float(pts[-1, 1]), wrap_angle(last_yaw), 0.0, 0.0
            idx = max(0, int(np.searchsorted(s_cum, dist, side="right")) - 1)
            idx = min(idx, len(seg_lens) - 1)
            frac = (dist - s_cum[idx]) / max(seg_lens[idx], 1e-6)
            pos = pts[idx] + frac * diffs[idx]
            yaw = math.atan2(diffs[idx, 1], diffs[idx, 0])
            # Smoothly blend heading & front-wheel steering across waypoint corners
            steer = 0.0
            corner_blend_m = 3.0
            if idx + 1 < len(diffs):
                next_yaw = math.atan2(diffs[idx + 1, 1], diffs[idx + 1, 0])
                dyaw = wrap_angle(next_yaw - yaw)
                rem_to_corner = s_cum[idx + 1] - dist
                if rem_to_corner < corner_blend_m:
                    alpha = 0.5 * (1.0 - rem_to_corner / corner_blend_m)
                    yaw = wrap_angle(yaw + alpha * dyaw)
                    steer = float(np.clip(0.55 * dyaw, -0.52, 0.52))
            if idx > 0:
                prev_yaw = math.atan2(diffs[idx - 1, 1], diffs[idx - 1, 0])
                dyaw_prev = wrap_angle(yaw - prev_yaw)
                dist_from_corner = dist - s_cum[idx]
                if dist_from_corner < corner_blend_m:
                    alpha = 0.5 * (1.0 - dist_from_corner / corner_blend_m)
                    yaw = wrap_angle(yaw - alpha * dyaw_prev)
                    steer = float(np.clip(0.55 * dyaw_prev * (1.0 - dist_from_corner / corner_blend_m), -0.52, 0.52))
            return float(pos[0]), float(pos[1]), wrap_angle(yaw), abs(self.speed), steer

        # Default: constant_velocity along yaw_deg
        yaw = wrap_angle(math.radians(self.yaw_deg))
        x = self.start[0] + self.speed * eff_t * math.cos(yaw)
        y = self.start[1] + self.speed * eff_t * math.sin(yaw)
        return x, y, yaw, self.speed, 0.0

    def physical_polygon_at(self, t: float) -> Polygon:
        """Return the uninflated physical 2D footprint of the moving object at time ``t``."""
        x, y, yaw, _, _ = self.pose_at(t)
        if self.kind == "vehicle":
            return self.vehicle_params.footprint_polygon(x, y, yaw, margin=0.0)
        return Point(x, y).buffer(self.radius, quad_segs=12)

    def inflated_polygon_at(
        self,
        t: float,
        inflation_cfg: InflationConfig,
        ego_disk_radius: float = 0.0,
    ) -> Polygon:
        """Return the velocity-anisotropic inflated safety polygon at time ``t`` [s].

        Includes:
        - Object physical geometry
        - Safety margin (``pedestrian_safety_margin_m`` or ``dynamic_vehicle_margin_m``)
        - Forward velocity lookahead capsule ``v * velocity_lookahead_time_s``
        - Ego vehicle disk radius ``ego_disk_radius`` (for point-wise C-space checks)
        """
        x, y, yaw, v, _ = self.pose_at(t)
        if self.safety_margin is not None:
            base_margin = self.safety_margin
        elif self.kind == "pedestrian":
            base_margin = inflation_cfg.pedestrian_safety_margin_m
        else:
            base_margin = inflation_cfg.dynamic_vehicle_margin_m

        total_margin = (base_margin if inflation_cfg.enabled else 0.1) + ego_disk_radius
        fwd_lookahead = abs(v) * (inflation_cfg.velocity_lookahead_time_s if inflation_cfg.enabled else 0.0)

        if self.kind == "vehicle":
            vp = self.vehicle_params
            x_rear = vp.rear_bumper_x - total_margin
            x_front = vp.front_bumper_x + total_margin + fwd_lookahead
            y_half = vp.half_width + total_margin
            local_pts = np.array([
                [x_rear, -y_half],
                [x_front, -y_half],
                [x_front, y_half],
                [x_rear, y_half],
            ], dtype=float)
            c, s = math.cos(yaw), math.sin(yaw)
            rot = np.array([[c, s], [-s, c]], dtype=float)
            world_pts = local_pts @ rot + np.array([x, y], dtype=float)
            return Polygon(world_pts).buffer(0.15, quad_segs=8)

        # Pedestrian / Cyclist: capsule stretched in travel direction
        x_end = x + fwd_lookahead * math.cos(yaw)
        y_end = y + fwd_lookahead * math.sin(yaw)
        spine = LineString([(x, y), (x_end, y_end)]) if fwd_lookahead > 1e-3 else Point(x, y)
        return spine.buffer(self.radius + total_margin, quad_segs=12)

    def predict_trajectory(self, t0: float, dt: float, steps: int) -> np.ndarray:
        """Return (steps+1, 5) array of predicted [x, y, yaw, v, steer] over [t0, t0 + steps*dt]."""
        out = np.zeros((steps + 1, 5), dtype=float)
        for k in range(steps + 1):
            out[k, :] = self.pose_at(t0 + k * dt)
        return out

    def to_dict(self) -> Dict[str, object]:
        """Serialize DynamicObstacle to a YAML-compatible dictionary."""
        d: Dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "start": [self.start[0], self.start[1]],
            "yaw_deg": self.yaw_deg,
            "speed": self.speed,
            "motion_mode": self.motion_mode,
        }
        if self.waypoints:
            d["waypoints"] = [[p[0], p[1]] for p in self.waypoints]
        if self.kind == "vehicle":
            d["length"] = self.length
            d["width"] = self.width
        else:
            d["radius"] = self.radius
        if self.safety_margin is not None:
            d["safety_margin"] = self.safety_margin
        if self.start_delay_s > 0.0:
            d["start_delay_s"] = self.start_delay_s
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DynamicObstacle":
        """Construct a DynamicObstacle from a YAML dictionary."""
        start_raw = data.get("start", [0.0, 0.0])
        start_xy = (float(start_raw[0]), float(start_raw[1]))  # type: ignore[index]
        wps_raw = data.get("waypoints", [])
        wps = [(float(p[0]), float(p[1])) for p in wps_raw]  # type: ignore[attr-defined]
        return cls(
            id=str(data.get("id", "obs")),
            kind=str(data.get("kind", "pedestrian")),  # type: ignore[arg-type]
            start=start_xy,
            yaw_deg=float(data.get("yaw_deg", 0.0)),  # type: ignore[arg-type]
            speed=float(data.get("speed", 1.2)),  # type: ignore[arg-type]
            motion_mode=str(data.get("motion_mode", "constant_velocity")),  # type: ignore[arg-type]
            waypoints=wps,
            length=float(data.get("length", 4.5)),  # type: ignore[arg-type]
            width=float(data.get("width", 2.0)),  # type: ignore[arg-type]
            radius=float(data.get("radius", 0.42)),  # type: ignore[arg-type]
            safety_margin=float(data["safety_margin"]) if "safety_margin" in data else None,  # type: ignore[arg-type]
            start_delay_s=float(data.get("start_delay_s", 0.0)),  # type: ignore[arg-type]
        )

