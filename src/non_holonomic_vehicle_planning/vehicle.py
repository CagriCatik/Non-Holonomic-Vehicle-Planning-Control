"""2D Planar Vehicle geometry and kinematic parameters.

Provides a unified vehicle model used across path planning (Dubins & Reeds-Shepp
curvature bounds, collision footprints), MPC trajectory tracking (kinematic
bicycle model), and 2D planar visualization (chassis body, axles, four tires,
front steering angle, and driving direction vectors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Tuple

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon


def wrap_angle(angle: float) -> float:
    """Wrap angle in radians to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def unwrap_angles(angles: np.ndarray | list[float]) -> np.ndarray:
    """Unwrap a sequence of angles [rad] so consecutive jumps never exceed pi."""
    arr = np.asarray(angles, dtype=float)
    if arr.size == 0:
        return arr
    return np.unwrap(arr)


@dataclass
class VehicleParams:
    """Unified 2D planar vehicle parameters.

    Pose convention: ego planning/control poses are at the rear axle center.
    Signed speed is rear-axle longitudinal speed. Geometry can explicitly use
    ``pose_reference="center"`` for externally specified moving-obstacle poses.
    """

    wheelbase: float = 2.9          # [m] distance between rear and front axles (WB)
    width: float = 2.0              # [m] overall chassis width (W)
    rear_overhang: float = 0.8      # [m] rear axle to rear bumper
    front_overhang: float = 0.8     # [m] front axle to front bumper
    wheel_track: float = 1.50       # [m] lateral distance between left and right wheel centers
    tire_radius: float = 0.40       # [m] tire outer radius (half of tire rolling length)
    tire_width: float = 0.30        # [m] tire lateral width

    max_steer: float = field(default_factory=lambda: math.radians(45.0))         # [rad] max front wheel steer angle
    max_steer_rate: float = field(default_factory=lambda: math.radians(50.0))    # [rad/s] max physical steering rate
    max_steer_accel: float = field(default_factory=lambda: math.radians(180.0))  # [rad/s^2] max steering angular accel
    steer_tau: float = 0.16         # [s] first-order steering servo time constant (PT1 Aktuatordynamik)
    accel_tau: float = 0.10         # [s] first-order drive/brake actuator time constant
    ackermann_steering: bool = True # Enable true Ackermann inner/outer front wheel angles (delta_FL != delta_FR)
    max_speed: float = 55.0 / 3.6   # [m/s] max forward speed
    min_speed: float = -20.0 / 3.6  # [m/s] max reverse speed
    max_accel: float = 1.5          # [m/s^2] max longitudinal acceleration

    # Optional explicit max curvature override [1/m] for path planners
    explicit_max_curvature: float | None = 1.0 / 4.39
    pose_reference: str = "rear_axle"

    def __post_init__(self):
        for name in ("wheelbase","width","wheel_track","tire_radius","tire_width",
                     "max_steer_rate","max_steer_accel","max_accel","steer_tau","accel_tau","max_speed"):
            value = getattr(self,name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name+" must be finite and positive")
        if not 0 < self.max_steer < math.pi/2:
            raise ValueError("max_steer must lie in (0,pi/2)")
        if not math.isfinite(self.min_speed) or self.min_speed>0:
            raise ValueError("min_speed must be finite and <=0")
        if any(not math.isfinite(v) or v<0 for v in (self.rear_overhang,self.front_overhang)):
            raise ValueError("Overhangs must be finite and nonnegative")
        if self.pose_reference not in ("rear_axle","center"):
            raise ValueError("pose_reference must be rear_axle or center")
        if self.explicit_max_curvature is not None:
            if not math.isfinite(self.explicit_max_curvature) or self.explicit_max_curvature <= 0:
                raise ValueError("Explicit curvature must be finite and positive")
            if self.explicit_max_curvature > math.tan(self.max_steer)/self.wheelbase+1e-9:
                raise ValueError("Planner curvature exceeds physical steering limit")

    def ackermann_wheel_angles(self, steer: float) -> Tuple[float, float]:
        """Compute (delta_FL, delta_FR) [rad] using exact Ackermann steering geometry.

        For a bicycle-model reference steering angle ``steer`` (positive = left turn):
        - Inner wheel steers sharper than ``steer``
        - Outer wheel steers gentler than ``steer``
        so that all four wheel axes intersect at the instantaneous center of rotation.
        """
        delta = float(np.clip(steer, -self.max_steer, self.max_steer))
        if not self.ackermann_steering or abs(delta) < 1e-5:
            return delta, delta
        tan_d = math.tan(delta)
        wb = self.wheelbase
        ht = self.half_track
        denom_fl = max(wb - ht * tan_d, 0.25 * wb)
        denom_fr = max(wb + ht * tan_d, 0.25 * wb)
        delta_fl = math.atan2(wb * tan_d, denom_fl) if delta >= 0.0 else -math.atan2(wb * abs(tan_d), wb + ht * abs(tan_d))
        delta_fr = math.atan2(wb * tan_d, denom_fr) if delta >= 0.0 else -math.atan2(wb * abs(tan_d), max(wb - ht * abs(tan_d), 0.25 * wb))
        return float(delta_fl), float(delta_fr)

    @property
    def length(self) -> float:
        """Total bumper-to-bumper vehicle length [m]."""
        return self.rear_overhang + self.wheelbase + self.front_overhang

    @property
    def rear_axle_x(self) -> float:
        """Longitudinal position of rear axle in pose frame [m]."""
        return -0.5 * self.wheelbase if self.pose_reference == "center" else 0.

    @property
    def front_axle_x(self) -> float:
        """Longitudinal position of front axle in pose frame [m]."""
        return self.rear_axle_x + self.wheelbase

    @property
    def rear_bumper_x(self) -> float:
        """Longitudinal position of rear bumper in pose frame [m]."""
        return self.rear_axle_x - self.rear_overhang

    @property
    def front_bumper_x(self) -> float:
        """Longitudinal position of front bumper in pose frame [m]."""
        return self.front_axle_x + self.front_overhang

    @property
    def half_track(self) -> float:
        """Half of the wheel track [m]."""
        return 0.5 * self.wheel_track

    @property
    def half_width(self) -> float:
        """Half of the vehicle body width [m]."""
        return 0.5 * self.width

    @property
    def wheel_len(self) -> float:
        """Visual/physical tire length along rolling direction [m]."""
        return 2.0 * self.tire_radius

    @property
    def wheel_w(self) -> float:
        """Visual/physical tire width [m]."""
        return self.tire_width

    @property
    def max_curvature(self) -> float:
        """Maximum path curvature [1/m]."""
        if self.explicit_max_curvature is not None and self.explicit_max_curvature > 0.0:
            return float(self.explicit_max_curvature)
        return math.tan(self.max_steer) / self.wheelbase

    @property
    def curvature_candidates(self) -> list[float]:
        """Set of candidate curvatures [1/m] used by RRT* curve connectors."""
        c_max = self.max_curvature
        return np.linspace(min(c_max,max(0.1,0.35*c_max)),c_max,5).tolist()

    @property
    def min_turning_radius(self) -> float:
        """Minimum turning radius [m]."""
        return 1.0 / self.max_curvature

    @property
    def bounding_radius(self) -> float:
        """Conservative bounding circle radius around geometric center [m]."""
        return math.hypot(max(abs(self.rear_bumper_x),abs(self.front_bumper_x)),self.half_width)

    def footprint_polygon(self, x: float, y: float, yaw: float, margin: float = 0.0) -> ShapelyPolygon:
        """Return the oriented 2D chassis footprint as a Shapely Polygon in world coordinates."""
        x_rear = self.rear_bumper_x - margin
        x_front = self.front_bumper_x + margin
        y_half = self.half_width + margin
        local_pts = np.array([
            [x_rear, -y_half],
            [x_front, -y_half],
            [x_front, y_half],
            [x_rear, y_half],
        ], dtype=float)
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, s], [-s, c]], dtype=float)
        world_pts = local_pts @ rot + np.array([x, y], dtype=float)
        return ShapelyPolygon(world_pts)
