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

    Coordinate convention:
    The state reference point ``(x, y)`` is the **geometric center** of the
    vehicle chassis, with ``+x`` pointing forward along the longitudinal axis
    at ``yaw = 0`` and ``+y`` pointing left:
    - Rear bumper is at ``x_local = -0.5 * length``
    - Rear axle is at ``x_local = -0.5 * wheelbase``
    - Front axle is at ``x_local = +0.5 * wheelbase``
    - Front bumper is at ``x_local = +0.5 * length``
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
        """Longitudinal position of rear axle in vehicle-center frame [m]."""
        return -0.5 * self.wheelbase

    @property
    def front_axle_x(self) -> float:
        """Longitudinal position of front axle in vehicle-center frame [m]."""
        return 0.5 * self.wheelbase

    @property
    def rear_bumper_x(self) -> float:
        """Longitudinal position of rear bumper in vehicle-center frame [m]."""
        return self.rear_axle_x - self.rear_overhang

    @property
    def front_bumper_x(self) -> float:
        """Longitudinal position of front bumper in vehicle-center frame [m]."""
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
        return np.linspace(max(0.1, 0.35 * c_max), c_max, 5).tolist()

    @property
    def min_turning_radius(self) -> float:
        """Minimum turning radius [m]."""
        return 1.0 / self.max_curvature

    @property
    def bounding_radius(self) -> float:
        """Conservative bounding circle radius around geometric center [m]."""
        return math.hypot(0.5 * self.length, 0.5 * self.width)

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
