"""Obstacle Inflation Technique (Minkowski Safety Buffers, Multi-Disk Footprint & Cost Fields).

Implements two-tier obstacle inflation for collision-free non-holonomic planning:
1. **Multi-Disk Vehicle Footprint Decomposition**:
   Decomposes the 2D planar vehicle chassis (L x W) into ``N`` overlapping circles along
   the vehicle's longitudinal axis with radius ``r_disk = hypot(W/2, L/(2*N))``.
   Compared to a coarse single bounding circle of radius ``hypot(L/2, W/2)`` (~2.5 m),
   the 3-disk decomposition reduces the effective inflation radius to ~1.25 m, enabling
   safe passage through tight corridors and parking bays without sacrificing exactness.
2. **Hard Collision Inflation & Soft Safety Halo**:
   - **Hard Inflation Zone** (``r_hard = r_disk + hard_margin_m``): strictly forbidden
     Minkowski-sum boundary for the disk centers.
   - **Soft Cost Inflation Zone** (``r_soft = r_hard + soft_margin_m``): repulsive safety
     potential field that biases RRT*, local real-time bypass trajectories, and MPC
     away from obstacle edges.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import shapely.geometry
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from non_holonomic_vehicle_planning.config import InflationConfig
from non_holonomic_vehicle_planning.vehicle import VehicleParams


class ObstacleInflater:
    """Computes hard and soft inflated obstacle layers and multi-disk collision checks."""

    def __init__(
        self,
        vehicle: VehicleParams,
        config: InflationConfig | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.config = config or InflationConfig()

    @property
    def disk_radius(self) -> float:
        """Radius [m] of each covering circle in the multi-disk vehicle decomposition."""
        if not self.config.use_multi_disk_footprint or self.config.num_footprint_disks <= 1:
            return self.vehicle.bounding_radius
        n = max(1, self.config.num_footprint_disks)
        half_seg_len = self.vehicle.length / (2.0 * n)
        return math.hypot(self.vehicle.half_width, half_seg_len)

    @property
    def disk_longitudinal_offsets(self) -> np.ndarray:
        """Longitudinal offsets [m] of the covering disk centers in the vehicle-center frame."""
        if not self.config.use_multi_disk_footprint or self.config.num_footprint_disks <= 1:
            return np.array([0.0], dtype=float)
        n = self.config.num_footprint_disks
        half_L = 0.5 * self.vehicle.length
        step = self.vehicle.length / n
        return np.linspace(-half_L + 0.5 * step, half_L - 0.5 * step, n)

    @property
    def hard_inflation_radius(self) -> float:
        """Total hard inflation buffer radius [m] around raw static obstacles."""
        if not self.config.enabled:
            return self.disk_radius
        return self.disk_radius + self.config.hard_margin_m

    @property
    def soft_inflation_radius(self) -> float:
        """Total soft safety halo radius [m] around raw static obstacles."""
        if not self.config.enabled:
            return self.hard_inflation_radius
        return self.hard_inflation_radius + self.config.soft_margin_m

    def build_static_layers(
        self,
        obstacles: Sequence[Polygon],
    ) -> Tuple[List[Polygon], List[Polygon], shapely.geometry.base.BaseGeometry | None, shapely.geometry.base.BaseGeometry | None]:
        """Compute hard-inflated and soft-inflated obstacle polygon lists and their unions."""
        if not obstacles:
            return [], [], None, None

        r_hard = self.hard_inflation_radius
        r_soft = self.soft_inflation_radius

        hard_polys = [obs.buffer(r_hard, cap_style=1, join_style=1) for obs in obstacles]
        soft_polys = [obs.buffer(r_soft, cap_style=1, join_style=1) for obs in obstacles]

        hard_union = unary_union(hard_polys)
        soft_union = unary_union(soft_polys)
        return hard_polys, soft_polys, hard_union, soft_union

    def pose_disks_world(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Return (N, 2) world coordinates of the vehicle's covering disk centers."""
        offsets = self.disk_longitudinal_offsets
        c, s = math.cos(yaw), math.sin(yaw)
        wx = x + offsets * c
        wy = y + offsets * s
        return np.column_stack((wx, wy))

    def is_pose_collision_free(
        self,
        x: float,
        y: float,
        yaw: float,
        hard_union: shapely.geometry.base.BaseGeometry | None,
        boundary: LineString | None = None,
    ) -> bool:
        """Check if all covering disks of the 2D planar vehicle at (x, y, yaw) are collision-free."""
        disks = self.pose_disks_world(x, y, yaw)
        if hard_union is not None:
            for dx, dy in disks:
                if hard_union.contains(Point(float(dx), float(dy))):
                    return False
        if boundary is not None:
            r = self.disk_radius
            for dx, dy in disks:
                if Point(float(dx), float(dy)).distance(boundary) < r:
                    return False
        return True

    def clearance_cost(
        self,
        x: float,
        y: float,
        raw_union: shapely.geometry.base.BaseGeometry | None,
    ) -> float:
        """Compute soft inflation repulsion cost >= 0 at world position (x, y)."""
        if raw_union is None or not self.config.enabled or self.config.soft_margin_m <= 1e-6:
            return 0.0
        dist = float(raw_union.distance(Point(x, y)))
        r_hard = self.hard_inflation_radius
        r_soft = self.soft_inflation_radius
        if dist >= r_soft:
            return 0.0
        if dist <= r_hard:
            return self.config.soft_cost_weight * 5.0
        norm_penetration = (r_soft - dist) / (r_soft - r_hard)
        return float(self.config.soft_cost_weight * (norm_penetration ** 2))

