"""Core trajectory and path data structures for non-holonomic planning."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString

from non_holonomic_vehicle_planning.vehicle import wrap_angle


@dataclass
class TrajectorySegment:
    """A single curve connector segment (Dubins, Reeds-Shepp, or straight line) between two RRT nodes."""

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray         # [rad] vehicle heading in [-pi, pi)
    direction: np.ndarray   # +1 for forward, -1 for reverse
    ctypes: List[str] = field(default_factory=list)
    lengths: List[float] = field(default_factory=list)
    total_length: float = 0.0
    _linestring: LineString | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.yaw = np.asarray(self.yaw, dtype=float)
        self.direction = np.asarray(self.direction, dtype=int)
        if self.total_length <= 0.0 and self.x.size >= 2:
            self.total_length = float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))

    @property
    def length(self) -> float:
        return self.total_length

    @property
    def linestring(self) -> LineString:
        if self._linestring is None:
            coords = np.column_stack((self.x, self.y))
            if len(coords) < 2:
                coords = np.vstack((coords, coords))
            self._linestring = LineString(coords)
        return self._linestring

    @property
    def xy(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x, self.y


@dataclass
class PlannedPath:
    """Complete planned trajectory from start to goal with rich kinematic profiles."""

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray          # [rad] wrapped to [-pi, pi)
    direction: np.ndarray    # +1 forward, -1 reverse
    segments: List[TrajectorySegment] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.yaw = np.asarray(self.yaw, dtype=float)
        self.direction = np.asarray(self.direction, dtype=int)

    @property
    def reversing(self) -> np.ndarray:
        """Boolean array where True indicates reverse gear (direction < 0)."""
        return self.direction < 0

    @property
    def length(self) -> float:
        if self.x.size < 2:
            return 0.0
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))

    @property
    def linestring(self) -> LineString:
        coords = np.column_stack((self.x, self.y))
        if len(coords) < 2:
            coords = np.vstack((coords, coords))
        return LineString(coords)

    @classmethod
    def from_segments(cls, segments_start_to_goal: Sequence[TrajectorySegment]) -> "PlannedPath":
        """Concatenate ordered segments (start -> goal) and deduplicate boundary points."""
        if not segments_start_to_goal:
            return cls(
                x=np.array([], dtype=float),
                y=np.array([], dtype=float),
                yaw=np.array([], dtype=float),
                direction=np.array([], dtype=int),
                segments=[],
            )
        xs: List[float] = []
        ys: List[float] = []
        yaws: List[float] = []
        dirs: List[int] = []

        for seg_idx, seg in enumerate(segments_start_to_goal):
            start_k = 0
            if seg_idx > 0 and len(xs) > 0 and len(seg.x) > 0:
                if math.hypot(seg.x[0] - xs[-1], seg.y[0] - ys[-1]) < 1e-4:
                    start_k = 1
            for k in range(start_k, len(seg.x)):
                if len(xs) > 0 and math.hypot(seg.x[k] - xs[-1], seg.y[k] - ys[-1]) < 1e-5:
                    continue
                xs.append(float(seg.x[k]))
                ys.append(float(seg.y[k]))
                yaws.append(wrap_angle(float(seg.yaw[k])))
                dirs.append(int(seg.direction[k]))

        return cls(
            x=np.asarray(xs, dtype=float),
            y=np.asarray(ys, dtype=float),
            yaw=np.asarray(yaws, dtype=float),
            direction=np.asarray(dirs, dtype=int),
            segments=list(segments_start_to_goal),
        )

    def resample_uniform(self, step: float = 0.2) -> "PlannedPath":
        """Resample path so consecutive points within each gear segment have uniform arc-length spacing."""
        if len(self.x) < 2:
            return self

        # Split into contiguous same-direction subpaths at cusps
        cusp_breaks = [0]
        for i in range(1, len(self.direction)):
            if self.direction[i] != self.direction[i - 1]:
                cusp_breaks.append(i)
        cusp_breaks.append(len(self.x))

        new_x: List[float] = []
        new_y: List[float] = []
        new_yaw: List[float] = []
        new_dir: List[int] = []

        for b_idx in range(len(cusp_breaks) - 1):
            i0, i1 = cusp_breaks[b_idx], cusp_breaks[b_idx + 1]
            sx = self.x[i0:i1]
            sy = self.y[i0:i1]
            syaw = np.unwrap(self.yaw[i0:i1])
            sdir = int(self.direction[i0])

            if len(sx) < 2:
                if not new_x or math.hypot(sx[0] - new_x[-1], sy[0] - new_y[-1]) > 1e-5:
                    new_x.append(float(sx[0]))
                    new_y.append(float(sy[0]))
                    new_yaw.append(wrap_angle(float(syaw[0])))
                    new_dir.append(sdir)
                continue

            ds = np.hypot(np.diff(sx), np.diff(sy))
            s_cum = np.concatenate(([0.0], np.cumsum(ds)))
            total_s = float(s_cum[-1])
            if total_s < 1e-5:
                continue

            n_pts = max(2, int(math.ceil(total_s / step)) + 1)
            s_query = np.linspace(0.0, total_s, n_pts)
            rx = np.interp(s_query, s_cum, sx)
            ry = np.interp(s_query, s_cum, sy)
            ryaw = np.interp(s_query, s_cum, syaw)

            start_j = 1 if (new_x and math.hypot(rx[0] - new_x[-1], ry[0] - new_y[-1]) < 1e-4) else 0
            for j in range(start_j, len(rx)):
                new_x.append(float(rx[j]))
                new_y.append(float(ry[j]))
                new_yaw.append(wrap_angle(float(ryaw[j])))
                new_dir.append(sdir)

        return PlannedPath(
            x=np.asarray(new_x, dtype=float),
            y=np.asarray(new_y, dtype=float),
            yaw=np.asarray(new_yaw, dtype=float),
            direction=np.asarray(new_dir, dtype=int),
            segments=self.segments,
        )


def sanitize_csv_trajectory(
    x: Sequence[float],
    y: Sequence[float],
    yaw_deg: Sequence[float],
    reversing: Sequence[float | bool],
    resample_step: float = 0.2,
) -> PlannedPath:
    """Clean and sanitize raw CSV trajectory coordinates.

    Eliminates duplicate zero-distance vertices and bogus yaw spikes (such as 0.0 deg
    or 225.0 deg artifacts from ``arctan2(0, 0)`` at segment boundaries in legacy CSVs),
    recomputes consistent tangent headings aligned with the driving direction, and
    resamples to uniform spacing ``resample_step`` [m].
    """
    raw_x = np.asarray(x, dtype=float)
    raw_y = np.asarray(y, dtype=float)
    raw_rev = np.asarray(reversing, dtype=float) > 0.5

    # 1. Remove consecutive duplicate coordinates while preserving cusp points
    keep_indices = [0]
    for i in range(1, len(raw_x)):
        dist = math.hypot(raw_x[i] - raw_x[keep_indices[-1]], raw_y[i] - raw_y[keep_indices[-1]])
        if dist > 1e-3:
            keep_indices.append(i)
        elif raw_rev[i] != raw_rev[keep_indices[-1]]:
            # Preserve reversing flag update at cusp
            raw_rev[keep_indices[-1]] = raw_rev[i]

    cx = raw_x[keep_indices]
    cy = raw_y[keep_indices]
    crev = raw_rev[keep_indices]
    cdir = np.where(crev, -1, 1)

    n = len(cx)
    if n < 2:
        y0 = math.radians(float(yaw_deg[0])) if len(yaw_deg) else 0.0
        return PlannedPath(cx, cy, np.array([wrap_angle(y0)]), cdir)

    # 2. Compute clean analytical tangent yaw along each continuous direction segment
    cyaw = np.zeros(n, dtype=float)
    for i in range(n):
        if i < n - 1 and (i == 0 or cdir[i] == cdir[i + 1]):
            dx = cx[i + 1] - cx[i]
            dy = cy[i + 1] - cy[i]
            d_sign = cdir[i]
        else:
            dx = cx[i] - cx[i - 1]
            dy = cy[i] - cy[i - 1]
            d_sign = cdir[i - 1]
        tangent = math.atan2(dy, dx)
        if d_sign < 0:
            tangent = wrap_angle(tangent + math.pi)
        cyaw[i] = wrap_angle(tangent)

    path = PlannedPath(x=cx, y=cy, yaw=cyaw, direction=cdir)
    return path.resample_uniform(step=resample_step)

