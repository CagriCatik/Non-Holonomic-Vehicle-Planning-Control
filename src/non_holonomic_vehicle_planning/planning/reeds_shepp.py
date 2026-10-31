"""Reeds-Shepp curve path planner for forward & reverse non-holonomic vehicles."""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np

from non_holonomic_vehicle_planning.planning.base import TrajectorySegment
from non_holonomic_vehicle_planning.vehicle import wrap_angle


class _RSPathCandidate:
    __slots__ = ("lengths", "ctypes", "L")

    def __init__(self, lengths: List[float], ctypes: List[str]) -> None:
        self.lengths = lengths
        self.ctypes = ctypes
        self.L = float(sum(abs(x) for x in lengths))


def _mod2pi(x: float) -> float:
    v = math.fmod(x, math.copysign(2.0 * math.pi, x))
    if v < -math.pi:
        v += 2.0 * math.pi
    elif v > math.pi:
        v -= 2.0 * math.pi
    return v


def _polar(x: float, y: float) -> Tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


def _add_path(
    paths: List[_RSPathCandidate],
    lengths: List[float],
    ctypes: List[str],
    step_size: float,
) -> List[_RSPathCandidate]:
    cand = _RSPathCandidate(lengths, ctypes)
    for existing in paths:
        if existing.ctypes == cand.ctypes and abs(existing.L - cand.L) <= step_size:
            return paths
    if cand.L <= step_size:
        return paths
    paths.append(cand)
    return paths


def _straight_left_straight(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    phi = _mod2pi(phi)
    if y > 0.0 and 0.0 < phi < math.pi * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = math.hypot(x - xd, y) - math.tan(phi / 2.0)
        return True, t, u, v
    if y < 0.0 < phi < math.pi * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = -math.hypot(x - xd, y) - math.tan(phi / 2.0)
        return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _left_straight_left(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= 0.0:
        v = _mod2pi(phi - t)
        if v >= 0.0:
            return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _left_right_left(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    u1, t1 = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = _mod2pi(t1 + 0.5 * u + math.pi)
        v = _mod2pi(phi - t + u)
        if t >= 0.0 >= u:
            return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _left_straight_right(x: float, y: float, phi: float) -> Tuple[bool, float, float, float]:
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1_sq = u1 * u1
    if u1_sq >= 4.0:
        u = math.sqrt(u1_sq - 4.0)
        theta = math.atan2(2.0, u)
        t = _mod2pi(t1 + theta)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return True, t, u, v
    return False, 0.0, 0.0, 0.0


def _scs(x: float, y: float, phi: float, paths: List[_RSPathCandidate], step: float) -> List[_RSPathCandidate]:
    flag, t, u, v = _straight_left_straight(x, y, phi)
    if flag:
        paths = _add_path(paths, [t, u, v], ["S", "L", "S"], step)
    flag, t, u, v = _straight_left_straight(x, -y, -phi)
    if flag:
        paths = _add_path(paths, [t, u, v], ["S", "R", "S"], step)
    return paths


def _csc(x: float, y: float, phi: float, paths: List[_RSPathCandidate], step: float) -> List[_RSPathCandidate]:
    flag, t, u, v = _left_straight_left(x, y, phi)
    if flag:
        paths = _add_path(paths, [t, u, v], ["L", "S", "L"], step)
    flag, t, u, v = _left_straight_left(-x, y, -phi)
    if flag:
        paths = _add_path(paths, [-t, -u, -v], ["L", "S", "L"], step)
    flag, t, u, v = _left_straight_left(x, -y, -phi)
    if flag:
        paths = _add_path(paths, [t, u, v], ["R", "S", "R"], step)
    flag, t, u, v = _left_straight_left(-x, -y, phi)
    if flag:
        paths = _add_path(paths, [-t, -u, -v], ["R", "S", "R"], step)

    flag, t, u, v = _left_straight_right(x, y, phi)
    if flag:
        paths = _add_path(paths, [t, u, v], ["L", "S", "R"], step)
    flag, t, u, v = _left_straight_right(-x, y, -phi)
    if flag:
        paths = _add_path(paths, [-t, -u, -v], ["L", "S", "R"], step)
    flag, t, u, v = _left_straight_right(x, -y, -phi)
    if flag:
        paths = _add_path(paths, [t, u, v], ["R", "S", "L"], step)
    flag, t, u, v = _left_straight_right(-x, -y, phi)
    if flag:
        paths = _add_path(paths, [-t, -u, -v], ["R", "S", "L"], step)
    return paths


def _ccc(x: float, y: float, phi: float, paths: List[_RSPathCandidate], step: float) -> List[_RSPathCandidate]:
    flag, t, u, v = _left_right_left(x, y, phi)
    if flag:
        paths = _add_path(paths, [t, u, v], ["L", "R", "L"], step)
    flag, t, u, v = _left_right_left(-x, y, -phi)
    if flag:
        paths = _add_path(paths, [-t, -u, -v], ["L", "R", "L"], step)
    flag, t, u, v = _left_right_left(x, -y, -phi)
    if flag:
        paths = _add_path(paths, [t, u, v], ["R", "L", "R"], step)
    flag, t, u, v = _left_right_left(-x, -y, phi)
    if flag:
        paths = _add_path(paths, [-t, -u, -v], ["R", "L", "R"], step)

    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)

    flag, t, u, v = _left_right_left(xb, yb, phi)
    if flag:
        paths = _add_path(paths, [v, u, t], ["L", "R", "L"], step)
    flag, t, u, v = _left_right_left(-xb, yb, -phi)
    if flag:
        paths = _add_path(paths, [-v, -u, -t], ["L", "R", "L"], step)
    flag, t, u, v = _left_right_left(xb, -yb, -phi)
    if flag:
        paths = _add_path(paths, [v, u, t], ["R", "L", "R"], step)
    flag, t, u, v = _left_right_left(-xb, -yb, phi)
    if flag:
        paths = _add_path(paths, [-v, -u, -t], ["R", "L", "R"], step)
    return paths


def _interpolate_rs(
    dist: float,
    length: float,
    mode: str,
    max_curvature: float,
    origin_x: float,
    origin_y: float,
    origin_yaw: float,
) -> Tuple[float, float, float, int]:
    if mode == "S":
        x = origin_x + dist / max_curvature * math.cos(origin_yaw)
        y = origin_y + dist / max_curvature * math.sin(origin_yaw)
        yaw = origin_yaw
    else:
        ldx = math.sin(dist) / max_curvature
        if mode == "L":
            ldy = (1.0 - math.cos(dist)) / max_curvature
            yaw = origin_yaw + dist
        else:
            ldy = (1.0 - math.cos(dist)) / (-max_curvature)
            yaw = origin_yaw - dist
        c, s = math.cos(-origin_yaw), math.sin(-origin_yaw)
        gdx = c * ldx + s * ldy
        gdy = -s * ldx + c * ldy
        x = origin_x + gdx
        y = origin_y + gdy

    return x, y, yaw, (1 if length >= 0.0 else -1)


def _generate_rs_local_course(
    lengths: List[float],
    modes: List[str],
    max_curvature: float,
    step_norm: float,
) -> Tuple[List[float], List[float], List[float], List[int]]:
    origin_x, origin_y, origin_yaw = 0.0, 0.0, 0.0
    xs: List[float] = []
    ys: List[float] = []
    yaws: List[float] = []
    directions: List[int] = []

    for seg_idx, (length, mode) in enumerate(zip(lengths, modes)):
        if abs(length) <= 1e-9:
            continue
        d_dist = step_norm if length >= 0.0 else -step_norm
        start_d = 0.0 if seg_idx == 0 else d_dist
        if (length >= 0.0 and start_d < length) or (length < 0.0 and start_d > length):
            interp_dists = np.arange(start_d, length, d_dist)
            interp_dists = np.append(interp_dists, length)
        else:
            interp_dists = np.array([length], dtype=float)

        for dist in interp_dists:
            x, y, yaw, direction = _interpolate_rs(
                float(dist), length, mode, max_curvature, origin_x, origin_y, origin_yaw
            )
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            directions.append(direction)

        origin_x = xs[-1]
        origin_y = ys[-1]
        origin_yaw = yaws[-1]

    return xs, ys, yaws, directions


def plan_reeds_shepp_segments(
    sx: float,
    sy: float,
    syaw: float,
    gx: float,
    gy: float,
    gyaw: float,
    curvatures: float | Iterable[float],
    step: float = 0.2,
) -> List[TrajectorySegment]:
    """Compute all valid Reeds-Shepp curve segments between two poses, sorted by path length."""
    if isinstance(curvatures, (int, float)):
        curv_list = [float(curvatures)]
    else:
        curv_list = [float(c) for c in curvatures]

    dx = gx - sx
    dy = gy - sy
    dth = gyaw - syaw
    c0, s0 = math.cos(syaw), math.sin(syaw)
    lx_unit = c0 * dx + s0 * dy
    ly_unit = -s0 * dx + c0 * dy

    segments: List[TrajectorySegment] = []

    for maxc in curv_list:
        x = lx_unit * maxc
        y = ly_unit * maxc
        step_norm = max(step * maxc, 1e-3)

        cands: List[_RSPathCandidate] = []
        cands = _scs(x, y, dth, cands, step_norm)
        cands = _csc(x, y, dth, cands, step_norm)
        cands = _ccc(x, y, dth, cands, step_norm)

        for cand in cands:
            xs, ys, yaws, directions = _generate_rs_local_course(
                cand.lengths, cand.ctypes, maxc, step_norm
            )
            if len(xs) < 2:
                continue
            xs_arr = np.asarray(xs, dtype=float)
            ys_arr = np.asarray(ys, dtype=float)
            gx_arr = c0 * xs_arr - s0 * ys_arr + sx
            gy_arr = s0 * xs_arr + c0 * ys_arr + sy
            gyaw_arr = np.array([wrap_angle(yaw + syaw) for yaw in yaws], dtype=float)
            dirs_arr = np.asarray(directions, dtype=int)

            segments.append(
                TrajectorySegment(
                    x=gx_arr,
                    y=gy_arr,
                    yaw=gyaw_arr,
                    direction=dirs_arr,
                    ctypes=cand.ctypes,
                    lengths=[l / maxc for l in cand.lengths],
                    total_length=cand.L / maxc,
                )
            )

    segments.sort(key=lambda s: s.length)
    return segments


def reeds_shepp_path_planning(
    sx: float,
    sy: float,
    syaw: float,
    gx: float,
    gy: float,
    gyaw: float,
    maxc: float | Iterable[float],
    step: float = 0.2,
) -> List[TrajectorySegment]:
    """Alias for ``plan_reeds_shepp_segments``."""
    return plan_reeds_shepp_segments(sx, sy, syaw, gx, gy, gyaw, maxc, step=step)

