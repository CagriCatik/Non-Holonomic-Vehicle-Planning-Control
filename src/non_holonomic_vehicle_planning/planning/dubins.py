"""Dubins curve path planner for forward-only non-holonomic vehicles."""

from __future__ import annotations

import math
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np
from non_holonomic_vehicle_planning.trajectory import sample_primitive

from non_holonomic_vehicle_planning.planning.base import TrajectorySegment
from non_holonomic_vehicle_planning.vehicle import wrap_angle


def _mod2pi(theta: float) -> float:
    return theta - 2.0 * math.pi * math.floor(theta / (2.0 * math.pi))


def _lsl(alpha: float, beta: float, dist: float) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    sin_a, sin_b = math.sin(alpha), math.sin(beta)
    cos_a, cos_b = math.cos(alpha), math.cos(beta)
    cos_ab = math.cos(alpha - beta)

    p_sq = 2.0 + dist * dist - 2.0 * cos_ab + 2.0 * dist * (sin_a - sin_b)
    if p_sq < 0.0:
        return None, None, None, ["L", "S", "L"]
    p = math.sqrt(p_sq)
    denom = dist + sin_a - sin_b
    t = _mod2pi(-alpha + math.atan2(cos_b - cos_a, denom))
    q = _mod2pi(beta - math.atan2(cos_b - cos_a, denom))
    return t, p, q, ["L", "S", "L"]


def _rsr(alpha: float, beta: float, dist: float) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    sin_a, sin_b = math.sin(alpha), math.sin(beta)
    cos_a, cos_b = math.cos(alpha), math.cos(beta)
    cos_ab = math.cos(alpha - beta)

    p_sq = 2.0 + dist * dist - 2.0 * cos_ab + 2.0 * dist * (sin_b - sin_a)
    if p_sq < 0.0:
        return None, None, None, ["R", "S", "R"]
    p = math.sqrt(p_sq)
    denom = dist - sin_a + sin_b
    t = _mod2pi(alpha - math.atan2(cos_a - cos_b, denom))
    q = _mod2pi(-beta + math.atan2(cos_a - cos_b, denom))
    return t, p, q, ["R", "S", "R"]


def _lsr(alpha: float, beta: float, dist: float) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    sin_a, sin_b = math.sin(alpha), math.sin(beta)
    cos_a, cos_b = math.cos(alpha), math.cos(beta)
    cos_ab = math.cos(alpha - beta)

    p_sq = -2.0 + dist * dist + 2.0 * cos_ab + 2.0 * dist * (sin_a + sin_b)
    if p_sq < 0.0:
        return None, None, None, ["L", "S", "R"]
    p = math.sqrt(p_sq)
    rec = math.atan2(-cos_a - cos_b, dist + sin_a + sin_b) - math.atan2(-2.0, p)
    t = _mod2pi(-alpha + rec)
    q = _mod2pi(-_mod2pi(beta) + rec)
    return t, p, q, ["L", "S", "R"]


def _rsl(alpha: float, beta: float, dist: float) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    sin_a, sin_b = math.sin(alpha), math.sin(beta)
    cos_a, cos_b = math.cos(alpha), math.cos(beta)
    cos_ab = math.cos(alpha - beta)

    p_sq = -2.0 + dist * dist + 2.0 * cos_ab - 2.0 * dist * (sin_a + sin_b)
    if p_sq < 0.0:
        return None, None, None, ["R", "S", "L"]
    p = math.sqrt(p_sq)
    rec = math.atan2(cos_a + cos_b, dist - sin_a - sin_b) - math.atan2(2.0, p)
    t = _mod2pi(alpha - rec)
    q = _mod2pi(beta - rec)
    return t, p, q, ["R", "S", "L"]


def _rlr(alpha: float, beta: float, dist: float) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    sin_a, sin_b = math.sin(alpha), math.sin(beta)
    cos_a, cos_b = math.cos(alpha), math.cos(beta)
    cos_ab = math.cos(alpha - beta)

    rec = (6.0 - dist * dist + 2.0 * cos_ab + 2.0 * dist * (sin_a - sin_b)) / 8.0
    if abs(rec) > 1.0:
        return None, None, None, ["R", "L", "R"]
    p = _mod2pi(2.0 * math.pi - math.acos(rec))
    t = _mod2pi(alpha - math.atan2(cos_a - cos_b, dist - sin_a + sin_b) + _mod2pi(p / 2.0))
    q = _mod2pi(alpha - beta - t + _mod2pi(p))
    return t, p, q, ["R", "L", "R"]


def _lrl(alpha: float, beta: float, dist: float) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    sin_a, sin_b = math.sin(alpha), math.sin(beta)
    cos_a, cos_b = math.cos(alpha), math.cos(beta)
    cos_ab = math.cos(alpha - beta)

    rec = (6.0 - dist * dist + 2.0 * cos_ab + 2.0 * dist * (sin_b - sin_a)) / 8.0
    if abs(rec) > 1.0:
        return None, None, None, ["L", "R", "L"]
    p = _mod2pi(2.0 * math.pi - math.acos(rec))
    t = _mod2pi(-alpha - math.atan2(cos_a - cos_b, dist + sin_a - sin_b) + p / 2.0)
    q = _mod2pi(_mod2pi(beta) - alpha - t + _mod2pi(p))
    return t, p, q, ["L", "R", "L"]


_DUBINS_PRIMITIVES = (_lsl, _rsr, _lsr, _rsl, _rlr, _lrl)


def _interpolate_dubins_segment(
    dist: float,
    mode: str,
    curvature: float,
    ox: float,
    oy: float,
    oyaw: float,
) -> Tuple[float, float, float]:
    if mode == "S":
        x = ox + dist / curvature * math.cos(oyaw)
        y = oy + dist / curvature * math.sin(oyaw)
        yaw = oyaw
    else:
        ldx = math.sin(dist) / curvature
        if mode == "L":
            ldy = (1.0 - math.cos(dist)) / curvature
            yaw = oyaw + dist
        else:
            ldy = (1.0 - math.cos(dist)) / (-curvature)
            yaw = oyaw - dist
        c, s = math.cos(-oyaw), math.sin(-oyaw)
        gdx = c * ldx + s * ldy
        gdy = -s * ldx + c * ldy
        x = ox + gdx
        y = oy + gdy
    return x, y, yaw


def _generate_dubins_local_course(
    lengths: List[float],
    modes: List[str],
    curvature: float,
    step_m: float,
) -> Tuple[List[float], List[float], List[float]]:
    step_norm = max(step_m*curvature,1e-3)
    origin = (0.,0.,0.)
    chunks = [[np.array([0.])] for _ in range(3)]
    for length,mode in zip(lengths,modes):
        if length<=1e-9:
            continue
        distances=np.r_[np.arange(step_norm,length,step_norm),length]
        coordinates=sample_primitive(distances,mode,curvature,origin)
        for chunk,values in zip(chunks,coordinates):
            chunk.append(values)
        origin=tuple(float(values[-1]) for values in coordinates)
    return tuple(np.concatenate(chunk) for chunk in chunks)


def iter_dubins_segments(
    sx: float,
    sy: float,
    syaw: float,
    gx: float,
    gy: float,
    gyaw: float,
    curvatures: float | Iterable[float],
    step: float = 0.2,
) -> Iterator[TrajectorySegment]:
    """Yield valid Dubins curve segments between two poses, shortest first."""
    if isinstance(curvatures, (int, float)):
        curv_list = [float(curvatures)]
    else:
        curv_list = [float(c) for c in curvatures]

    if not math.isfinite(step) or step<=0 or not curv_list or any(not math.isfinite(c) or c<=0 for c in curv_list):
        raise ValueError("Connector step and curvature must be finite and positive")
    dx = gx - sx
    dy = gy - sy
    c0, s0 = math.cos(syaw), math.sin(syaw)
    # Goal in local frame of start pose
    lx = c0 * dx + s0 * dy
    ly = -s0 * dx + c0 * dy
    lyaw = gyaw - syaw

    D = math.hypot(lx, ly)
    theta = _mod2pi(math.atan2(ly, lx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(lyaw - theta)

    candidates = []

    for curv in curv_list:
        d = D * curv
        for primitive in _DUBINS_PRIMITIVES:
            t, p, q, mode = primitive(alpha, beta, d)
            if t is None or p is None or q is None:
                continue
            lengths_norm = [t, p, q]
            total_len_m = (abs(t) + abs(p) + abs(q)) / curv
            if total_len_m <= 1e-4:
                continue
            candidates.append((total_len_m, curv, lengths_norm, mode))

    # Sampling is the expensive step, so it only happens for requested candidates.
    candidates.sort(key=lambda c: c[0])
    for total_len_m, curv, lengths_norm, mode in candidates:
        xs_loc, ys_loc, yaws_loc = _generate_dubins_local_course(lengths_norm, mode, curv, step)
        xs_arr = np.asarray(xs_loc, dtype=float)
        ys_arr = np.asarray(ys_loc, dtype=float)

        # Transform back to global frame
        gx_arr = c0 * xs_arr - s0 * ys_arr + sx
        gy_arr = s0 * xs_arr + c0 * ys_arr + sy
        gyaw_arr = (np.asarray(yaws_loc)+syaw+np.pi)%(2*np.pi)-np.pi
        dirs_arr = np.ones(len(gx_arr), dtype=int)

        yield TrajectorySegment(
            x=gx_arr,
            y=gy_arr,
            yaw=gyaw_arr,
            direction=dirs_arr,
            ctypes=mode,
            lengths=[l / curv for l in lengths_norm],
            total_length=total_len_m,
        )


def plan_dubins_segments(
    sx: float,
    sy: float,
    syaw: float,
    gx: float,
    gy: float,
    gyaw: float,
    curvatures: float | Iterable[float],
    step: float = 0.2,
) -> List[TrajectorySegment]:
    """Compute all valid Dubins curve segments between two poses, sorted by path length."""
    return list(iter_dubins_segments(sx, sy, syaw, gx, gy, gyaw, curvatures, step))


def plan_dubins_path(
    sx: float,
    sy: float,
    syaw: float,
    gx: float,
    gy: float,
    gyaw: float,
    maxc: float | Iterable[float],
    step: float = 0.2,
) -> List[TrajectorySegment]:
    """Alias for ``plan_dubins_segments``."""
    return plan_dubins_segments(sx, sy, syaw, gx, gy, gyaw, maxc, step=step)

