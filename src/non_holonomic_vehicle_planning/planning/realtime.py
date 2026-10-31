"""Real-Time Local Spatio-Temporal Replanner for Moving Obstacles (Vehicles & Pedestrians).

Runs alongside the MPC; latency is measured rather than guaranteed:
- Monitors the active reference path over a spatio-temporal lookahead horizon ``[t, t + T_pred]``
  against velocity-anisotropic inflated dynamic obstacles (other vehicles, pedestrians, cyclists).
- Generates a bank of 3-phase non-holonomic local overtaking / bypass trajectories
  (pull-out -> parallel pass alongside obstacle -> smooth merge-back) when an obstacle blocks
  the lane, selecting the minimum-cost collision-free detour and splicing it seamlessly into
  the reference trajectory.
- Falls back to a protective yield / braking profile (``YIELD_STOP``) when a narrow corridor is
  temporarily blocked by a crossing pedestrian, resuming automatically once clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import List, Literal, Optional, Tuple

import numpy as np
from non_holonomic_vehicle_planning.trajectory import geometry, lateral_bypass_profile, min_transition_length
from shapely.geometry import Point

from non_holonomic_vehicle_planning.config import RealTimePlannerConfig
from non_holonomic_vehicle_planning.control.kinematic_model import KinematicState
from non_holonomic_vehicle_planning.planning.base import PlannedPath, TrajectorySegment
from non_holonomic_vehicle_planning.planning.dubins import plan_dubins_segments
from non_holonomic_vehicle_planning.scenario import Scenario
from non_holonomic_vehicle_planning.vehicle import wrap_angle

PlannerMode = Literal["CRUISE", "AVOID_BYPASS", "YIELD_STOP"]

BYPASS_CURVATURE_SHARE = 0.6  # Share of the curvature limit used by the lateral ramp; rest stays for base-path curves.
MAX_BYPASS_HEADING_ERROR = math.radians(45.0)  # [rad] Frenet start values stop being meaningful beyond this.


@dataclass
class RealTimeStepInfo:
    """Diagnostics from a single real-time replanning cycle."""

    mode: PlannerMode = "CRUISE"
    replan_triggered: bool = False
    replan_latency_ms: float = 0.0
    speed_scale: float = 1.0
    min_dynamic_clearance_m: float = float("inf")
    candidate_paths: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    selected_bypass: Optional[Tuple[np.ndarray, np.ndarray]] = None
    blocking_obstacle_id: Optional[str] = None


class RealTimePlanner:
    """Real-time local path adapter & dynamic obstacle avoidance planner."""

    def __init__(
        self,
        scenario: Scenario,
        config: Optional[RealTimePlannerConfig] = None,
    ) -> None:
        self.scenario = scenario
        self.config = config or RealTimePlannerConfig()
        self._active_mode: PlannerMode = "CRUISE"
        self._cooldown_steps: int = 0
        self._last_bypass_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._last_candidates: List[Tuple[np.ndarray, np.ndarray]] = []

    def check_lookahead_conflict(
        self,
        path: PlannedPath,
        start_idx: int,
        current_time: float,
        nominal_speed: float,
    ) -> Tuple[bool, Optional[int], Optional[int], Optional[str], float]:
        """Check whether the upcoming lookahead segment collides with any dynamic obstacle in space-time.

        Returns:
            (has_conflict, conflict_start_idx, conflict_end_idx, blocking_obstacle_id, conflict_distance_m)
        """
        if not self.scenario.dynamic_obstacles or not self.config.enabled:
            return False, None, None, None, float("inf")

        n = len(path.x)
        dist_acc = 0.0
        v_est = max(abs(nominal_speed), 1.8)
        r_ego = self.scenario.inflater.disk_radius + self.config.replan_clearance_trigger_m

        conflict_start_idx: Optional[int] = None
        conflict_end_idx: Optional[int] = None
        blocker_id: Optional[str] = None
        conflict_dist = float("inf")

        for idx in range(start_idx, n):
            if idx > start_idx:
                dist_acc += math.hypot(
                    float(path.x[idx] - path.x[idx - 1]),
                    float(path.y[idx] - path.y[idx - 1]),
                )
            t_arrival = current_time + dist_acc / v_est

            if conflict_start_idx is None:
                if dist_acc > self.config.lookahead_distance_m:
                    break
                if t_arrival - current_time > self.config.prediction_horizon_s:
                    break
            else:
                if dist_acc > self.config.lookahead_distance_m + 12.0:
                    break

            disks = self.scenario.inflater.pose_disks_world(float(path.x[idx]),float(path.y[idx]),float(path.yaw[idx]))
            points = [Point(float(px),float(py)) for px,py in disks]

            for dyn in self.scenario.dynamic_obstacles:
                if blocker_id is not None and dyn.id != blocker_id:
                    continue
                for t_probe in (current_time, 0.5 * (current_time + t_arrival), t_arrival):
                    inf_poly = dyn.inflated_polygon_at(
                        t_probe,
                        self.scenario.inflation_config,
                        ego_disk_radius=r_ego,
                    )
                    if any(inf_poly.intersects(pt) for pt in points):
                        if conflict_start_idx is None:
                            conflict_start_idx = idx
                            conflict_dist = dist_acc
                            blocker_id = dyn.id
                        conflict_end_idx = idx
                        break

        return (conflict_start_idx is not None), conflict_start_idx, conflict_end_idx, blocker_id, conflict_dist

    def _segment_spatio_temporal_safe(
        self,
        seg: TrajectorySegment,
        t_start: float,
        nominal_speed: float,
        blocker_id: Optional[str] = None,
        s_pass_end: float = float("inf"),
    ) -> Tuple[bool, float]:
        """Verify a candidate local segment against both static and dynamic inflated obstacles."""
        if not self.scenario.collision_free(seg):
            return False, float("inf")

        v_est = max(abs(nominal_speed), 2.0)
        r_ego = self.scenario.inflater.disk_radius
        s_acc = 0.0
        min_dyn_dist = float("inf")

        step = 1
        for i in range(0, len(seg.x), step):
            if i > 0:
                prev_i = max(0, i - step)
                s_acc += math.hypot(float(seg.x[i] - seg.x[prev_i]), float(seg.y[i] - seg.y[prev_i]))
            t_i = t_start + s_acc / v_est
            # Cover all footprint disks, including the front/rear swept pose.
            disks = self.scenario.inflater.pose_disks_world(float(seg.x[i]),float(seg.y[i]),float(seg.yaw[i]))
            for dyn in self.scenario.dynamic_obstacles:
                for t_check in (t_start, t_i) if dyn.id == blocker_id else (t_i,):
                    poly = dyn.inflated_polygon_at(t_check, self.scenario.inflation_config, ego_disk_radius=r_ego)
                    points = [Point(float(px),float(py)) for px,py in disks]
                    if any(poly.intersects(pt) for pt in points):
                        return False, 0.0
                    d = min(float(poly.distance(pt)) for pt in points)
                    if d < min_dyn_dist:
                        min_dyn_dist = d

        soft_cost = self.scenario.segment_inflation_cost(seg)
        return True, soft_cost + 2.0 / max(min_dyn_dist, 0.2)

    def generate_local_bypass(
        self,
        state: KinematicState,
        active_path: PlannedPath,
        current_idx: int,
        conflict_start_idx: int,
        conflict_end_idx: int,
        current_time: float,
        nominal_speed: float,
        blocker_id: Optional[str] = None,
    ) -> Tuple[Optional[PlannedPath], List[Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Search a bank of 3-phase C² Frenet quintic overtaking trajectories around the conflict zone.

        Phases along reference arc-length ``s`` (measured from the projected vehicle pose):
          1. Pull-out: quintic from the measured ``(d, d', d'')`` to ``d = lat_offset`` over ``[0, s_pull]``.
          2. Parallel pass: maintains ``d = lat_offset`` alongside the obstacle over ``[s_pull, s_pass]``.
          3. Merge-back: rest-to-rest quintic from ``d = lat_offset`` to ``0`` over ``[s_pass, s_rejoin]``.
        """
        n = len(active_path.x)
        if current_idx >= n - 6 or np.any(active_path.reversing[current_idx:conflict_end_idx+1]):
            return None, [], None

        # Compute cumulative arc-length from current_idx along active_path
        ref_x = active_path.x[current_idx:]
        ref_y = active_path.y[current_idx:]
        ref_yaw = active_path.yaw[current_idx:]
        ds = np.hypot(np.diff(ref_x), np.diff(ref_y))
        s_rel = np.concatenate(([0.0], np.cumsum(ds)))
        s_total_rem = float(s_rel[-1])
        if s_total_rem < 6.0:
            return None, [], None

        c_start_rel = max(0, conflict_start_idx - current_idx)
        c_end_rel = max(c_start_rel + 1, min(len(s_rel) - 1, conflict_end_idx - current_idx))
        s_c_start = float(s_rel[min(c_start_rel, len(s_rel) - 1)])
        s_c_end = float(s_rel[c_end_rel])

        # Phase boundaries along arc-length s:
        # Reach full lateral offset right before the conflict box starts (s_c_start - 0.3)
        s_pull = max(3.2, s_c_start - 0.3)
        s_pass = min(s_total_rem - 1.5, max(s_pull + 2.5, s_c_end + 0.8))
        max_kappa = self.scenario.vehicle.max_curvature
        widest = max(map(abs, self.config.lateral_offsets_m))
        s_rejoin = min(s_total_rem, s_pass + max(10.0, s_pull, min_transition_length(widest, BYPASS_CURVATURE_SHARE*max_kappa)))
        if s_rejoin <= s_pull + 1.5:
            return None, [], None

        rejoin_rel_idx = int(np.searchsorted(s_rel, s_rejoin, side="left"))
        rejoin_rel_idx = min(len(s_rel) - 1, max(c_end_rel + 2, rejoin_rel_idx))
        rejoin_idx = current_idx + rejoin_rel_idx

        s_slice = s_rel[:rejoin_rel_idx + 1]
        x_slice = ref_x[:rejoin_rel_idx + 1]
        y_slice = ref_y[:rejoin_rel_idx + 1]
        yaw_slice = np.unwrap(ref_yaw[:rejoin_rel_idx + 1])
        if np.any(np.diff(s_slice) <= 1e-8) or len(s_slice) < 6:
            return None, [], None
        nx = -np.sin(yaw_slice)
        ny = np.cos(yaw_slice)
        forward = len(x_slice)*[False]
        _, base_kappa = geometry(x_slice, y_slice, yaw_slice, forward)

        # Frenet boundary values of the measured pose at the projection on the base path.
        dx = state.x - float(x_slice[0])
        dy = state.y - float(y_slice[0])
        s_vehicle = dx*math.cos(yaw_slice[0]) + dy*math.sin(yaw_slice[0])
        d0 = dx*float(nx[0]) + dy*float(ny[0])
        heading_error = wrap_angle(state.yaw - float(yaw_slice[0]))
        if abs(heading_error) > MAX_BYPASS_HEADING_ERROR:
            return None, [], None
        d1 = (1.0 - float(base_kappa[0])*d0)*math.tan(heading_error)
        d2 = math.tan(state.steer)/self.scenario.vehicle.wheelbase - float(base_kappa[0])
        s_local = s_slice - s_vehicle
        s_end = float(s_local[-1])

        candidates_xy: List[Tuple[np.ndarray, np.ndarray]] = []
        best_seg: Optional[TrajectorySegment] = None
        best_score = float("inf")

        for lat_offset in self.config.lateral_offsets_m:
            # A longer pull-out is only a candidate; collision checks below decide.
            pull_length = max(s_pull, min_transition_length(lat_offset - d0, BYPASS_CURVATURE_SHARE*max_kappa))
            pass_end = max(s_pass, pull_length)
            if s_end <= pass_end + 1.5:
                continue
            d_prof, d_slope = lateral_bypass_profile(s_local, pull_length, pass_end, s_end, lat_offset, d0, d1, d2)
            scale = 1.0 - base_kappa*d_prof
            if np.min(scale) < 0.1:
                continue
            cand_x = x_slice + d_prof * nx
            cand_y = y_slice + d_prof * ny
            cand_yaw = yaw_slice + np.arctan2(d_slope, scale)
            _, kappa = geometry(cand_x, cand_y, cand_yaw, forward)
            if np.max(abs(kappa)) > max_kappa + 1e-3:
                continue

            if (
                np.any(cand_x < 0.8)
                or np.any(cand_x > self.scenario.width - 0.8)
                or np.any(cand_y < 0.8)
                or np.any(cand_y > self.scenario.height - 0.8)
            ):
                continue

            seg_ds = np.hypot(np.diff(cand_x), np.diff(cand_y))

            cand_len = float(np.sum(seg_ds))
            combined = TrajectorySegment(
                x=cand_x,
                y=cand_y,
                yaw=cand_yaw,
                direction=np.ones(len(cand_x), dtype=int),
                total_length=cand_len,
            )
            candidates_xy.append((cand_x, cand_y))

            is_safe, clearance_penalty = self._segment_spatio_temporal_safe(
                combined, current_time, nominal_speed, blocker_id=blocker_id, s_pass_end=s_pass + 1.0
            )
            if not is_safe:
                continue

            score = combined.length + 0.35 * abs(lat_offset) + clearance_penalty
            if score < best_score:
                best_score = score
                best_seg = combined

        if best_seg is None:
            return None, candidates_xy, None

        # Splice bypass into active_path: [0..current_idx] + best_seg + [rejoin_idx+1..]
        new_x = np.concatenate((active_path.x[:current_idx], best_seg.x, active_path.x[rejoin_idx + 1:]))
        new_y = np.concatenate((active_path.y[:current_idx], best_seg.y, active_path.y[rejoin_idx + 1:]))
        new_yaw = np.concatenate((active_path.yaw[:current_idx], best_seg.yaw, active_path.yaw[rejoin_idx + 1:]))
        new_dir = np.concatenate((active_path.direction[:current_idx], best_seg.direction, active_path.direction[rejoin_idx + 1:]))

        spliced = PlannedPath(x=new_x, y=new_y, yaw=new_yaw, direction=new_dir).resample_uniform(step=0.2)
        selected_xy = (best_seg.x.copy(), best_seg.y.copy())
        return spliced, candidates_xy, selected_xy

    def step(
        self,
        state: KinematicState,
        active_path: PlannedPath,
        current_idx: int,
        current_time: float,
        nominal_speed: float,
    ) -> Tuple[PlannedPath, RealTimeStepInfo]:
        """Run one real-time local planning & collision-avoidance supervision step."""
        t0 = time.perf_counter()
        min_dyn_clear = self.scenario.min_dynamic_clearance_at(state.x, state.y, state.yaw, current_time)

        if not self.scenario.dynamic_obstacles or not self.config.enabled:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return active_path, RealTimeStepInfo(
                mode="CRUISE",
                replan_triggered=False,
                replan_latency_ms=elapsed_ms,
                speed_scale=1.0,
                min_dynamic_clearance_m=min_dyn_clear,
            )

        if self._cooldown_steps > 0:
            self._cooldown_steps -= 1

        has_conflict, c_start, c_end, blocker_id, conflict_dist = self.check_lookahead_conflict(
            active_path, current_idx, current_time, nominal_speed
        )

        if not has_conflict or c_start is None or c_end is None:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            mode: PlannerMode = "AVOID_BYPASS" if self._cooldown_steps > 0 else "CRUISE"
            return active_path, RealTimeStepInfo(
                mode=mode,
                replan_triggered=False,
                replan_latency_ms=elapsed_ms,
                speed_scale=1.0,
                min_dynamic_clearance_m=min_dyn_clear,
                candidate_paths=self._last_candidates if self._cooldown_steps > 0 else [],
                selected_bypass=self._last_bypass_xy if self._cooldown_steps > 0 else None,
            )

        # Conflict detected: try local 3-phase non-holonomic bypass if not in cooldown
        if self._cooldown_steps == 0:
            spliced_path, cands, sel_xy = self.generate_local_bypass(
                state,
                active_path,
                current_idx,
                c_start,
                c_end,
                current_time,
                nominal_speed,
                blocker_id=blocker_id,
            )
            self._last_candidates = cands
            if spliced_path is not None and sel_xy is not None:
                self._last_bypass_xy = sel_xy
                self._active_mode = "AVOID_BYPASS"
                self._cooldown_steps = 15
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return spliced_path, RealTimeStepInfo(
                    mode="AVOID_BYPASS",
                    replan_triggered=True,
                    replan_latency_ms=elapsed_ms,
                    speed_scale=1.0,
                    min_dynamic_clearance_m=min_dyn_clear,
                    candidate_paths=cands,
                    selected_bypass=sel_xy,
                    blocking_obstacle_id=blocker_id,
                )

        # If no spatial bypass is possible (e.g. pedestrian crossing a narrow corridor), yield/brake smoothly!
        stop_buf = self.config.yield_stop_buffer_m
        if conflict_dist <= stop_buf:
            speed_scale = 0.0
        else:
            speed_scale = float(np.clip((conflict_dist - stop_buf) / max(self.config.lookahead_distance_m - stop_buf, 2.0), 0.0, 0.7))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return active_path, RealTimeStepInfo(
            mode="YIELD_STOP",
            replan_triggered=False,
            replan_latency_ms=elapsed_ms,
            speed_scale=speed_scale,
            min_dynamic_clearance_m=min_dyn_clear,
            candidate_paths=self._last_candidates,
            selected_bypass=None,
            blocking_obstacle_id=blocker_id,
        )
