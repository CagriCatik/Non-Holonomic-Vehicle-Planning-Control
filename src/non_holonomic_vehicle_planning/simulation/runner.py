"""Closed-loop MPC trajectory tracking & Real-Time Dynamic Obstacle Avoidance simulation runner."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import List, Optional, Tuple

import numpy as np

from non_holonomic_vehicle_planning.config import RealTimePlannerConfig, VisualizationConfig, load_config
from non_holonomic_vehicle_planning.control.kinematic_model import KinematicState
from non_holonomic_vehicle_planning.control.mpc import MPCConfig, MPCController, ReferencePath, calc_speed_profile
from non_holonomic_vehicle_planning.planning.base import PlannedPath
from non_holonomic_vehicle_planning.planning.realtime import PlannerMode, RealTimePlanner
from non_holonomic_vehicle_planning.scenario import Scenario
from non_holonomic_vehicle_planning.vehicle import wrap_angle


@dataclass
class SimulationResult:
    """Telemetry history and summary metrics from a closed-loop MPC + Real-Time Planner simulation."""

    scenario_name: str
    scenario: Scenario
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray         # [rad]
    v: np.ndarray           # [m/s]
    steer: np.ndarray       # [rad] front wheel steering angle (Radeinschlag)
    accel: np.ndarray       # [m/s^2] longitudinal acceleration command
    target_ind: np.ndarray  # index of tracked reference point
    lateral_error: np.ndarray
    heading_error: np.ndarray
    reversing: np.ndarray   # boolean array: True when reversing

    ref_x: np.ndarray
    ref_y: np.ndarray
    ref_yaw: np.ndarray
    ref_reversing: np.ndarray

    # Initial global planned path (before any real-time local bypasses)
    initial_ref_x: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    initial_ref_y: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    horizon_x: List[np.ndarray] = field(default_factory=list)
    horizon_y: List[np.ndarray] = field(default_factory=list)
    ref_horizon_x: List[np.ndarray] = field(default_factory=list)
    ref_horizon_y: List[np.ndarray] = field(default_factory=list)

    # Real-time dynamic replanning telemetry
    planner_modes: List[PlannerMode] = field(default_factory=list)
    replan_latencies_ms: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    min_dynamic_clearances_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    candidate_bypasses: List[List[Tuple[np.ndarray, np.ndarray]]] = field(default_factory=list)
    active_bypasses: List[Optional[Tuple[np.ndarray, np.ndarray]]] = field(default_factory=list)
    replan_count: int = 0

    wall_time_s: float = 0.0
    planned_length_m: float = 0.0
    simulated_length_m: float = 0.0
    reached_goal: bool = False


class SimulationRunner:
    """Executes closed-loop MPC tracking and real-time dynamic obstacle avoidance."""

    def __init__(
        self,
        mpc_config: Optional[MPCConfig] = None,
        realtime_config: Optional[RealTimePlannerConfig] = None,
    ) -> None:
        self.mpc_config = mpc_config or MPCConfig()
        self.realtime_config = realtime_config or RealTimePlannerConfig()

    def run(
        self,
        scenario: Scenario,
        planned_path: Optional[PlannedPath] = None,
        label: Optional[str] = None,
    ) -> SimulationResult:
        """Simulate the vehicle tracking ``scenario.planned_path`` with real-time dynamic avoidance."""
        base_path = planned_path or scenario.planned_path
        if base_path is None or len(base_path.x) < 2:
            raise ValueError("Scenario has no planned path to simulate.")

        cfg = self.mpc_config
        veh = scenario.vehicle
        controller = MPCController(config=cfg, vehicle=veh)
        rt_planner = RealTimePlanner(scenario=scenario, config=self.realtime_config)

        active_path = base_path
        initial_rx = base_path.x.copy()
        initial_ry = base_path.y.copy()

        sp = calc_speed_profile(
            active_path.x, active_path.y, active_path.yaw, cfg.target_speed, active_path.reversing
        )
        ref_path = ReferencePath(active_path.x, active_path.y, active_path.yaw, active_path.reversing)

        state = KinematicState(
            x=float(active_path.x[0]),
            y=float(active_path.y[0]),
            yaw=float(ref_path.cyaw[0]),
            v=0.0,
            steer=0.0,
            vehicle=veh,
        )

        init_dyn_clear = scenario.min_dynamic_clearance_at(state.x, state.y, state.yaw, 0.0)

        t_hist: List[float] = [0.0]
        x_hist: List[float] = [state.x]
        y_hist: List[float] = [state.y]
        yaw_hist: List[float] = [state.yaw]
        v_hist: List[float] = [state.v]
        steer_hist: List[float] = [0.0]
        accel_hist: List[float] = [0.0]
        ind_hist: List[int] = [0]
        lat_err_hist: List[float] = [0.0]
        head_err_hist: List[float] = [0.0]
        rev_hist: List[bool] = [bool(active_path.reversing[0])]
        horizon_x_hist: List[np.ndarray] = [np.full(cfg.horizon + 1, state.x)]
        horizon_y_hist: List[np.ndarray] = [np.full(cfg.horizon + 1, state.y)]
        ref_horizon_x_hist: List[np.ndarray] = [np.full(cfg.horizon + 1, state.x)]
        ref_horizon_y_hist: List[np.ndarray] = [np.full(cfg.horizon + 1, state.y)]

        modes_hist: List[PlannerMode] = ["CRUISE"]
        latencies_hist: List[float] = [0.0]
        dyn_clear_hist: List[float] = [init_dyn_clear]
        cands_hist: List[List[Tuple[np.ndarray, np.ndarray]]] = [[]]
        bypasses_hist: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None]
        replan_count = 0

        a_opt: Optional[np.ndarray] = None
        delta_opt: Optional[np.ndarray] = None
        sim_time = 0.0
        reached_goal = False

        t0_wall = time.perf_counter()

        while sim_time < cfg.time_max:
            # 1. Real-Time Dynamic Obstacle Supervision & Local Replanning
            new_path, rt_info = rt_planner.step(
                state=state,
                active_path=active_path,
                current_idx=ref_path.ind_old,
                current_time=sim_time,
                nominal_speed=cfg.target_speed,
            )
            if rt_info.replan_triggered:
                replan_count += 1
                active_path = new_path
                scenario.set_planned_path(active_path)
                sp = calc_speed_profile(
                    active_path.x, active_path.y, active_path.yaw, cfg.target_speed, active_path.reversing
                )
                ref_path = ReferencePath(active_path.x, active_path.y, active_path.yaw, active_path.reversing)
                dists_all = np.hypot(ref_path.cx - state.x, ref_path.cy - state.y)
                ref_path.ind_old = int(np.argmin(dists_all))

            # 2. Compute Reference Horizon for Linear MPC
            z_ref, target_ind, lat_err = ref_path.calc_ref_trajectory(state, sp, cfg)
            if rt_info.speed_scale < 0.999:
                z_ref[2, :] *= rt_info.speed_scale

            z0 = [
                state.x,
                state.y,
                state.v,
                state.yaw,
                state.steer,
                state.steer_rate,
                state.accel,
            ]
            a_opt, delta_opt, x_opt, y_opt, _, _ = controller.step(z_ref, z0, a_opt, delta_opt)

            a_exc = float(a_opt[0]) if a_opt is not None else 0.0
            delta_exc = float(delta_opt[0]) if delta_opt is not None else 0.0

            # Enforce smooth, exact zero-overshoot braking if yielding to a pedestrian right ahead
            if rt_info.mode == "YIELD_STOP" and rt_info.speed_scale <= 0.05:
                if state.v > 1e-3:
                    a_exc = -min(veh.max_accel, state.v / cfg.dt)
                elif state.v < -1e-3:
                    a_exc = min(veh.max_accel, -state.v / cfg.dt)
                else:
                    a_exc = 0.0
                    state.v = 0.0

            state.update(a_exc, delta_exc, direct=1.0, dt=cfg.dt)
            sim_time += cfg.dt

            head_err = wrap_angle(state.yaw - float(z_ref[3, 0]))
            is_rev = (state.v < -0.05) or (abs(state.v) <= 0.05 and bool(active_path.reversing[target_ind]))
            dyn_clear = scenario.min_dynamic_clearance_at(state.x, state.y, state.yaw, sim_time)

            t_hist.append(sim_time)
            x_hist.append(state.x)
            y_hist.append(state.y)
            yaw_hist.append(state.yaw)
            v_hist.append(state.v)
            steer_hist.append(state.steer)
            accel_hist.append(a_exc)
            ind_hist.append(min(target_ind, len(active_path.x) - 1))
            lat_err_hist.append(lat_err)
            head_err_hist.append(head_err)
            rev_hist.append(is_rev)
            modes_hist.append(rt_info.mode)
            latencies_hist.append(rt_info.replan_latency_ms)
            dyn_clear_hist.append(dyn_clear)
            cands_hist.append(rt_info.candidate_paths)
            bypasses_hist.append(rt_info.selected_bypass)

            if x_opt is not None and y_opt is not None:
                horizon_x_hist.append(x_opt.copy())
                horizon_y_hist.append(y_opt.copy())
            else:
                horizon_x_hist.append(np.full(cfg.horizon + 1, state.x))
                horizon_y_hist.append(np.full(cfg.horizon + 1, state.y))
            ref_horizon_x_hist.append(z_ref[0, :].copy())
            ref_horizon_y_hist.append(z_ref[1, :].copy())

            on_final_segment = ref_path.active_cusp_ptr >= len(ref_path.cusp_ends) - 1
            dist_to_goal = math.hypot(state.x - float(active_path.x[-1]), state.y - float(active_path.y[-1]))
            if on_final_segment and dist_to_goal < cfg.dist_stop and abs(state.v) < cfg.speed_stop:
                reached_goal = True
                break
            if on_final_segment and target_ind >= len(active_path.x) - 3 and dist_to_goal < 1.35 * cfg.dist_stop:
                reached_goal = True
                break

        wall_time = time.perf_counter() - t0_wall
        x_arr = np.asarray(x_hist, dtype=float)
        y_arr = np.asarray(y_hist, dtype=float)
        sim_len = float(np.sum(np.hypot(np.diff(x_arr), np.diff(y_arr)))) if len(x_arr) >= 2 else 0.0

        return SimulationResult(
            scenario_name=label or scenario.name,
            scenario=scenario,
            t=np.asarray(t_hist, dtype=float),
            x=x_arr,
            y=y_arr,
            yaw=np.asarray(yaw_hist, dtype=float),
            v=np.asarray(v_hist, dtype=float),
            steer=np.asarray(steer_hist, dtype=float),
            accel=np.asarray(accel_hist, dtype=float),
            target_ind=np.asarray(ind_hist, dtype=int),
            lateral_error=np.asarray(lat_err_hist, dtype=float),
            heading_error=np.asarray(head_err_hist, dtype=float),
            reversing=np.asarray(rev_hist, dtype=bool),
            ref_x=active_path.x,
            ref_y=active_path.y,
            ref_yaw=active_path.yaw,
            ref_reversing=active_path.reversing,
            initial_ref_x=initial_rx,
            initial_ref_y=initial_ry,
            horizon_x=horizon_x_hist,
            horizon_y=horizon_y_hist,
            ref_horizon_x=ref_horizon_x_hist,
            ref_horizon_y=ref_horizon_y_hist,
            planner_modes=modes_hist,
            replan_latencies_ms=np.asarray(latencies_hist, dtype=float),
            min_dynamic_clearances_m=np.asarray(dyn_clear_hist, dtype=float),
            candidate_bypasses=cands_hist,
            active_bypasses=bypasses_hist,
            replan_count=replan_count,
            wall_time_s=wall_time,
            planned_length_m=active_path.length,
            simulated_length_m=sim_len,
            reached_goal=reached_goal,
        )


def run_simulation(
    scenario: Scenario,
    name: str,
    animate: bool = True,
    gif_output_path: Optional[str | Path] = None,
    plot_output_path: Optional[str | Path] = None,
    mpc_config: Optional[MPCConfig] = None,
    realtime_config: Optional[RealTimePlannerConfig] = None,
    vis_config: Optional[VisualizationConfig] = None,
) -> SimulationResult:
    """Run closed-loop simulation and optionally export a GIF animation and summary PNG."""
    from non_holonomic_vehicle_planning.visualization.animation import animate_simulation
    from non_holonomic_vehicle_planning.visualization.plots import plot_simulation_summary

    runner = SimulationRunner(mpc_config=mpc_config, realtime_config=realtime_config)
    result = runner.run(scenario, label=name)

    print(f"[{name}] Simulation Wall Time:        {result.wall_time_s:.3f} s")
    print(f"[{name}] Planned Trajectory Length:   {result.planned_length_m:.2f} m")
    print(f"[{name}] Simulated Trajectory Length: {result.simulated_length_m:.2f} m")
    print(f"[{name}] Max Cross-Track Error:       {np.max(np.abs(result.lateral_error)):.3f} m")
    if scenario.dynamic_obstacles:
        min_clr = float(np.min(result.min_dynamic_clearances_m))
        mean_lat = float(np.mean(result.replan_latencies_ms))
        print(f"[{name}] Real-Time Replans Triggered: {result.replan_count} (avg step latency: {mean_lat:.2f} ms)")
        print(f"[{name}] Min Dynamic Obstacle Clear.: {min_clr:.2f} m")

    if plot_output_path is not None:
        plot_simulation_summary(result, output_path=plot_output_path, vis_config=vis_config)

    if animate:
        out_gif = Path(gif_output_path) if gif_output_path is not None else Path("plots") / f"{name}.gif"
        animate_simulation(result, output_path=out_gif, vis_config=vis_config)
        print(f"[{name}] Saved 2D Planar Vehicle GIF animation -> {out_gif}")

    return result
