"""Closed-loop MPC trajectory tracking & Real-Time Dynamic Obstacle Avoidance simulation runner."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import copy
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
from non_holonomic_vehicle_planning.trajectory import geometry, profile_speed, inspect_path


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
    termination_reason: str = "time_limit"
    telemetry: dict = field(default_factory=dict)

    def export(self, destination):
        """Write aligned CSV rows and strict JSON metrics (SI units)."""
        from non_holonomic_vehicle_planning.evaluation import export_result
        return export_result(self, destination)


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

        if scenario.goal is not None:
            gx,gy,gyaw = scenario.goal_pose_rad
            if (math.hypot(base_path.x[-1]-gx,base_path.y[-1]-gy)>1e-3
                    or abs(wrap_angle(base_path.yaw[-1]-gyaw))>1e-3):
                raise ValueError("Planned endpoint does not match the configured goal pose")
        if not inspect_path(base_path,scenario.vehicle)["curvature_feasible"]:
            raise ValueError("Reference curvature exceeds physical steering limit")
        cfg = self.mpc_config
        veh = scenario.vehicle
        controller = MPCController(config=cfg, vehicle=veh)
        rt_planner = RealTimePlanner(scenario=scenario, config=self.realtime_config)

        active_path = base_path
        initial_rx = base_path.x.copy()
        initial_ry = base_path.y.copy()

        sp = profile_speed(
            active_path.x, active_path.y, active_path.yaw, active_path.reversing, cfg.target_speed, veh.max_accel, cfg.max_lateral_accel, veh.max_steer_rate, veh.wheelbase
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

        columns = ("wall_time_s", "target_speed_m_s", "accel_cmd_m_s2", "accel_actual_m_s2",
                   "steer_cmd_rad", "steer_rate_rad_s", "steer_accel_rad_s2", "reference_index",
                   "progress_m", "reference_curvature_m_inv", "replan_event", "path_revision",
                   "solver_status", "solver_iterations", "solver_runtime_s", "control_runtime_s", "sqp_iterations", "sqp_status",
                   "solver_fallback", "static_clearance_m", "static_collision", "dynamic_collision", "goal_reached")
        telemetry = {name: ["not_run" if name in ("solver_status","sqp_status") else 0.] for name in columns}
        telemetry["static_clearance_m"][0] = self._static_clearance(scenario,state)
        revision = 0
        progress_offset = 0.
        total_progress = 0.
        _, curvature = geometry(active_path.x,active_path.y,active_path.yaw,active_path.reversing)

        a_opt: Optional[np.ndarray] = None
        delta_opt: Optional[np.ndarray] = None
        sim_time = 0.0
        reached_goal = False
        termination_reason = "time_limit"

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
                revision += 1
                previous_progress = total_progress
                previous_index = ref_path.ind_old
                replan_count += 1
                active_path = new_path
                sp = profile_speed(
                    active_path.x, active_path.y, active_path.yaw, active_path.reversing, cfg.target_speed, veh.max_accel, cfg.max_lateral_accel, veh.max_steer_rate, veh.wheelbase
                )
                ref_path = ReferencePath(active_path.x, active_path.y, active_path.yaw, active_path.reversing)
                dists_all = np.hypot(ref_path.cx - state.x, ref_path.cy - state.y)
                lo = max(0,previous_index-cfg.search_window)
                hi = min(len(dists_all),previous_index+cfg.search_window+1)
                ref_path.ind_old = lo+int(np.argmin(dists_all[lo:hi]))
                ref_path.active_cusp_ptr = int(np.searchsorted(ref_path.cusp_ends,ref_path.ind_old,side="left"))
                ref_path.progress_s = float(ref_path.s[ref_path.ind_old])
                # The traversed prefix may meet the bypass with a kink; only what lies ahead may limit speed.
                ahead = min(ref_path.ind_old+1,len(active_path.x)-2)
                sp[ahead:] = profile_speed(
                    active_path.x[ahead:], active_path.y[ahead:], active_path.yaw[ahead:], active_path.reversing[ahead:], cfg.target_speed, veh.max_accel, cfg.max_lateral_accel, veh.max_steer_rate, veh.wheelbase
                )
                sp[:ahead] = sp[ahead]
                progress_offset = previous_progress-ref_path.progress_s
                _, curvature = geometry(active_path.x,active_path.y,active_path.yaw,active_path.reversing)
                a_opt = delta_opt = None

            # 2. Compute Reference Horizon for Linear MPC
            z_ref, target_ind, lat_err = ref_path.calc_ref_trajectory(state, sp, cfg)
            if rt_info.speed_scale < 0.999:
                indices = np.arange(cfg.horizon+1)
                for component in (0,1,3):
                    z_ref[component] = np.interp(indices*rt_info.speed_scale,indices,z_ref[component])
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
            a_opt, delta_opt, x_opt, y_opt, _, _ = controller.step(z_ref, z0, a_opt, delta_opt, ref_path.steering_hold)

            a_exc = float(a_opt[0]) if a_opt is not None else 0.0
            delta_exc = float(delta_opt[0]) if delta_opt is not None else 0.0

            previous_rate = state.steer_rate
            state.update(a_exc, delta_exc, direct=1.0, dt=cfg.dt)
            sim_time += cfg.dt

            measured_ref = copy.copy(ref_path)
            measured_index, lat_err = measured_ref.nearest_index(state,cfg.search_window,cfg.speed_stop)
            head_err = wrap_angle(state.yaw-measured_ref.projected_yaw)
            total_progress = max(total_progress,progress_offset+measured_ref.progress_s)
            values = dict(wall_time_s=time.perf_counter()-t0_wall,target_speed_m_s=z_ref[2,0],
                accel_cmd_m_s2=state.accel_cmd,accel_actual_m_s2=state.accel,
                steer_cmd_rad=state.steer_cmd,steer_rate_rad_s=state.steer_rate,
                steer_accel_rad_s2=(state.steer_rate-previous_rate)/cfg.dt,
                reference_index=measured_index,progress_m=total_progress,
                reference_curvature_m_inv=curvature[measured_index],replan_event=rt_info.replan_triggered,
                path_revision=revision,solver_status=controller.last_solver_status,
                solver_iterations=controller.last_solver_iterations,solver_runtime_s=controller.last_solver_runtime_s,
                control_runtime_s=controller.last_control_runtime_s,
                sqp_iterations=controller.last_sqp_iterations,sqp_status=controller.last_sqp_status,solver_fallback=controller.last_fallback,
                static_clearance_m=self._static_clearance(scenario,state),goal_reached=False,
                static_collision=self._static_clearance(scenario,state)<=0.,
                dynamic_collision=scenario.min_dynamic_clearance_at(state.x,state.y,state.yaw,sim_time)<=0.)
            for key in columns:
                telemetry[key].append(values[key])
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

            if values["static_collision"] or values["dynamic_collision"]:
                termination_reason = "collision"
                break
            on_final_segment = ref_path.active_cusp_ptr >= len(ref_path.cusp_ends) - 1
            dist_to_goal = math.hypot(state.x - float(active_path.x[-1]), state.y - float(active_path.y[-1]))
            if (on_final_segment and dist_to_goal < cfg.dist_stop and abs(state.v) < cfg.speed_stop
                    and abs(wrap_angle(state.yaw-float(active_path.yaw[-1]))) < cfg.heading_stop):
                reached_goal = True
                termination_reason = "goal"
                telemetry["goal_reached"][-1] = True
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
            termination_reason=termination_reason,
            telemetry={key: np.asarray(value) for key,value in telemetry.items()},
        )

    @staticmethod
    def _static_clearance(scenario, state):
        footprint = scenario.vehicle.footprint_polygon(state.x,state.y,state.yaw)
        obstacles = list(scenario.obstacles)
        if scenario.boundary is not None:
            obstacles.append(scenario.boundary)
        return min((float(footprint.distance(obstacle)) for obstacle in obstacles),default=float("inf"))


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

    export_stem = Path(plot_output_path).with_suffix("") if plot_output_path is not None else Path("plots") / name
    result.export(export_stem)

    if plot_output_path is not None:
        plot_simulation_summary(result, output_path=plot_output_path, vis_config=vis_config)

    if animate:
        out_gif = Path(gif_output_path) if gif_output_path is not None else Path("plots") / f"{name}.gif"
        animate_simulation(result, output_path=out_gif, vis_config=vis_config)
        print(f"[{name}] Saved 2D Planar Vehicle GIF animation -> {out_gif}")

    return result
