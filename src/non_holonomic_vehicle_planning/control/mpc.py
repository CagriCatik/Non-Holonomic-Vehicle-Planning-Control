"""Linear Model Predictive Controller (MPC) for non-holonomic trajectory tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Sequence, Tuple

import cvxpy
import numpy as np

from non_holonomic_vehicle_planning.control.kinematic_model import KinematicState, calc_linear_discrete_model
from non_holonomic_vehicle_planning.vehicle import VehicleParams, wrap_angle


@dataclass
class MPCConfig:
    """Configuration for the Linear Model Predictive Controller."""

    nx: int = 4  # [x, y, v, yaw]
    nu: int = 2  # [acceleration, steer]
    horizon: int = 14

    Q: np.ndarray = field(default_factory=lambda: np.diag([4.0, 4.0, 2.0, 3.5]))
    Qf: np.ndarray = field(default_factory=lambda: np.diag([5.0, 5.0, 1.5, 4.0]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.15]))
    Rd: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.15]))

    dt: float = 0.2
    target_speed: float = 10.0 / 3.6
    dist_stop: float = 0.65
    speed_stop: float = 0.5 / 3.6
    time_max: float = 180.0
    iter_max: int = 4
    du_res: float = 0.08
    search_window: int = 15


def smooth_reference_yaw(cyaw: Sequence[float]) -> np.ndarray:
    """Unwrap reference yaw sequence so there are no 2*pi discontinuities across +-pi."""
    arr = np.asarray(cyaw, dtype=float)
    if arr.size == 0:
        return arr
    return np.unwrap(arr)


def calc_speed_profile(
    cx: Sequence[float],
    cy: Sequence[float],
    cyaw: Sequence[float],
    target_speed: float,
    reversing: Sequence[bool | float],
) -> np.ndarray:
    """Design a smooth speed profile with deceleration toward cusps and the final goal."""
    n = len(cx)
    rev = np.asarray(reversing, dtype=float) > 0.5
    sp = np.where(rev, -abs(target_speed), abs(target_speed)).astype(float)

    # Identify cusp indices (gear changes) and final index
    stop_indices: List[int] = []
    for i in range(n - 1):
        if rev[i] != rev[i + 1]:
            sp[i] = 0.0
            stop_indices.append(i)
    sp[-1] = 0.0
    stop_indices.append(n - 1)

    # Smoothly ramp down speed approaching stop/cusp points
    x_arr = np.asarray(cx, dtype=float)
    y_arr = np.asarray(cy, dtype=float)
    for stop_idx in stop_indices:
        dist_back = 0.0
        for j in range(stop_idx - 1, -1, -1):
            if j < stop_idx - 1 and rev[j] != rev[j + 1]:
                break
            dist_back += math.hypot(x_arr[j + 1] - x_arr[j], y_arr[j + 1] - y_arr[j])
            if dist_back > 3.5:
                break
            ramp = max(0.20, min(1.0, dist_back / 3.5))
            sp[j] = math.copysign(abs(target_speed) * ramp, sp[j])

    return sp


class ReferencePath:
    """Reference trajectory with arc-length parameterization and cusp-aware indexing."""

    def __init__(
        self,
        cx: Sequence[float],
        cy: Sequence[float],
        cyaw: Sequence[float],
        reversing: Optional[Sequence[bool | float]] = None,
    ) -> None:
        self.cx = np.asarray(cx, dtype=float)
        self.cy = np.asarray(cy, dtype=float)
        self.cyaw = smooth_reference_yaw(cyaw)
        self.length = len(self.cx)
        if reversing is None:
            self.reversing = np.zeros(self.length, dtype=bool)
        else:
            self.reversing = np.asarray(reversing, dtype=float) > 0.5

        if self.length >= 2:
            ds = np.hypot(np.diff(self.cx), np.diff(self.cy))
            self.s = np.concatenate(([0.0], np.cumsum(ds)))
            median_ds = float(np.median(ds[ds > 1e-5])) if np.any(ds > 1e-5) else 0.2
            self.d_dist = max(0.05, median_ds)
        else:
            self.s = np.zeros(self.length, dtype=float)
            self.d_dist = 0.2

        # Compute cusp segment boundaries [start_idx, end_idx_inclusive]
        self.cusp_ends: List[int] = []
        for i in range(self.length - 1):
            if self.reversing[i] != self.reversing[i + 1]:
                self.cusp_ends.append(i)
        self.cusp_ends.append(self.length - 1)
        self.active_cusp_ptr: int = 0
        self.ind_old: int = 0

    @property
    def current_segment_end(self) -> int:
        return self.cusp_ends[min(self.active_cusp_ptr, len(self.cusp_ends) - 1)]

    def nearest_index(self, state: KinematicState, search_window: int = 15) -> Tuple[int, float]:
        """Find nearest reference index within the active gear segment and compute cross-track error."""
        self.ind_old = min(max(0, self.ind_old), self.length - 1)
        while self.active_cusp_ptr < len(self.cusp_ends) - 1 and self.ind_old > self.cusp_ends[self.active_cusp_ptr]:
            self.active_cusp_ptr += 1

        seg_end = self.current_segment_end
        search_end = min(seg_end + 1, self.ind_old + search_window)
        if search_end <= self.ind_old:
            search_end = min(self.length, self.ind_old + 1)

        dx = self.cx[self.ind_old:search_end] - state.x
        dy = self.cy[self.ind_old:search_end] - state.y
        dist = np.hypot(dx, dy)

        ind_in_win = int(np.argmin(dist))
        ind = self.ind_old + ind_in_win
        self.ind_old = ind

        # Check if we reached a cusp and should advance to the next gear segment
        if (
            self.active_cusp_ptr < len(self.cusp_ends) - 1
            and ind >= seg_end - 1
            and math.hypot(state.x - self.cx[seg_end], state.y - self.cy[seg_end]) < 0.75
        ):
            self.active_cusp_ptr += 1
            self.ind_old = seg_end + 1
            ind = self.ind_old

        # Signed lateral cross-track error
        normal_vec = np.array([
            math.cos(state.yaw + 0.5 * math.pi),
            math.sin(state.yaw + 0.5 * math.pi),
        ])
        err_vec = np.array([state.x - self.cx[ind], state.y - self.cy[ind]])
        lateral_error = float(np.dot(err_vec, normal_vec))

        return ind, lateral_error

    def calc_ref_trajectory(
        self,
        state: KinematicState,
        speed_profile: Sequence[float],
        cfg: MPCConfig,
    ) -> Tuple[np.ndarray, int, float]:
        """Construct the (4 x (T+1)) reference state horizon aligned with ``state.yaw``."""
        z_ref = np.zeros((cfg.nx, cfg.horizon + 1), dtype=float)
        ind, lat_err = self.nearest_index(state, search_window=cfg.search_window)
        seg_end = self.current_segment_end

        # Align reference yaw branch with current vehicle yaw so |yaw_ref - state.yaw| <= pi
        base_ref_yaw = float(self.cyaw[ind])
        yaw_shift = round((state.yaw - base_ref_yaw) / (2.0 * math.pi)) * (2.0 * math.pi)

        z_ref[0, 0] = self.cx[ind]
        z_ref[1, 0] = self.cy[ind]
        z_ref[2, 0] = speed_profile[ind]
        z_ref[3, 0] = self.cyaw[ind] + yaw_shift

        dist_move = 0.0
        ref_v = max(abs(state.v), 0.35 * cfg.target_speed)

        for i in range(1, cfg.horizon + 1):
            dist_move += ref_v * cfg.dt
            target_s = self.s[ind] + dist_move
            index = int(np.searchsorted(self.s, target_s, side="left"))
            index = min(index, seg_end)

            z_ref[0, i] = self.cx[index]
            z_ref[1, i] = self.cy[index]
            z_ref[2, i] = speed_profile[index]
            z_ref[3, i] = self.cyaw[index] + yaw_shift

        return z_ref, ind, lat_err


class MPCController:
    """Iterative Linear Model Predictive Controller for 2D planar vehicle tracking."""

    def __init__(
        self,
        config: Optional[MPCConfig] = None,
        vehicle: Optional[VehicleParams] = None,
    ) -> None:
        self.config = config or MPCConfig()
        self.vehicle = vehicle or VehicleParams()

    def predict_states(
        self,
        z0: Sequence[float],
        a_seq: Sequence[float],
        delta_seq: Sequence[float],
    ) -> np.ndarray:
        """Roll out the nonlinear kinematic bicycle model over the horizon ``T``."""
        T = self.config.horizon
        z_bar = np.zeros((self.config.nx, T + 1), dtype=float)
        z_bar[:, 0] = np.asarray(z0[: self.config.nx], dtype=float)

        sim_state = KinematicState(
            x=float(z0[0]),
            y=float(z0[1]),
            v=float(z0[2]),
            yaw=float(z0[3]),
            steer=float(z0[4]) if len(z0) > 4 else 0.0,
            vehicle=self.vehicle,
        )
        if len(z0) > 5:
            sim_state.steer_rate = float(z0[5])
        for t_idx in range(T):
            sim_state.update(float(a_seq[t_idx]), float(delta_seq[t_idx]), direct=1.0, dt=self.config.dt)
            z_bar[0, t_idx + 1] = sim_state.x
            z_bar[1, t_idx + 1] = sim_state.y
            z_bar[2, t_idx + 1] = sim_state.v
            z_bar[3, t_idx + 1] = sim_state.yaw

        return z_bar

    def predict_actuators(
        self,
        z0: Sequence[float],
        accel_seq: Sequence[float],
        delta_seq: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict physical acceleration and steering from commanded inputs."""
        accelerations = np.empty(self.config.horizon + 1, dtype=float)
        angles = np.empty(self.config.horizon + 1, dtype=float)
        accelerations[0] = float(z0[6]) if len(z0) > 6 else 0.0
        angles[0] = float(z0[4]) if len(z0) > 4 else 0.0
        sim_state = KinematicState(steer=angles[0], vehicle=self.vehicle)
        sim_state.accel = accelerations[0]
        if len(z0) > 5:
            sim_state.steer_rate = float(z0[5])
        for t_idx, (accel_cmd, delta_cmd) in enumerate(zip(accel_seq, delta_seq)):
            sim_state.update(float(accel_cmd), float(delta_cmd), dt=self.config.dt)
            accelerations[t_idx + 1] = sim_state.accel
            angles[t_idx + 1] = sim_state.steer
        return accelerations, angles

    def solve_qp(
        self,
        z_ref: np.ndarray,
        z_bar: np.ndarray,
        z0: Sequence[float],
        d_bar: Sequence[float],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve a single linearized QP iteration using CVXPY (OSQP / Clarabel)."""
        cfg = self.config
        veh = self.vehicle
        T = cfg.horizon

        z = cvxpy.Variable((cfg.nx, T + 1))
        u = cvxpy.Variable((cfg.nu, T))
        accel_act = cvxpy.Variable(T + 1)
        steer_act = cvxpy.Variable(T + 1)
        steer_rate = cvxpy.Variable(T)

        cost = 0.0
        constraints = [z[:, 0] == np.asarray(z0[: cfg.nx], dtype=float)]
        initial_accel = float(z0[6]) if len(z0) > 6 else 0.0
        initial_steer = float(z0[4]) if len(z0) > 4 else 0.0
        initial_steer_rate = float(z0[5]) if len(z0) > 5 else 0.0
        constraints.extend([
            accel_act[0] == initial_accel,
            accel_act >= -veh.max_accel,
            accel_act <= veh.max_accel,
            steer_act[0] == initial_steer,
            steer_act >= -veh.max_steer,
            steer_act <= veh.max_steer,
            steer_rate == (steer_act[1:] - steer_act[:-1]) / cfg.dt,
            cvxpy.abs(steer_rate) <= veh.max_steer_rate,
        ])
        if T:
            constraints.append(
                cvxpy.abs(steer_rate[0] - initial_steer_rate)
                <= veh.max_steer_accel * cfg.dt
            )

        for t in range(T):
            cost += cvxpy.quad_form(u[:, t], cfg.R)
            cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], cfg.Q)

            A, B, C = calc_linear_discrete_model(
                v=float(z_bar[2, t]),
                phi=float(z_bar[3, t]),
                delta=float(d_bar[t]),
                wheelbase=veh.wheelbase,
                dt=cfg.dt,
            )
            # The bicycle model turns using physical steering, while MPC controls
            # actual actuator outputs rather than their commanded setpoints.
            constraints.append(
                z[:, t + 1]
                == A @ z[:, t]
                + B[:, 0] * 0.5 * (accel_act[t] + accel_act[t + 1])
                + B[:, 1] * 0.5 * (steer_act[t] + steer_act[t + 1])
                + C
            )
            accel_alpha = 1.0 - math.exp(-cfg.dt / max(veh.accel_tau, 1e-6))
            constraints.append(
                accel_act[t + 1]
                == accel_act[t] + accel_alpha * (u[0, t] - accel_act[t])
            )
            actuator_alpha = 1.0 - math.exp(-cfg.dt / max(veh.steer_tau, 1e-6))
            constraints.append(
                steer_act[t + 1]
                == steer_act[t] + actuator_alpha * (u[1, t] - steer_act[t])
            )

            if t < T - 1:
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], cfg.Rd)
                constraints.append(
                    cvxpy.abs(steer_rate[t + 1] - steer_rate[t])
                    <= veh.max_steer_accel * cfg.dt
                )

        cost += cvxpy.quad_form(z_ref[:, T] - z[:, T], cfg.Qf)

        constraints.extend([
            z[2, :] <= veh.max_speed,
            z[2, :] >= veh.min_speed,
            cvxpy.abs(u[0, :]) <= veh.max_accel,
            cvxpy.abs(u[1, :]) <= veh.max_steer,
        ])

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        try:
            prob.solve(solver=cvxpy.OSQP, warm_start=True, verbose=False)
            if prob.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
                prob.solve(solver=cvxpy.CLARABEL, verbose=False)
        except cvxpy.SolverError:
            prob.solve(solver=cvxpy.SCS, verbose=False)

        if prob.status in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE) and z.value is not None and u.value is not None:
            x_opt = np.asarray(z.value[0, :], dtype=float)
            y_opt = np.asarray(z.value[1, :], dtype=float)
            v_opt = np.asarray(z.value[2, :], dtype=float)
            yaw_opt = np.asarray(z.value[3, :], dtype=float)
            a_opt = np.asarray(u.value[0, :], dtype=float)
            delta_opt = np.asarray(u.value[1, :], dtype=float)
            return a_opt, delta_opt, x_opt, y_opt, yaw_opt, v_opt

        return None, None, None, None, None, None

    def step(
        self,
        z_ref: np.ndarray,
        z0: Sequence[float],
        a_old: Optional[Sequence[float]] = None,
        delta_old: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Run iterative sequential quadratic programming (SQP) MPC step."""
        T = self.config.horizon
        a_curr = np.zeros(T, dtype=float) if a_old is None else np.asarray(a_old, dtype=float).copy()
        d_curr = np.zeros(T, dtype=float) if delta_old is None else np.asarray(delta_old, dtype=float).copy()

        x_opt = y_opt = yaw_opt = v_opt = None

        for _ in range(self.config.iter_max):
            z_bar = self.predict_states(z0, a_curr, d_curr)
            _, d_bar = self.predict_actuators(z0, a_curr, d_curr)
            a_prev, d_prev = a_curr.copy(), d_curr.copy()
            a_sol, d_sol, x_sol, y_sol, yaw_sol, v_sol = self.solve_qp(
                z_ref, z_bar, z0, d_bar
            )
            if a_sol is None or d_sol is None:
                break
            a_curr, d_curr = a_sol, d_sol
            x_opt, y_opt, yaw_opt, v_opt = x_sol, y_sol, yaw_sol, v_sol

            du_a = float(np.max(np.abs(a_curr - a_prev)))
            du_d = float(np.max(np.abs(d_curr - d_prev)))
            if max(du_a, du_d) < self.config.du_res:
                break

        return a_curr, d_curr, x_opt, y_opt, yaw_opt, v_opt


# Legacy compatibility aliases
PATH = ReferencePath
pi_2_pi = wrap_angle
