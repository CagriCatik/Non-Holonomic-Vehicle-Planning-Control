"""Linear Model Predictive Controller (MPC) for non-holonomic trajectory tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import List, Optional, Sequence, Tuple

import cvxpy
import numpy as np

from non_holonomic_vehicle_planning.control.kinematic_model import KinematicState, discrete_transition, linearize_transition
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
    du_res: float = 0.08  # acceleration convergence [m/s²]
    steer_du_res: float = .005  # steering convergence [rad]
    search_window: int = 15
    heading_stop: float = math.radians(5.)
    max_lateral_accel: float | None = None  # opt-in, m/s²; no invented vehicle limit

    def __post_init__(self):
        if self.nx != 4 or self.nu != 2:
            raise ValueError("Public tracking state/input dimensions must be 4/2")
        for name in ("horizon", "iter_max", "search_window"):
            if not isinstance(getattr(self,name),(int,np.integer)) or isinstance(getattr(self,name),bool) or getattr(self,name) < 1:
                raise ValueError(name+" must be positive")
        for name in ("dt", "time_max", "target_speed", "dist_stop", "speed_stop", "du_res", "steer_du_res", "heading_stop"):
            value = getattr(self,name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name+" must be finite and positive")
        if self.max_lateral_accel is not None and (not math.isfinite(self.max_lateral_accel) or self.max_lateral_accel <= 0):
            raise ValueError("max_lateral_accel must be positive or None")
        for name,size in (("Q",4),("Qf",4),("R",2),("Rd",2)):
            matrix = np.asarray(getattr(self,name),dtype=float)
            if matrix.shape != (size,size) or not np.all(np.isfinite(matrix)):
                raise ValueError(name+" has invalid shape or values")
            if not np.allclose(matrix,matrix.T) or np.linalg.eigvalsh(matrix).min() < 0:
                raise ValueError(name+" must be symmetric positive semidefinite")
            setattr(self,name,matrix)


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
    """Legacy wrapper; use trajectory.profile_speed with explicit limits."""
    from non_holonomic_vehicle_planning.trajectory import profile_speed
    return profile_speed(cx,cy,cyaw,reversing,target_speed)


STEER_ALIGN_TOLERANCE = math.radians(2.)  # [rad] Wheel angle error accepted before leaving a cusp.


class ReferencePath:
    """Reference trajectory with arc-length parameterization and cusp-aware indexing."""

    def __init__(
        self,
        cx: Sequence[float],
        cy: Sequence[float],
        cyaw: Sequence[float],
        reversing: Optional[Sequence[bool | float]] = None,
    ) -> None:
        self.cx = np.array(cx, dtype=float, copy=True)
        self.cy = np.array(cy, dtype=float, copy=True)
        self.cyaw = smooth_reference_yaw(cyaw)
        self.length = len(self.cx)
        if self.length < 2 or len(self.cy) != self.length or len(self.cyaw) != self.length:
            raise ValueError("Reference requires at least two aligned points")
        if not np.all(np.isfinite(np.r_[self.cx,self.cy,self.cyaw])):
            raise ValueError("Reference values must be finite")
        if reversing is None:
            self.reversing = np.zeros(self.length, dtype=bool)
        else:
            self.reversing = np.asarray(reversing, dtype=float) > 0.5

        if len(self.reversing) != self.length:
            raise ValueError("Reversing flags must match reference length")
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
        self.progress_s = 0.
        self.projected_yaw = float(self.cyaw[0])
        self.steering_hold: Optional[float] = None

    @property
    def current_segment_end(self) -> int:
        return self.cusp_ends[min(self.active_cusp_ptr, len(self.cusp_ends) - 1)]

    def _cusp_steering_target(self, state, speed_stop):
        """Wheel angle to reach at standstill before driving off a cusp, else None.

        Short gear sections are consumed before a lagging wheel could catch up,
        and the kinematic model allows steering without motion.
        """
        if self.active_cusp_ptr == 0 or abs(state.v) >= speed_stop:
            return None
        start = self.cusp_ends[self.active_cusp_ptr-1]+1
        stop = min(self.current_segment_end,start+5)
        distance = self.s[stop]-self.s[start]
        if distance < 1e-6 or self.progress_s > self.s[start]+.05:
            return None
        sign = -1. if self.reversing[start] else 1.
        target = math.atan(state.vehicle.wheelbase*sign*(self.cyaw[stop]-self.cyaw[start])/distance)
        return target if abs(state.steer-target) > STEER_ALIGN_TOLERANCE else None

    def nearest_index(self, state, search_window=15, speed_stop=.14):
        """Project on the active gear segment, retaining monotone local progress."""
        seg_start = 0 if self.active_cusp_ptr == 0 else self.cusp_ends[self.active_cusp_ptr-1]+1
        end = self.current_segment_end
        lo = max(seg_start,self.ind_old-1)
        hi = min(end, self.ind_old+search_window)
        if lo < hi:
            indices = np.arange(lo,hi)
            xy = np.column_stack((self.cx,self.cy))
            ab = xy[indices+1]-xy[indices]
            denom = np.sum(ab*ab,axis=1)
            fraction = np.clip(np.sum((np.array([state.x,state.y])-xy[indices])*ab,axis=1)
                               /np.maximum(denom,1e-12),0,1)
            positions = xy[indices]+fraction[:,None]*ab
            best = int(np.argmin(np.sum((positions-[state.x,state.y])**2,axis=1)))
            k,f = int(indices[best]),float(fraction[best])
            projected_s = self.s[k]+f*(self.s[k+1]-self.s[k])
            self.progress_s = max(self.progress_s,float(projected_s))
            self.progress_s = min(self.progress_s,float(self.s[end]))
        else:
            self.progress_s = max(self.progress_s,float(self.s[end]))
        self.ind_old = min(end,max(seg_start,int(np.searchsorted(self.s,self.progress_s,side="right")-1)))
        distance = math.hypot(state.x-self.cx[end],state.y-self.cy[end])
        if self.active_cusp_ptr < len(self.cusp_ends)-1 and distance < .35 and abs(state.v) < speed_stop:
            self.active_cusp_ptr += 1
            self.ind_old = end+1
            self.progress_s = float(self.s[self.ind_old])
        start = 0 if self.active_cusp_ptr == 0 else self.cusp_ends[self.active_cusp_ptr-1]+1
        stop = self.current_segment_end
        ss = self.s[start:stop+1]
        px = np.interp(self.progress_s,ss,self.cx[start:stop+1])
        py = np.interp(self.progress_s,ss,self.cy[start:stop+1])
        self.projected_yaw = float(np.interp(self.progress_s,ss,self.cyaw[start:stop+1]))
        error = -(state.x-px)*math.sin(self.projected_yaw)+(state.y-py)*math.cos(self.projected_yaw)
        return self.ind_old,float(error)

    def calc_ref_trajectory(
        self,
        state: KinematicState,
        speed_profile: Sequence[float],
        cfg: MPCConfig,
    ) -> Tuple[np.ndarray, int, float]:
        """Construct the (4 x (T+1)) reference state horizon aligned with ``state.yaw``."""
        z_ref = np.zeros((cfg.nx,cfg.horizon+1))
        ind,lat_err = self.nearest_index(state,cfg.search_window,cfg.speed_stop)
        self.steering_hold = self._cusp_steering_target(state,cfg.speed_stop)
        end = self.current_segment_end
        start = 0 if self.active_cusp_ptr == 0 else self.cusp_ends[self.active_cusp_ptr-1]+1
        section = slice(start,end+1)
        ss = self.s[section]
        yaw_shift = round((state.yaw-self.projected_yaw)/(2*math.pi))*2*math.pi
        progress = self.progress_s
        profile = np.asarray(speed_profile)
        reference_speed = abs(state.v)
        for i in range(cfg.horizon+1):
            z_ref[0,i] = np.interp(progress,ss,self.cx[section])
            z_ref[1,i] = np.interp(progress,ss,self.cy[section])
            z_ref[2,i] = np.interp(progress,ss,profile[section])
            z_ref[3,i] = np.interp(progress,ss,self.cyaw[section])+yaw_shift
            # A time reference cannot accelerate faster than the physical vehicle.
            desired_speed = abs(z_ref[2,i])
            next_speed = reference_speed + float(np.clip(desired_speed-reference_speed,
                                  -state.vehicle.max_accel*cfg.dt,state.vehicle.max_accel*cfg.dt))
            z_ref[2,i] = math.copysign(reference_speed,z_ref[2,i])
            progress = min(float(self.s[end]),progress+.5*(reference_speed+next_speed)*cfg.dt)
            reference_speed = next_speed
        if self.steering_hold is not None:
            # Any pose error here would pull the vehicle off before the wheel is aligned.
            z_ref[:] = np.array([[state.x],[state.y],[0.],[state.yaw]])
        return z_ref,ind,lat_err


class MPCController:
    """Iterative Linear Model Predictive Controller for 2D planar vehicle tracking."""

    def __init__(
        self,
        config: Optional[MPCConfig] = None,
        vehicle: Optional[VehicleParams] = None,
    ) -> None:
        self.config = config or MPCConfig()
        self.vehicle = vehicle or VehicleParams()
        if self.vehicle.pose_reference != "rear_axle":
            raise ValueError("Control requires rear-axle pose coordinates")
        self.previous_command = np.zeros(2)
        self.last_solver_status = "not_run"
        self.last_solver_iterations = 0
        self.last_solver_runtime_s = 0.
        self.last_control_runtime_s = 0.
        self.last_fallback = False
        self.last_sqp_status = "not_run"
        self._steering_target: Optional[float] = None

    @staticmethod
    def _augment(z0):
        z = np.zeros(7)
        z[:len(z0)] = z0
        return z

    def _rollout(self, z0, a_seq, delta_seq):
        states = np.zeros((7, self.config.horizon+1))
        states[:, 0] = self._augment(z0)
        for k in range(self.config.horizon):
            states[:, k+1] = discrete_transition(states[:, k], [a_seq[k],delta_seq[k]],
                                                 self.vehicle, self.config.dt)
        return states

    def predict_states(
        self,
        z0: Sequence[float],
        a_seq: Sequence[float],
        delta_seq: Sequence[float],
    ) -> np.ndarray:
        """Roll out the nonlinear kinematic bicycle model over the horizon ``T``."""
        return self._rollout(z0, a_seq, delta_seq)[:4]

    def predict_actuators(
        self,
        z0: Sequence[float],
        accel_seq: Sequence[float],
        delta_seq: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict physical acceleration and steering from commanded inputs."""
        states = self._rollout(z0, accel_seq, delta_seq)
        return states[6], states[4]

    def _feedforward(self,reference,z0):
        if self._steering_target is not None:
            return np.vstack((np.diff(reference[2])/self.config.dt,np.full(self.config.horizon,self._steering_target)))
        ds = np.hypot(np.diff(reference[0]),np.diff(reference[1]))
        dyaw = np.diff(reference[3])
        steering = np.empty(self.config.horizon)
        curvature = math.tan(float(z0[4]))/self.vehicle.wheelbase if len(z0)>4 else 0.
        for k in range(self.config.horizon):
            if ds[k]>1e-6:
                direction = -1. if reference[2,k]+reference[2,k+1]<0 else 1.
                curvature = dyaw[k]/ds[k]*direction
            steering[k] = math.atan(self.vehicle.wheelbase*curvature)
        return np.vstack((np.diff(reference[2])/self.config.dt,steering))

    def solve_qp(
        self,
        z_ref: np.ndarray,
        z_bar: np.ndarray,
        z0: Sequence[float],
        d_bar: Sequence[float],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve a single linearized QP iteration using CVXPY (OSQP / Clarabel)."""
        cfg, veh = self.config, self.vehicle
        T = cfg.horizon
        z = cvxpy.Variable((7, T+1))
        u = cvxpy.Variable((2, T))
        commands = getattr(self,"_nominal_commands",np.vstack((np.zeros(T),np.asarray(d_bar)[:T])))
        nominal = getattr(self,"_nominal_states",None)
        if nominal is None:
            nominal = self._rollout(z0,commands[0],commands[1])
        constraints = [z[:, 0] == self._augment(z0)]
        feedforward = self._feedforward(z_ref,z0)
        cost = 0.
        for t in range(T):
            A, B, C = linearize_transition(nominal[:, t], commands[:, t], veh, cfg.dt)
            constraints.append(z[:, t+1] == A@z[:, t]+B@u[:, t]+C)
            cost += cvxpy.quad_form(z[:4,t]-z_ref[:,t], cfg.Q)
            cost += cvxpy.quad_form(u[:,t]-feedforward[:,t], cfg.R)
            previous = self.previous_command if t == 0 else u[:,t-1]
            cost += cvxpy.quad_form(u[:,t]-previous, cfg.Rd)
        cost += cvxpy.quad_form(z[:4,T]-z_ref[:,T], cfg.Qf)
        constraints.extend([
            z[2] <= veh.max_speed, z[2] >= veh.min_speed,
            cvxpy.abs(z[4]) <= veh.max_steer,
            cvxpy.abs(z[5]) <= veh.max_steer_rate,
            cvxpy.abs(z[6]) <= veh.max_accel,
            cvxpy.abs(z[5,1:]-z[5,:-1]) <= veh.max_steer_accel*cfg.dt,
            cvxpy.abs(u[0]) <= veh.max_accel,
            cvxpy.abs(u[1]) <= veh.max_steer,
            # Keep each local QP inside the actuator's command sensitivity region.
            cvxpy.abs(u[1]-commands[1]) <= .5*veh.steer_tau*veh.max_steer_rate,
            cvxpy.abs(u[0]-commands[0]) <= veh.max_accel,

        ])
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        self.last_solver_status = "failed"
        for solver in (cvxpy.OSQP, cvxpy.CLARABEL):
            try:
                prob.solve(solver=solver, warm_start=True, verbose=False)
            except cvxpy.SolverError:
                continue
            self.last_solver_status = str(prob.status)
            stats = prob.solver_stats
            self.last_solver_iterations += int(stats.num_iters or 0)
            self.last_solver_runtime_s += float(stats.solve_time or 0.)
            if prob.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
                continue
            if z.value is None or u.value is None or not np.all(np.isfinite(u.value)):
                continue
            violation = max(float(np.max(c.violation())) for c in constraints)
            if violation > 1e-3:
                self.last_solver_status = "rejected_residual"
                continue
            return (np.clip(u.value[0], -veh.max_accel, veh.max_accel),
                    np.clip(u.value[1], -veh.max_steer, veh.max_steer),
                    z.value[0].copy(),z.value[1].copy(),z.value[3].copy(),z.value[2].copy())
        return (None,)*6

    def _merit(self,states,commands,reference):
        error = states[:4]-reference
        changes = np.diff(np.column_stack((self.previous_command,commands)),axis=1)
        input_error = commands-self._feedforward(reference,states[:,0])
        cfg = self.config
        return (float(np.sum(error[:,:-1]*(cfg.Q@error[:,:-1])))
                +float(error[:,-1]@cfg.Qf@error[:,-1])
                +float(np.sum(input_error*(cfg.R@input_error)))
                +float(np.sum(changes*(cfg.Rd@changes))))

    def step(
        self,
        z_ref: np.ndarray,
        z0: Sequence[float],
        a_old: Optional[Sequence[float]] = None,
        delta_old: Optional[Sequence[float]] = None,
        steering_target: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Run iterative sequential quadratic programming (SQP) MPC step."""
        self._steering_target = steering_target
        z0 = np.asarray(z0,dtype=float)
        if z0.ndim != 1 or not 4 <= len(z0) <= 7 or not np.all(np.isfinite(z0)):
            raise ValueError("Feedback must contain 4..7 finite state values")
        started = time.perf_counter()
        T = self.config.horizon
        def shift(old, measured):
            # Without a usable previous plan, hold the measured actuator state.
            if old is None:
                return np.full(T, measured)
            arr = np.asarray(old, dtype=float)
            if arr.shape != (T,) or not np.all(np.isfinite(arr)):
                return np.full(T, measured)
            return np.r_[arr[1:], arr[-1]]
        feedback = self._augment(z0)
        a_curr, d_curr = shift(a_old, feedback[6]), shift(delta_old, feedback[4])
        self.last_solver_iterations = 0
        self.last_solver_runtime_s = 0.
        self.last_sqp_iterations = 0
        self.last_fallback = False
        self.last_solver_status = "not_run"
        self.last_sqp_status = "max_iterations"
        x_opt = y_opt = yaw_opt = v_opt = None
        accepted = False
        for iteration in range(self.config.iter_max):
            self.last_sqp_iterations = iteration+1
            self._nominal_states = self._rollout(z0, a_curr, d_curr)
            self._nominal_commands = np.vstack((a_curr,d_curr))
            sol = self.solve_qp(z_ref,self._nominal_states[:4],z0,self._nominal_states[4])
            if sol[0] is None:
                self.last_sqp_status = "solver_failure"
                break
            a_sol,d_sol,x_opt,y_opt,yaw_opt,v_opt = sol
            previous_a,previous_d = a_curr.copy(),d_curr.copy()
            accepted_step = False
            # A QP decrease must also improve the saturated nonlinear rollout.
            merit = self._merit(self._nominal_states,self._nominal_commands,z_ref)
            for alpha in (1.,.5,.25,.1):
                trial_a = a_curr+alpha*(a_sol-a_curr)
                trial_d = d_curr+alpha*(d_sol-d_curr)
                trial_states = self._rollout(z0,trial_a,trial_d)
                if self._merit(trial_states,np.vstack((trial_a,trial_d)),z_ref) <= merit+1e-8:
                    a_curr,d_curr = trial_a,trial_d
                    accepted_step = True
                    break
            accepted = True
            if not accepted_step:
                self.last_sqp_status = "line_search_stalled"
                break
            if (np.max(abs(a_curr-previous_a)) < self.config.du_res
                    and np.max(abs(d_curr-previous_d)) < self.config.steer_du_res):
                self.last_sqp_status = "converged"
                break
        if not accepted:
            # Signed braking and hold measured angle; plant enforces actuator limits.
            from scipy.optimize import brentq
            fallback_state = self._augment(z0)
            a_curr = np.zeros(T)
            d_curr = np.full(T,float(fallback_state[4]))
            for k in range(T):
                def next_velocity(command):
                    return float(discrete_transition(fallback_state,[command,d_curr[k]],
                                                      self.vehicle,self.config.dt)[2])
                low,high = -self.vehicle.max_accel,self.vehicle.max_accel
                vl,vh = next_velocity(low),next_velocity(high)
                command = brentq(next_velocity,low,high,xtol=1e-7) if vl*vh<=0 and vl!=vh else (low if abs(vl)<abs(vh) else high)
                a_curr[k] = command
                fallback_state = discrete_transition(fallback_state,[command,d_curr[k]],self.vehicle,self.config.dt)
            self.last_fallback = True
            self.last_sqp_status = "fallback"
            self.last_solver_status = "fallback:"+self.last_solver_status
        if accepted:
            prediction = self._rollout(z0,a_curr,d_curr)
            x_opt,y_opt,v_opt,yaw_opt = (prediction[i].copy() for i in range(4))
        self.previous_command = np.array([a_curr[0],d_curr[0]])
        self.last_control_runtime_s = time.perf_counter()-started
        return a_curr,d_curr,x_opt,y_opt,yaw_opt,v_opt


# Legacy compatibility aliases
PATH = ReferencePath
pi_2_pi = wrap_angle
