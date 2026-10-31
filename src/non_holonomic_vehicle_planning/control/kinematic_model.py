"""Kinematic single-track (bicycle) model and discrete-time linearization for MPC."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from non_holonomic_vehicle_planning.vehicle import VehicleParams


class KinematicState:
    """2D planar kinematic vehicle state with PT1 Steering & Drive Actuator Dynamics."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
        v: float = 0.0,
        steer: float = 0.0,
        direct: float = 1.0,
        vehicle: VehicleParams | None = None,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.v = float(v)
        self.steer = float(steer)          # [rad] actual physical front steering angle (delta_act)
        self.steer_cmd = float(steer)      # [rad] commanded steering angle from MPC (delta_cmd)
        self.steer_rate = 0.0              # [rad/s] actual physical steering rate (d_delta/dt)
        self.accel = 0.0                   # [m/s^2] actual longitudinal acceleration (a_act)
        self.accel_cmd = 0.0               # [m/s^2] commanded longitudinal acceleration (a_cmd)
        self.direct = float(direct)
        self.vehicle = vehicle or VehicleParams()

    def update(
        self,
        a: float,
        delta: float,
        direct: float = 1.0,
        dt: float = 0.2,
        use_actuator_dynamics: bool = True,
    ) -> None:
        """Advance vehicle state by one time step ``dt`` [s] including steering & drive actuator dynamics."""
        veh = self.vehicle
        self.direct = float(direct)
        self.steer_cmd = float(np.clip(delta, -veh.max_steer, veh.max_steer))
        self.accel_cmd = float(np.clip(a, -veh.max_accel, veh.max_accel))

        if not use_actuator_dynamics:
            # Ideal kinematic step used inside internal MPC horizon linearization
            max_d_step = veh.max_steer_rate * dt
            d_diff = float(np.clip(self.steer_cmd - self.steer, -max_d_step, max_d_step))
            self.steer_rate = d_diff / max(dt, 1e-4)
            self.steer = float(np.clip(self.steer + d_diff, -veh.max_steer, veh.max_steer))
            self.accel = self.accel_cmd
            self.x += self.v * math.cos(self.yaw) * dt
            self.y += self.v * math.sin(self.yaw) * dt
            self.yaw += (self.v / veh.wheelbase) * math.tan(self.steer) * dt
            self.v = float(np.clip(self.v + self.direct * self.accel * dt, veh.min_speed, veh.max_speed))
            return

        # Sub-stepped PT1 steering servo + rate/acceleration saturation + drive actuator dynamics
        n_sub = max(1, int(round(dt / 0.02)))
        dt_sub = dt / n_sub
        tau_s = max(veh.steer_tau, dt_sub)
        tau_a = max(veh.accel_tau, dt_sub)

        steer_prev = self.steer
        for _ in range(n_sub):
            # 1. Steering servo PT1 + rate & angular-acceleration saturation
            des_rate = (self.steer_cmd - self.steer) / tau_s
            clamped_rate = float(np.clip(des_rate, -veh.max_steer_rate, veh.max_steer_rate))
            max_drate = veh.max_steer_accel * dt_sub
            self.steer_rate = float(
                np.clip(clamped_rate, self.steer_rate - max_drate, self.steer_rate + max_drate)
            )
            self.steer = float(np.clip(self.steer + self.steer_rate * dt_sub, -veh.max_steer, veh.max_steer))

            # 2. Longitudinal drive/brake PT1 lag
            self.accel += (dt_sub / tau_a) * (self.accel_cmd - self.accel)
            self.accel = float(np.clip(self.accel, -veh.max_accel, veh.max_accel))

            # 3. Kinematic bicycle integration using actual physical wheel angle & acceleration
            v_mid = float(np.clip(self.v + 0.5 * self.direct * self.accel * dt_sub, veh.min_speed, veh.max_speed))
            self.x += v_mid * math.cos(self.yaw) * dt_sub
            self.y += v_mid * math.sin(self.yaw) * dt_sub
            self.yaw += (v_mid / veh.wheelbase) * math.tan(self.steer) * dt_sub
            self.v = float(np.clip(self.v + self.direct * self.accel * dt_sub, veh.min_speed, veh.max_speed))

        self.steer_rate = (self.steer - steer_prev) / max(dt, 1e-4)

    @property
    def vector(self) -> np.ndarray:
        """Return state vector [x, y, v, yaw]."""
        return np.array([self.x, self.y, self.v, self.yaw], dtype=float)


def calc_linear_discrete_model(
    v: float,
    phi: float,
    delta: float,
    wheelbase: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute discrete-time linearized state-space matrices (A, B, C) around (v, phi, delta).

    State vector: ``z = [x, y, v, phi]^T``
    Input vector: ``u = [a, delta]^T``
    Dynamics: ``z_{k+1} = A z_k + B u_k + C``
    """
    c_phi = math.cos(phi)
    s_phi = math.sin(phi)
    c_del = max(abs(math.cos(delta)), 1e-4)
    tan_del = math.tan(delta)

    A = np.array([
        [1.0, 0.0, dt * c_phi, -dt * v * s_phi],
        [0.0, 1.0, dt * s_phi,  dt * v * c_phi],
        [0.0, 0.0, 1.0,         0.0],
        [0.0, 0.0, dt * tan_del / wheelbase, 1.0],
    ], dtype=float)

    B = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [dt,  0.0],
        [0.0, dt * v / (wheelbase * c_del * c_del)],
    ], dtype=float)

    C = np.array([
        dt * v * s_phi * phi,
        -dt * v * c_phi * phi,
        0.0,
        -dt * v * delta / (wheelbase * c_del * c_del),
    ], dtype=float)

    return A, B, C


# Legacy alias
Node = KinematicState

