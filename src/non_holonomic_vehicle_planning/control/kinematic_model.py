"""Shared rear-axle bicycle integration and discrete linearization in SI units."""
from __future__ import annotations
import math
import numpy as np
from numpy.typing import ArrayLike
from non_holonomic_vehicle_planning.vehicle import VehicleParams


def discrete_transition(states: ArrayLike, inputs: ArrayLike, vehicle: VehicleParams, dt: float, actuator_mode: str = "full") -> np.ndarray:
    """Integrate [x,y,v,yaw,steer,steer_rate,accel], supporting a batch axis.

    Commands are signed longitudinal acceleration and front-wheel angle.
    The same substep map is used for plant, nonlinear prediction and QP Jacobians.
    Ablation modes: ideal, pt1, rate (PT1+rate), full (+angular acceleration).
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if actuator_mode not in ("ideal", "pt1", "rate", "full"):
        raise ValueError("Unknown actuator mode")
    z = np.asarray(states, dtype=float).copy()
    if z.shape[-1] != 7 or not np.all(np.isfinite(z)) or not np.all(np.isfinite(inputs)):
        raise ValueError("Transition requires finite states and inputs")
    u = np.asarray(inputs, dtype=float)
    a_cmd = np.clip(u[..., 0], -vehicle.max_accel, vehicle.max_accel)
    d_cmd = np.clip(u[..., 1], -vehicle.max_steer, vehicle.max_steer)
    count = max(1, math.ceil(dt / .02))
    h = dt / count
    for _ in range(count):
        if actuator_mode == "ideal":
            z[..., 5] = (d_cmd-z[..., 4])/h
            z[..., 4] = d_cmd
            z[..., 6] = a_cmd
        else:
            rate = (d_cmd-z[..., 4])/max(vehicle.steer_tau, h)
            if actuator_mode in ("rate", "full"):
                rate = np.clip(rate, -vehicle.max_steer_rate, vehicle.max_steer_rate)
            if actuator_mode == "full":
                rate = np.clip(rate, z[..., 5]-vehicle.max_steer_accel*h,
                               z[..., 5]+vehicle.max_steer_accel*h)
                # Brake before mechanical stops: reserve the next substep plus
                # continuous stopping distance, rather than clipping a moving wheel.
                alpha = vehicle.max_steer_accel
                positive_limit = np.sqrt((alpha*h)**2+2*alpha*np.maximum(vehicle.max_steer-z[...,4],0.))-alpha*h
                negative_limit = -np.sqrt((alpha*h)**2+2*alpha*np.maximum(vehicle.max_steer+z[...,4],0.))+alpha*h
                rate = np.clip(rate,negative_limit,positive_limit)
            z[..., 5] = rate
            z[..., 4] = np.clip(z[..., 4]+rate*h, -vehicle.max_steer, vehicle.max_steer)
            z[..., 6] += h/max(vehicle.accel_tau, h)*(a_cmd-z[..., 6])
        v_mid = np.clip(z[..., 2]+.5*z[..., 6]*h, vehicle.min_speed, vehicle.max_speed)
        z[..., 0] += v_mid*np.cos(z[..., 3])*h
        z[..., 1] += v_mid*np.sin(z[..., 3])*h
        z[..., 3] += v_mid/vehicle.wheelbase*np.tan(z[..., 4])*h
        z[..., 2] = np.clip(z[..., 2]+z[..., 6]*h, vehicle.min_speed, vehicle.max_speed)
    return z


def linearize_transition(state: ArrayLike, command: ArrayLike, vehicle: VehicleParams, dt: float, actuator_mode: str = "full") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered numerical Jacobian of the complete discrete actuator/plant map.

    Batched perturbations avoid divergent integration paths between the optimizer
    and simulation. At saturation boundaries this is a local generalized Jacobian.
    """
    z = np.asarray(state, dtype=float)
    u = np.asarray(command, dtype=float)
    eps = 1e-5
    zs = np.repeat(z[None, :], 19, axis=0)
    us = np.repeat(u[None, :], 19, axis=0)
    for j in range(7):
        zs[1+j, j] += eps
        zs[10+j, j] -= eps
    for j in range(2):
        us[8+j, j] += eps
        us[17+j, j] -= eps
    f = discrete_transition(zs, us, vehicle, dt, actuator_mode)
    A = ((f[1:8]-f[10:17])/(2*eps)).T
    B = ((f[8:10]-f[17:19])/(2*eps)).T
    return A, B, f[0]-A@z-B@u


class KinematicState:
    """Rear-axle pose, signed rear-axle velocity and physical actuator feedback."""
    x: float
    y: float
    v: float
    yaw: float
    steer: float
    steer_rate: float
    accel: float
    steer_cmd: float
    accel_cmd: float

    def __init__(self, x=0., y=0., yaw=0., v=0., steer=0., direct=1., vehicle=None):
        self.x, self.y, self.yaw, self.v = map(float, (x,y,yaw,v))
        self.steer = self.steer_cmd = float(steer)
        self.steer_rate = self.accel = self.accel_cmd = 0.
        self.direct = float(direct)
        self.vehicle = vehicle or VehicleParams()

    def update(self, a, delta, direct=1., dt=.2, use_actuator_dynamics=True, actuator_mode=None):
        self.direct = float(direct)
        self.accel_cmd = float(np.clip(a, -self.vehicle.max_accel, self.vehicle.max_accel))
        self.steer_cmd = float(np.clip(delta, -self.vehicle.max_steer, self.vehicle.max_steer))
        mode = actuator_mode or ("full" if use_actuator_dynamics else "ideal")
        z = discrete_transition(self.augmented_vector, [self.direct*self.accel_cmd, self.steer_cmd],
                                self.vehicle, dt, mode)
        self.x, self.y, self.v, self.yaw, self.steer, self.steer_rate, self.accel = map(float, z)

    @property
    def vector(self):
        return np.array([self.x,self.y,self.v,self.yaw])

    @property
    def augmented_vector(self):
        return np.array([self.x,self.y,self.v,self.yaw,self.steer,self.steer_rate,self.accel])


def calc_linear_discrete_model(v, phi, delta, wheelbase, dt):
    """Legacy affine Euler model for physical inputs; controller uses full map."""
    c, s = math.cos(phi), math.sin(phi)
    sec2 = 1/max(math.cos(delta)**2, 1e-8)
    A = np.array([[1,0,dt*c,-dt*v*s],[0,1,dt*s,dt*v*c],
                  [0,0,1,0],[0,0,dt*math.tan(delta)/wheelbase,1.]])
    B = np.array([[0,0],[0,0],[dt,0],[0,dt*v/wheelbase*sec2]])
    C = np.array([dt*v*s*phi,-dt*v*c*phi,0,-dt*v*delta/wheelbase*sec2])
    return A,B,C

Node = KinematicState
