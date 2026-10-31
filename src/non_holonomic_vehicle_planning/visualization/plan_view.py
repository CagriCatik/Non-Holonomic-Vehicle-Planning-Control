"""2D Planar Vehicle Geometry and Renderer.

Renders a complete 2D planar vehicle referenced at its geometric center ``(x, y)``:
- Chassis body polygon and directional front cabin/windshield contour
- Front and rear axles and longitudinal vehicle centerline
- Four wheels (FL, FR, RL, RR) with exact tire length & width and front wheel steering deflection (Radeinschlag)
- Steered front wheel rolling-direction rays to make steering angle visually unmistakable
- Driving direction & heading arrows (Fahrtrichtung: forward vs. reverse gear)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from non_holonomic_vehicle_planning.vehicle import VehicleParams
from non_holonomic_vehicle_planning.visualization.theme import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_EMERALD,
    ACCENT_ROSE,
    TEXT_PRIMARY,
    WHEEL_EDGE,
    WHEEL_FACE,
    WHEEL_FRONT_FACE,
)


def _rot_matrix_row(theta: float) -> np.ndarray:
    """Row-vector 2D rotation matrix (CCW for +theta): ``p_rot = (p.T @ R).T``."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, s], [-s, c]], dtype=float)


def _apply_rot(poly_2xn: np.ndarray, r_row: np.ndarray) -> np.ndarray:
    return (poly_2xn.T @ r_row).T


def _body_polygon_local(p: VehicleParams) -> np.ndarray:
    """2x5 closed polygon of the vehicle chassis in local vehicle-center frame."""
    x_rear = p.rear_bumper_x
    x_front = p.front_bumper_x
    y_half = p.half_width
    return np.array(
        [
            [x_rear, x_rear, x_front, x_front, x_rear],
            [y_half, -y_half, -y_half, y_half, y_half],
        ],
        dtype=float,
    )


def _windshield_polygon_local(p: VehicleParams) -> np.ndarray:
    """2x5 closed trapezoid indicating the front cabin/windshield in local frame."""
    x0 = 0.08 * p.wheelbase
    x1 = 0.44 * p.wheelbase
    y0 = 0.76 * p.half_width
    y1 = 0.62 * p.half_width
    return np.array(
        [
            [x0, x1, x1, x0, x0],
            [-y0, -y1, y1, y0, -y0],
        ],
        dtype=float,
    )


def _wheel_polygon_local(p: VehicleParams) -> np.ndarray:
    """2x5 closed polygon of a single tire centered at (0, 0) in wheel frame."""
    hl = 0.5 * p.wheel_len
    hw = 0.5 * p.wheel_w
    return np.array(
        [
            [hl, -hl, -hl, hl, hl],
            [-hw, -hw, hw, hw, -hw],
        ],
        dtype=float,
    )


def vehicle_geometry_world(
    x: float,
    y: float,
    yaw: float,
    steer_front: float,
    params: VehicleParams,
    steer_rear: float = 0.0,
    scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute all 2D planar vehicle polygons and key points in world coordinates.

    Returns a dictionary containing:
    - ``body``: (2, 5) chassis outline
    - ``windshield``: (2, 5) front cabin trapezoid
    - ``fr``, ``fl``, ``rr``, ``rl``: (2, 5) wheel polygons
    - ``front_axle``: (2, 2) line connecting FL and FR wheel centers
    - ``rear_axle``: (2, 2) line connecting RL and RR wheel centers
    - ``centerline``: (2, 2) line from rear axle center to front axle center
    - ``wheel_centers``: (4, 2) centers of [FL, FR, RL, RR]
    - ``fl_ray``, ``fr_ray``: (2, 2) rays along front wheel rolling direction
    """
    if abs(scale - 1.0) > 1e-6:
        p = VehicleParams(
            wheelbase=params.wheelbase * scale,
            width=params.width * scale,
            rear_overhang=params.rear_overhang * scale,
            front_overhang=params.front_overhang * scale,
            wheel_track=params.wheel_track * scale,
            tire_radius=params.tire_radius * scale,
            tire_width=params.tire_width * scale,
            max_steer=params.max_steer,
            explicit_max_curvature=params.explicit_max_curvature,
        )
    else:
        p = params

    body = _body_polygon_local(p)
    windshield = _windshield_polygon_local(p)
    wheel = _wheel_polygon_local(p)

    x_fa = p.front_axle_x
    x_ra = p.rear_axle_x

    # Apply Ackermann geometry so the inner front wheel steers more sharply.
    steer_fl, steer_fr = p.ackermann_wheel_angles(steer_front)
    r_fl = _rot_matrix_row(steer_fl)
    r_fr = _rot_matrix_row(steer_fr)
    fr = _apply_rot(wheel, r_fr) + np.array([[x_fa], [-p.half_track]])
    fl = _apply_rot(wheel, r_fl) + np.array([[x_fa], [p.half_track]])

    # Rotate rear wheels if 4WS is active
    if abs(steer_rear) > 1e-9:
        r_r = _rot_matrix_row(steer_rear)
        rr = _apply_rot(wheel, r_r) + np.array([[x_ra], [-p.half_track]])
        rl = _apply_rot(wheel, r_r) + np.array([[x_ra], [p.half_track]])
    else:
        rr = wheel + np.array([[x_ra], [-p.half_track]])
        rl = wheel + np.array([[x_ra], [p.half_track]])

    front_axle = np.array([[x_fa, x_fa], [p.half_track, -p.half_track]], dtype=float)
    rear_axle = np.array([[x_ra, x_ra], [p.half_track, -p.half_track]], dtype=float)
    centerline = np.array([[x_ra, x_fa], [0.0, 0.0]], dtype=float)

    # Front wheel direction rays in vehicle frame
    ray_len = 1.45 * p.wheel_len
    cfl, sfl = math.cos(steer_fl), math.sin(steer_fl)
    cfr, sfr = math.cos(steer_fr), math.sin(steer_fr)
    fl_ray = np.array(
        [[x_fa, x_fa + ray_len * cfl], [p.half_track, p.half_track + ray_len * sfl]],
        dtype=float,
    )
    fr_ray = np.array(
        [[x_fa, x_fa + ray_len * cfr], [-p.half_track, -p.half_track + ray_len * sfr]],
        dtype=float,
    )

    # Rotate all parts by vehicle yaw and translate by (x, y)
    r_yaw = _rot_matrix_row(yaw)
    offset = np.array([[x], [y]], dtype=float)

    def to_world(arr: np.ndarray) -> np.ndarray:
        return _apply_rot(arr, r_yaw) + offset

    body_w = to_world(body)
    windshield_w = to_world(windshield)
    fr_w = to_world(fr)
    fl_w = to_world(fl)
    rr_w = to_world(rr)
    rl_w = to_world(rl)
    front_axle_w = to_world(front_axle)
    rear_axle_w = to_world(rear_axle)
    centerline_w = to_world(centerline)
    fl_ray_w = to_world(fl_ray)
    fr_ray_w = to_world(fr_ray)

    wheel_centers = np.array(
        [
            [front_axle_w[0, 0], front_axle_w[1, 0]],  # FL
            [front_axle_w[0, 1], front_axle_w[1, 1]],  # FR
            [rear_axle_w[0, 0], rear_axle_w[1, 0]],    # RL
            [rear_axle_w[0, 1], rear_axle_w[1, 1]],    # RR
        ],
        dtype=float,
    )

    return {
        "body": body_w,
        "windshield": windshield_w,
        "fr": fr_w,
        "fl": fl_w,
        "rr": rr_w,
        "rl": rl_w,
        "front_axle": front_axle_w,
        "rear_axle": rear_axle_w,
        "centerline": centerline_w,
        "wheel_centers": wheel_centers,
        "fl_ray": fl_ray_w,
        "fr_ray": fr_ray_w,
    }


def vehicle_polygons(
    x: float,
    y: float,
    yaw: float,
    steer_front: float,
    params: VehicleParams,
    steer_rear: float = 0.0,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (body, fr, fl, rr, rl) 2x5 polygon arrays in world coordinates."""
    geom = vehicle_geometry_world(x, y, yaw, steer_front, params, steer_rear=steer_rear, scale=scale)
    return geom["body"], geom["fr"], geom["fl"], geom["rr"], geom["rl"]


def draw_heading_arrow(
    ax: plt.Axes,
    x: float,
    y: float,
    theta: float,
    length: float,
    color: str = ACCENT_EMERALD,
    linewidth: float = 2.0,
    alpha: float = 1.0,
    zorder: int = 7,
) -> None:
    """Draw a crisp directional arrow starting at (x, y) pointing along ``theta``."""
    head_ang = math.radians(28.0)
    head_len = 0.35 * length
    x_end = x + length * math.cos(theta)
    y_end = y + length * math.sin(theta)

    ax.plot([x, x_end], [y, y_end], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    for sign in (-1.0, 1.0):
        ang = theta + math.pi + sign * head_ang
        hx = x_end + head_len * math.cos(ang)
        hy = y_end + head_len * math.sin(ang)
        ax.plot([x_end, hx], [y_end, hy], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def draw_vehicle(
    x: float,
    y: float,
    yaw: float,
    steer_front: float,
    params: Optional[VehicleParams] = None,
    *,
    ax: Optional[plt.Axes] = None,
    velocity: float = 0.0,
    body_color: str = ACCENT_CYAN,
    alpha: float = 1.0,
    fill_alpha: float = 0.28,
    draw_axles: bool = True,
    draw_steer_rays: bool = True,
    draw_direction_arrow: bool = True,
    scale: float = 1.0,
    label: Optional[str] = None,
    zorder_base: int = 4,
) -> None:
    """Draw a complete 2D planar vehicle on ``ax`` with body, wheels, axles, steering rays, and direction arrow."""
    target_ax = ax if ax is not None else plt.gca()
    p = params or VehicleParams()
    geom = vehicle_geometry_world(x, y, yaw, steer_front, p, scale=scale)

    body = geom["body"]
    windshield = geom["windshield"]

    # 1. Chassis body fill & outline
    target_ax.fill(
        body[0, :],
        body[1, :],
        facecolor=body_color,
        alpha=fill_alpha * alpha,
        zorder=zorder_base,
    )
    target_ax.plot(
        body[0, :],
        body[1, :],
        color=body_color,
        linewidth=1.8,
        alpha=alpha,
        label=label,
        zorder=zorder_base + 1,
    )

    # 2. Front cabin / windshield indicator
    target_ax.fill(
        windshield[0, :],
        windshield[1, :],
        facecolor=body_color,
        alpha=min(1.0, 1.6 * fill_alpha * alpha),
        zorder=zorder_base + 1,
    )
    target_ax.plot(
        windshield[0, :],
        windshield[1, :],
        color=TEXT_PRIMARY,
        linewidth=0.9,
        alpha=0.65 * alpha,
        zorder=zorder_base + 1,
    )

    # 3. Axles and longitudinal centerline
    if draw_axles:
        fa = geom["front_axle"]
        ra = geom["rear_axle"]
        cl = geom["centerline"]
        target_ax.plot(fa[0, :], fa[1, :], color="#94a3b8", lw=1.4, alpha=0.85 * alpha, zorder=zorder_base + 1)
        target_ax.plot(ra[0, :], ra[1, :], color="#94a3b8", lw=1.4, alpha=0.85 * alpha, zorder=zorder_base + 1)
        target_ax.plot(cl[0, :], cl[1, :], "--", color="#94a3b8", lw=1.1, alpha=0.75 * alpha, zorder=zorder_base + 1)
        target_ax.plot(x, y, marker="o", color=TEXT_PRIMARY, markersize=4.0, alpha=alpha, zorder=zorder_base + 3)

    # 4. Four wheels (rear wheels in light slate, steered front wheels highlighted)
    for w_poly in (geom["rr"], geom["rl"]):
        target_ax.fill(w_poly[0, :], w_poly[1, :], facecolor=WHEEL_FACE, alpha=0.92 * alpha, zorder=zorder_base + 2)
        target_ax.plot(w_poly[0, :], w_poly[1, :], color=WHEEL_EDGE, lw=1.1, alpha=alpha, zorder=zorder_base + 2)

    for w_poly in (geom["fr"], geom["fl"]):
        target_ax.fill(w_poly[0, :], w_poly[1, :], facecolor=WHEEL_FRONT_FACE, alpha=0.95 * alpha, zorder=zorder_base + 2)
        target_ax.plot(w_poly[0, :], w_poly[1, :], color=TEXT_PRIMARY, lw=1.1, alpha=alpha, zorder=zorder_base + 2)

    # 5. Steered front wheel direction rays (Radeinschlag indicators)
    if draw_steer_rays:
        for ray in (geom["fl_ray"], geom["fr_ray"]):
            target_ax.plot(
                ray[0, :],
                ray[1, :],
                color=ACCENT_AMBER,
                lw=1.5,
                alpha=0.9 * alpha,
                zorder=zorder_base + 3,
            )

    # 6. Driving direction & heading arrow (Fahrtrichtung)
    if draw_direction_arrow:
        arrow_len = 0.65 * p.wheelbase * scale
        if velocity < -1e-2:
            draw_heading_arrow(
                target_ax,
                x,
                y,
                yaw + math.pi,
                length=arrow_len,
                color=ACCENT_ROSE,
                linewidth=2.2,
                alpha=alpha,
                zorder=zorder_base + 4,
            )
        else:
            draw_heading_arrow(
                target_ax,
                x,
                y,
                yaw,
                length=arrow_len,
                color=ACCENT_EMERALD,
                linewidth=2.2,
                alpha=alpha,
                zorder=zorder_base + 4,
            )


class PlanarVehicleArtist:
    """Reusable Matplotlib artist bundle for fast frame-by-frame GIF animation updates."""

    def __init__(
        self,
        ax: plt.Axes,
        params: VehicleParams,
        body_color: str = ACCENT_CYAN,
        scale: float = 1.0,
        zorder_base: int = 5,
    ) -> None:
        self.ax = ax
        self.params = params
        self.scale = scale

        self.body_patch = Polygon(
            np.zeros((4, 2)),
            closed=True,
            fc=body_color,
            ec=body_color,
            lw=1.8,
            alpha=0.35,
            zorder=zorder_base,
        )
        self.body_outline, = ax.plot([], [], color=body_color, lw=1.8, zorder=zorder_base + 1)
        self.windshield_patch = Polygon(
            np.zeros((4, 2)),
            closed=True,
            fc=body_color,
            ec=TEXT_PRIMARY,
            lw=0.9,
            alpha=0.55,
            zorder=zorder_base + 1,
        )
        ax.add_patch(self.body_patch)
        ax.add_patch(self.windshield_patch)

        self.front_axle_line, = ax.plot([], [], color="#cbd5e1", lw=1.5, zorder=zorder_base + 1)
        self.rear_axle_line, = ax.plot([], [], color="#cbd5e1", lw=1.5, zorder=zorder_base + 1)
        self.center_line, = ax.plot([], [], "--", color="#cbd5e1", lw=1.1, zorder=zorder_base + 1)
        self.ref_point, = ax.plot([], [], marker="o", color=TEXT_PRIMARY, markersize=4.5, zorder=zorder_base + 4)

        # 4 wheel patches: [FR, FL, RR, RL]
        self.wheel_patches: List[Polygon] = []
        for idx in range(4):
            is_front = idx < 2
            wp = Polygon(
                np.zeros((4, 2)),
                closed=True,
                fc=WHEEL_FRONT_FACE if is_front else WHEEL_FACE,
                ec=TEXT_PRIMARY if is_front else WHEEL_EDGE,
                lw=1.2,
                alpha=0.95,
                zorder=zorder_base + 2,
            )
            ax.add_patch(wp)
            self.wheel_patches.append(wp)

        # Front wheel steering rays (Radeinschlag)
        self.fl_ray_line, = ax.plot([], [], color=ACCENT_AMBER, lw=1.8, zorder=zorder_base + 3)
        self.fr_ray_line, = ax.plot([], [], color=ACCENT_AMBER, lw=1.8, zorder=zorder_base + 3)

        # Heading arrow lines (shaft + 2 head segments) for chassis heading and velocity direction
        self.heading_shaft, = ax.plot([], [], color=ACCENT_EMERALD, lw=2.3, zorder=zorder_base + 4)
        self.heading_hl, = ax.plot([], [], color=ACCENT_EMERALD, lw=2.3, zorder=zorder_base + 4)
        self.heading_hr, = ax.plot([], [], color=ACCENT_EMERALD, lw=2.3, zorder=zorder_base + 4)

    def update(self, x: float, y: float, yaw: float, steer_front: float, velocity: float = 0.0) -> None:
        """Update all vehicle geometry artists for the given pose, steering angle, and speed."""
        geom = vehicle_geometry_world(x, y, yaw, steer_front, self.params, scale=self.scale)

        self.body_patch.set_xy(geom["body"].T)
        self.body_outline.set_data(geom["body"][0, :], geom["body"][1, :])
        self.windshield_patch.set_xy(geom["windshield"].T)

        self.front_axle_line.set_data(geom["front_axle"][0, :], geom["front_axle"][1, :])
        self.rear_axle_line.set_data(geom["rear_axle"][0, :], geom["rear_axle"][1, :])
        self.center_line.set_data(geom["centerline"][0, :], geom["centerline"][1, :])
        self.ref_point.set_data([x], [y])

        for patch, key in zip(self.wheel_patches, ("fr", "fl", "rr", "rl")):
            patch.set_xy(geom[key].T)

        self.fl_ray_line.set_data(geom["fl_ray"][0, :], geom["fl_ray"][1, :])
        self.fr_ray_line.set_data(geom["fr_ray"][0, :], geom["fr_ray"][1, :])

        # Update driving direction arrow (Fahrtrichtung)
        is_reversing = velocity < -1e-2
        arrow_theta = (yaw + math.pi) if is_reversing else yaw
        arrow_col = ACCENT_ROSE if is_reversing else ACCENT_EMERALD
        arrow_len = 0.68 * self.params.wheelbase * self.scale

        x_end = x + arrow_len * math.cos(arrow_theta)
        y_end = y + arrow_len * math.sin(arrow_theta)
        head_ang = math.radians(28.0)
        head_len = 0.35 * arrow_len
        lt = arrow_theta + math.pi - head_ang
        rt = arrow_theta + math.pi + head_ang

        for ln in (self.heading_shaft, self.heading_hl, self.heading_hr):
            ln.set_color(arrow_col)

        self.heading_shaft.set_data([x, x_end], [y, y_end])
        self.heading_hl.set_data([x_end, x_end + head_len * math.cos(lt)], [y_end, y_end + head_len * math.sin(lt)])
        self.heading_hr.set_data([x_end, x_end + head_len * math.cos(rt)], [y_end, y_end + head_len * math.sin(rt)])
