"""2D Planar Vehicle & Dynamic Obstacle Real-Time GIF Animation Generator.

Produces smooth, multi-panel GIF animations (no video dependencies) featuring:
1. Full 2D Scenario Overview with static obstacles, hard & soft inflation boundaries, moving vehicles
   (rendered as 2D planar vehicles), walking pedestrians with velocity-anisotropic inflation zones,
   real-time local bypass candidates, MPC horizon, and the ego 2D planar vehicle.
2. 1:1 True-Scale Follow-Camera Close-Up showing the ego 2D planar vehicle chassis, windshield, front &
   rear axles, all 4 tires (FL, FR, RL, RR), live front wheel steering deflection (Radeinschlag δ),
   front wheel rolling-direction rays, forward/reverse driving direction arrow (Fahrtrichtung), and
   nearby moving obstacles + inflation halos.
3. Live Telemetry HUD + 3 Synchronized Telemetry Plots (Speed & Gear, Radeinschlag δ, Fahrtrichtung ψ)
   with a moving time cursor.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon
import matplotlib.pyplot as plt
import numpy as np

from non_holonomic_vehicle_planning.config import VisualizationConfig
from non_holonomic_vehicle_planning.dynamic_obstacles import DynamicObstacle
from non_holonomic_vehicle_planning.planning.rrt import TreeGrowthSnapshot
from non_holonomic_vehicle_planning.scenario import Scenario
from non_holonomic_vehicle_planning.simulation.runner import SimulationResult
from non_holonomic_vehicle_planning.visualization.plan_view import PlanarVehicleArtist
from non_holonomic_vehicle_planning.visualization.plots import draw_scenario_environment
from non_holonomic_vehicle_planning.visualization.theme import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_EMERALD,
    ACCENT_PURPLE,
    ACCENT_ROSE,
    CARD_COLOR,
    GRID_COLOR,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_dark_theme,
    style_axis,
)


class DynamicObstacleArtist:
    """Matplotlib artist bundle for a moving vehicle or pedestrian + its inflated safety zone."""

    def __init__(self, ax: plt.Axes, obs: DynamicObstacle, zorder_base: int = 4) -> None:
        self.ax = ax
        self.obs = obs
        inf_col = ACCENT_ROSE if obs.kind == "pedestrian" else ACCENT_AMBER

        self.inflation_patch = Polygon(
            np.zeros((4, 2)),
            closed=True,
            fc=inf_col,
            ec=inf_col,
            ls="--",
            lw=1.1,
            alpha=0.16,
            zorder=zorder_base - 1,
        )
        ax.add_patch(self.inflation_patch)
        self.pred_line, = ax.plot([], [], ":", color=inf_col, lw=1.1, alpha=0.75, zorder=zorder_base - 1)

        self.vehicle_artist: Optional[PlanarVehicleArtist] = None
        self.ped_circle: Optional[Circle] = None
        self.ped_vel_line = None

        if obs.kind == "vehicle":
            self.vehicle_artist = PlanarVehicleArtist(
                ax,
                obs.vehicle_params,
                body_color="#f97316",
                scale=1.0,
                zorder_base=zorder_base,
            )
        else:
            self.ped_circle = Circle(
                (obs.start[0], obs.start[1]),
                radius=obs.radius,
                fc=ACCENT_ROSE,
                ec=TEXT_PRIMARY,
                lw=1.3,
                alpha=0.95,
                zorder=zorder_base + 1,
            )
            ax.add_patch(self.ped_circle)
            self.ped_vel_line, = ax.plot([], [], color="#fbbf24", lw=2.0, zorder=zorder_base + 2)

    def update(self, t: float, inflation_cfg, show_inflation: bool = True, show_pred: bool = True) -> None:
        x, y, yaw, v, steer = self.obs.pose_at(t)

        if show_inflation:
            inf_poly = self.obs.inflated_polygon_at(t, inflation_cfg, ego_disk_radius=0.0)
            self.inflation_patch.set_xy(np.asarray(inf_poly.exterior.coords, dtype=float))
        else:
            self.inflation_patch.set_xy(np.zeros((4, 2)))

        if show_pred and abs(v) > 1e-2:
            pred = self.obs.predict_trajectory(t, 0.4, 8)
            self.pred_line.set_data(pred[:, 0], pred[:, 1])
        else:
            self.pred_line.set_data([], [])

        if self.vehicle_artist is not None:
            self.vehicle_artist.update(x, y, yaw, steer, velocity=v)
        elif self.ped_circle is not None and self.ped_vel_line is not None:
            self.ped_circle.center = (x, y)
            if abs(v) > 1e-2:
                vec_len = 1.25
                self.ped_vel_line.set_data(
                    [x, x + vec_len * math.cos(yaw)],
                    [y, y + vec_len * math.sin(yaw)],
                )
            else:
                self.ped_vel_line.set_data([], [])


def _select_frame_indices(n_samples: int, dt: float, fps: int, playback_speed: float, max_frames: int) -> np.ndarray:
    step = max(playback_speed / (fps * dt), 1.0)
    indices = np.arange(0, n_samples, step).round().astype(int)
    indices = np.clip(indices, 0, n_samples - 1)
    if len(indices) == 0 or indices[-1] != n_samples - 1:
        indices = np.append(indices, n_samples - 1)
    hold_count = 8
    max_motion = max(10, max_frames - hold_count)
    if len(indices) > max_motion:
        sub = np.linspace(0, len(indices) - 1, max_motion).round().astype(int)
        indices = indices[sub]
    indices = np.unique(indices)
    return np.concatenate((indices, np.full(hold_count, n_samples - 1, dtype=int)))


def animate_simulation(
    result: SimulationResult,
    output_path: Optional[str | Path] = None,
    fps: int = 15,
    playback_speed: float = 1.2,
    dpi: int = 90,
    closeup_radius: float = 9.0,
    max_frames: int = 140,
    vis_config: Optional[VisualizationConfig] = None,
) -> Path:
    """Render and save a multi-panel 2D Planar Vehicle GIF animation of ``result``."""
    apply_dark_theme()
    if vis_config is not None:
        fps = vis_config.fps
        playback_speed = vis_config.playback_speed
        dpi = vis_config.dpi
        closeup_radius = vis_config.closeup_radius_m
        max_frames = vis_config.max_frames
        show_inf = vis_config.show_inflation_zones
        show_pred = vis_config.show_dynamic_predictions
    else:
        show_inf = True
        show_pred = True

    scenario = result.scenario
    veh = scenario.vehicle
    n_samples = len(result.t)
    dt = float(result.t[1] - result.t[0]) if n_samples > 1 else 0.2
    frame_idx = _select_frame_indices(n_samples, dt, fps, playback_speed, max_frames)

    fig = plt.figure(figsize=(16.0, 9.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.18)
    left = outer[0].subgridspec(2, 2, height_ratios=[1.25, 1.0], width_ratios=[1.15, 0.85], hspace=0.24, wspace=0.18)
    right = outer[1].subgridspec(3, 1, hspace=0.34)

    ax_overview = fig.add_subplot(left[0, :])
    ax_closeup = fig.add_subplot(left[1, 0])
    ax_hud = fig.add_subplot(left[1, 1])
    ax_hud.axis("off")

    ax_v = fig.add_subplot(right[0, 0])
    ax_steer = fig.add_subplot(right[1, 0])
    ax_yaw = fig.add_subplot(right[2, 0])

    # 1. Configure Overview Map
    ax_overview.set_aspect("equal", adjustable="box")
    ax_overview.set_xlim(-1.5, scenario.width + 1.5)
    ax_overview.set_ylim(-1.5, scenario.height + 1.5)
    style_axis(ax_overview, f"2D Scenario & Inflation Overview — {result.scenario_name}", "X [m]", "Y [m]")
    draw_scenario_environment(
        ax_overview,
        scenario,
        draw_start_goal=True,
        show_inflation_zones=show_inf,
        add_inflation_legend=True,
    )

    if len(result.initial_ref_x) >= 2 and result.replan_count > 0:
        ax_overview.plot(result.initial_ref_x, result.initial_ref_y, ":", color=TEXT_MUTED, lw=1.2, alpha=0.6, label="Initial Global Path", zorder=3)
    ax_overview.plot(result.ref_x, result.ref_y, "--", color=TEXT_MUTED, lw=1.4, alpha=0.85, label="Active Reference", zorder=3)

    trail_overview, = ax_overview.plot([], [], color=ACCENT_CYAN, lw=2.2, label="Simulated Trail", zorder=4)
    bypass_overview, = ax_overview.plot([], [], color=ACCENT_EMERALD, lw=2.0, ls="-.", label="Real-Time Bypass", zorder=5)
    ref_horizon_overview, = ax_overview.plot(
        [], [], "--", color="#facc15", marker="+", ms=6, mew=1.5, lw=1.8, label="MPC Target Ref (z_ref)", zorder=15
    )
    horizon_overview, = ax_overview.plot(
        [], [], "-", color="#f43f5e", marker="o", ms=5.0, mec="#ffffff", mew=0.8, lw=2.6, label="MPC Horizon (Pred.)", zorder=16
    )
    target_overview, = ax_overview.plot([], [], marker="x", color=ACCENT_AMBER, ms=8, mew=2.2, ls="None", zorder=16)

    dyn_overview_artists = [
        DynamicObstacleArtist(ax_overview, dyn, zorder_base=4) for dyn in scenario.dynamic_obstacles
    ]
    veh_overview = PlanarVehicleArtist(ax_overview, veh, body_color=ACCENT_CYAN, scale=1.0, zorder_base=6)
    ax_overview.legend(loc="upper left", fontsize=6.8, ncol=3)

    # 2. Configure 1:1 True-Scale Close-Up Follow-Cam
    ax_closeup.set_aspect("equal", adjustable="box")
    style_axis(
        ax_closeup,
        "2D Planar Vehicle Close-Up (Räder, Radeinschlag δ, MPC Horizon & Inflation)",
        "X [m]",
        "Y [m]",
    )
    draw_scenario_environment(ax_closeup, scenario, draw_start_goal=True, show_inflation_zones=show_inf)
    ax_closeup.plot(result.ref_x, result.ref_y, "--", color=TEXT_MUTED, lw=1.5, alpha=0.85, zorder=3)
    trail_closeup, = ax_closeup.plot([], [], color=ACCENT_CYAN, lw=2.4, zorder=4)
    bypass_closeup, = ax_closeup.plot([], [], color=ACCENT_EMERALD, lw=2.0, ls="-.", zorder=5)
    ref_horizon_closeup, = ax_closeup.plot(
        [], [], "--", color="#facc15", marker="+", ms=7, mew=1.6, lw=2.0, zorder=15
    )
    horizon_closeup, = ax_closeup.plot(
        [], [], "-", color="#f43f5e", marker="o", ms=5.8, mec="#ffffff", mew=0.9, lw=3.0, zorder=16
    )

    dyn_closeup_artists = [
        DynamicObstacleArtist(ax_closeup, dyn, zorder_base=4) for dyn in scenario.dynamic_obstacles
    ]
    veh_closeup = PlanarVehicleArtist(ax_closeup, veh, body_color=ACCENT_CYAN, scale=1.0, zorder_base=6)

    # 3. Configure HUD Panel
    hud_box = ax_hud.text(
        0.03,
        0.97,
        "",
        transform=ax_hud.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.6,
        color=TEXT_PRIMARY,
        bbox=dict(
            boxstyle="round,pad=0.50",
            facecolor=CARD_COLOR,
            edgecolor=GRID_COLOR,
            linewidth=1.2,
        ),
    )

    # 4. Configure Telemetry Panels on the Right
    t_arr = result.t
    v_kmh = result.v * 3.6
    steer_deg = np.degrees(result.steer)
    yaw_deg = (np.degrees(result.yaw) + 360.0) % 360.0
    safe_inds = np.clip(result.target_ind, 0, len(result.ref_yaw) - 1)
    ref_yaw_matched_deg = (np.degrees(result.ref_yaw[safe_inds]) + 360.0) % 360.0

    # Panel A: Speed & Gear
    ax_v.axhline(0.0, color=TEXT_MUTED, ls=":", lw=1.0)
    ax_v.plot(t_arr, v_kmh, color=ACCENT_CYAN, lw=1.6)
    ax_v.fill_between(t_arr, 0.0, v_kmh, where=(v_kmh >= 0), color=ACCENT_EMERALD, alpha=0.18, label="Vorwärts (FWD)")
    if np.any(v_kmh < 0):
        ax_v.fill_between(t_arr, 0.0, v_kmh, where=(v_kmh < 0), color=ACCENT_ROSE, alpha=0.22, label="Rückwärts (REV)")
    style_axis(ax_v, "Geschwindigkeit & Fahrtrichtung (v)", None, "v [km/h]")
    ax_v.legend(loc="upper right", fontsize=7.0)

    # Panel B: Steering Angle (Radeinschlag)
    max_steer_deg = math.degrees(veh.max_steer)
    ax_steer.axhline(0.0, color=TEXT_MUTED, ls=":", lw=1.0)
    ax_steer.axhline(max_steer_deg, color=ACCENT_ROSE, ls="--", lw=0.8, alpha=0.6)
    ax_steer.axhline(-max_steer_deg, color=ACCENT_ROSE, ls="--", lw=0.8, alpha=0.6)
    ax_steer.plot(t_arr, steer_deg, color=ACCENT_AMBER, lw=1.6, label="Radeinschlag δ")
    ax_steer.set_ylim(-max_steer_deg * 1.15, max_steer_deg * 1.15)
    style_axis(ax_steer, "Radeinschlag / Lenkwinkel (Front Steer δ)", None, "δ [deg]")

    # Panel C: Vehicle Heading (Fahrtrichtung) & Reference
    ax_yaw.plot(t_arr, ref_yaw_matched_deg, "--", color=TEXT_MUTED, lw=1.2, label="Ref ψ [deg]")
    ax_yaw.plot(t_arr, yaw_deg, color=ACCENT_EMERALD, lw=1.6, label="Fahrtrichtung ψ [deg]")
    style_axis(ax_yaw, "Fahrtrichtung / Gierwinkel (Heading ψ)", "Time t [s]", "ψ [deg]")
    ax_yaw.legend(loc="upper right", fontsize=7.0)

    cursors = [ax.axvline(t_arr[0], color=TEXT_PRIMARY, lw=1.2, zorder=8) for ax in (ax_v, ax_steer, ax_yaw)]

    fig.suptitle(
        f"Non-Holonomic RRT* & Real-Time MPC — {result.scenario_name} (2D Planar Vehicle)",
        fontsize=13,
        fontweight="bold",
        color=TEXT_PRIMARY,
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.06, top=0.91)

    def _update(frame_number: int):
        i = int(frame_idx[frame_number])
        x_i = float(result.x[i])
        y_i = float(result.y[i])
        yaw_i = float(result.yaw[i])
        v_i = float(result.v[i])
        steer_i = float(result.steer[i])
        t_i = float(result.t[i])
        t_ind = int(safe_inds[i])
        is_rev = bool(result.reversing[i])
        vel_dir = -1.0 if is_rev else 1.0

        # Update trails & MPC horizon
        trail_overview.set_data(result.x[:i + 1], result.y[:i + 1])
        trail_closeup.set_data(result.x[:i + 1], result.y[:i + 1])

        if i < len(result.active_bypasses) and result.active_bypasses[i] is not None:
            bx, by = result.active_bypasses[i]  # type: ignore[misc]
            bypass_overview.set_data(bx, by)
            bypass_closeup.set_data(bx, by)
        else:
            bypass_overview.set_data([], [])
            bypass_closeup.set_data([], [])

        hx = result.horizon_x[i]
        hy = result.horizon_y[i]
        horizon_overview.set_data(hx, hy)
        horizon_closeup.set_data(hx, hy)
        if i < len(result.ref_horizon_x) and i < len(result.ref_horizon_y):
            ref_horizon_overview.set_data(result.ref_horizon_x[i], result.ref_horizon_y[i])
            ref_horizon_closeup.set_data(result.ref_horizon_x[i], result.ref_horizon_y[i])
        target_overview.set_data([result.ref_x[t_ind]], [result.ref_y[t_ind]])

        # Update Dynamic Obstacles (moving vehicles & pedestrians + inflation halos)
        for dart_ov, dart_cl in zip(dyn_overview_artists, dyn_closeup_artists):
            dart_ov.update(t_i, scenario.inflation_config, show_inflation=show_inf, show_pred=show_pred)
            dart_cl.update(t_i, scenario.inflation_config, show_inflation=show_inf, show_pred=show_pred)

        # Update Ego 2D planar vehicle in both views
        veh_overview.update(x_i, y_i, yaw_i, steer_i, velocity=vel_dir * max(abs(v_i), 0.1))
        veh_closeup.update(x_i, y_i, yaw_i, steer_i, velocity=vel_dir * max(abs(v_i), 0.1))

        # Pan close-up camera around vehicle center
        ax_closeup.set_xlim(x_i - closeup_radius, x_i + closeup_radius)
        ax_closeup.set_ylim(y_i - closeup_radius, y_i + closeup_radius)

        # Update time cursors
        for cur in cursors:
            cur.set_xdata([t_i, t_i])

        # Update HUD telemetry text
        gear_str = "RÜCKWÄRTS (REV)" if is_rev else "VORWÄRTS (FWD)"
        mode_str = result.planner_modes[i] if i < len(result.planner_modes) else "CRUISE"
        lat_ms = float(result.replan_latencies_ms[i]) if i < len(result.replan_latencies_ms) else 0.0
        dyn_clr = float(result.min_dynamic_clearances_m[i]) if i < len(result.min_dynamic_clearances_m) else float("inf")
        clr_str = f"{dyn_clr:5.2f} m" if math.isfinite(dyn_clr) else "  N/A (static)"
        gx, gy, _ = scenario.goal_pose_rad
        dist_goal = math.hypot(x_i - gx, y_i - gy)
        goal_status = f"ZIEL ERREICHT ✓ ({dist_goal:.2f} m)" if (i == n_samples - 1 and result.reached_goal) else f"Aktiv ({dist_goal:5.1f} m bis Ziel)"
        n_hor = max(0, len(hx) - 1)

        hud_lines = [
            "=== 2D PLANAR VEHICLE & REAL-TIME HUD ===",
            f"Time t          : {t_i:6.2f} s",
            f"Ziel-Status     : {goal_status}",
            f"Planner Mode    : {mode_str} ({lat_ms:.2f} ms)",
            f"MPC Horizont    : N={n_hor} Schritte ({n_hor * dt:.1f} s)",
            f"Fahrtrichtung   : {gear_str}",
            f"Speed v         : {v_i * 3.6:+6.2f} km/h ({v_i:+.2f} m/s)",
            f"Radeinschlag δ  : {math.degrees(steer_i):+6.2f} deg",
            f"Gierwinkel ψ    : {yaw_deg[i]:6.2f} deg",
            f"Querabweichung  : {result.lateral_error[i]:+6.3f} m",
            f"Dyn. Clearance  : {clr_str}",
            "",
            "--- LEGENDE (2D PLANAR, MPC & INFLATION) ---",
            "Pink (o-o-o) : MPC Prädiktionshorizont",
            "Gelb (+-+-+) : MPC Referenz-Horizont (z_ref)",
            "Blaue Räder  : Gelenkte Vorderräder (δ)",
            "Grün / Rot   : Fahrtrichtung (FWD / REV)",
        ]
        hud_box.set_text("\n".join(hud_lines))
        return []

    anim = FuncAnimation(fig, _update, frames=len(frame_idx), interval=1000.0 / fps, blit=False)
    out = Path(output_path) if output_path is not None else Path("plots") / f"{result.scenario_name}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out


def animate_rrt_tree_growth(
    scenario: Scenario,
    scenario_name: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    fps: int = 15,
    dpi: int = 90,
    max_frames: int = 90,
    connector: str = "reeds_shepp",
    iterations: int = 750,
    step_size: float = 13.0,
) -> Path:
    """Render a multi-panel GIF animation showing the RRT / RRT* tree growing step-by-step.

    Displays:
      - 2D Configuration space with static obstacles + Hard & Soft Inflation boundaries.
      - Progressive growth of non-holonomic RRT* branches (Forward Dubins/Reeds-Shepp in cyan,
        Reverse segments in purple, active frontier branch in amber, sampled target point).
      - Live evolution & rewiring of the current Best Path to the Goal (emerald).
      - Final optimal trajectory with 2D Planar Vehicle footprints (wheels, steering δ, heading).
      - Live RRT* Cost Convergence & Node Growth plots + Telemetry HUD.
    """
    from non_holonomic_vehicle_planning.planning.rrt import PlannerConfig, RRTStarPlanner

    apply_dark_theme()
    name = scenario_name or scenario.name

    # If tree_growth_history hasn't been populated yet, run RRTStarPlanner to record live tree growth
    import copy
    scenario = copy.deepcopy(scenario)
    saved_planned_path = scenario.planned_path
    if not scenario.tree_growth_history:
        conn_type = "dubins" if "dubin" in connector.lower() else "reeds_shepp"
        planner = RRTStarPlanner(
            PlannerConfig(
                max_iterations=iterations,
                step_size=step_size,
                connector=conn_type,  # type: ignore[arg-type]
                star=True,
                seed=42,
            )
        )
        try:
            planner.plan(scenario)
        except RuntimeError:
            pass
        if saved_planned_path is not None:
            scenario.set_planned_path(saved_planned_path)

    history: List[TreeGrowthSnapshot] = list(scenario.tree_growth_history)  # type: ignore[arg-type]
    n_steps = len(history)
    if n_steps == 0:
        raise ValueError(f"Scenario '{name}' has no RRT tree growth history to animate.")

    # Select frame indices across the growth history + 12 hold frames at the end showing 2D planar vehicle poses
    growth_frames = min(max_frames - 12, n_steps)
    step_indices = np.linspace(0, n_steps - 1, max(2, growth_frames)).round().astype(int)
    step_indices = np.unique(step_indices)
    hold_frames = 12
    total_frames = len(step_indices) + hold_frames

    fig = plt.figure(figsize=(15.5, 8.8))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.18)
    right = outer[1].subgridspec(3, 1, height_ratios=[1.0, 1.0, 1.05], hspace=0.34)

    ax_map = fig.add_subplot(outer[0])
    ax_cost = fig.add_subplot(right[0, 0])
    ax_nodes = fig.add_subplot(right[1, 0])
    ax_hud = fig.add_subplot(right[2, 0])
    ax_hud.axis("off")

    # 1. Configure 2D Configuration Space Map
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlim(-1.5, scenario.width + 1.5)
    ax_map.set_ylim(-1.5, scenario.height + 1.5)
    style_axis(
        ax_map,
        f"RRT* Tree Growth (Baumwachstum & Inflation) — {name}",
        "X [m]",
        "Y [m]",
    )
    draw_scenario_environment(
        ax_map,
        scenario,
        draw_start_goal=True,
        show_inflation_zones=True,
        add_inflation_legend=True,
    )

    fwd_collection = LineCollection([], colors=ACCENT_CYAN, linewidths=0.95, alpha=0.52, zorder=3, label="RRT* Branch (FWD)")
    rev_collection = LineCollection([], colors=ACCENT_PURPLE, linewidths=1.05, alpha=0.62, zorder=3, label="RRT* Branch (REV)")
    ax_map.add_collection(fwd_collection)
    ax_map.add_collection(rev_collection)

    nodes_scatter = ax_map.scatter([], [], s=10, c="#93c5fd", alpha=0.75, zorder=4, label="Tree Nodes")
    frontier_line, = ax_map.plot([], [], color=ACCENT_AMBER, lw=2.4, alpha=0.95, zorder=6, label="Active Frontier Branch")
    sample_marker, = ax_map.plot([], [], marker="x", color="#fde047", ms=8, mew=2.0, ls="None", zorder=7, label="Sampled Pose")
    best_path_line, = ax_map.plot([], [], color=ACCENT_EMERALD, lw=3.0, alpha=0.95, zorder=7, label="Best Path to Goal")

    # Pre-create a small bank of PlanarVehicleArtists for the final hold frames along the optimal path
    final_path = scenario.planned_path
    ghost_vehicles: List[PlanarVehicleArtist] = []
    ghost_poses: List[tuple] = []
    if final_path is not None and len(final_path.x) >= 4:
        g_idx = np.linspace(0, len(final_path.x) - 1, 7).round().astype(int)
        for idx in g_idx:
            art = PlanarVehicleArtist(ax_map, scenario.vehicle, body_color=ACCENT_CYAN, scale=1.0, zorder_base=8)
            art.update(-999.0, -999.0, 0.0, 0.0, velocity=0.0)
            ghost_vehicles.append(art)
            # Estimate local curvature / steering angle at idx
            steer_est = 0.0
            if 0 < idx < len(final_path.x) - 1:
                ds_loc = max(0.1, math.hypot(float(final_path.x[idx + 1] - final_path.x[idx - 1]), float(final_path.y[idx + 1] - final_path.y[idx - 1])))
                dyaw_loc = math.atan2(
                    math.sin(float(final_path.yaw[idx + 1] - final_path.yaw[idx - 1])),
                    math.cos(float(final_path.yaw[idx + 1] - final_path.yaw[idx - 1])),
                )
                steer_est = float(np.clip(math.atan(scenario.vehicle.wheelbase * (dyaw_loc / ds_loc)), -scenario.vehicle.max_steer, scenario.vehicle.max_steer))
            vel_est = -1.0 if bool(final_path.reversing[idx]) else 1.0
            ghost_poses.append((float(final_path.x[idx]), float(final_path.y[idx]), float(final_path.yaw[idx]), steer_est, vel_est))

    ax_map.legend(loc="upper left", fontsize=7.2, ncol=2)

    # 2. Precompute arrays for telemetry plots
    iters = np.array([h.iteration for h in history], dtype=float)
    nodes_arr = np.array([h.num_nodes for h in history], dtype=float)
    rewires_arr = np.array([h.rewire_count for h in history], dtype=float)
    costs_arr = np.array([h.best_cost if math.isfinite(h.best_cost) else np.nan for h in history], dtype=float)

    all_fwd_lines: List[np.ndarray] = []
    all_rev_lines: List[np.ndarray] = []
    all_node_pts: List[Tuple[float, float]] = []
    for h in history:
        seg = h.added_segment
        pts = np.column_stack((seg.x, seg.y))
        if np.any(seg.direction < 0):
            all_rev_lines.append(pts)
        else:
            all_fwd_lines.append(pts)
        all_node_pts.append((h.node_pose[0], h.node_pose[1]))

    # Configure Cost Convergence Plot
    style_axis(ax_cost, "RRT* Pfadlänge / Kosten-Konvergenz (Best Path Cost)", None, "Path Length [m]")
    ax_cost.set_xlim(0, max(float(iters[-1]), 10.0))
    finite_costs = costs_arr[np.isfinite(costs_arr)]
    if len(finite_costs) > 0:
        c_min, c_max = float(np.min(finite_costs)), float(np.max(finite_costs))
        ax_cost.set_ylim(max(0.0, c_min * 0.88), c_max * 1.12 + 2.0)
    else:
        ax_cost.set_ylim(0.0, 100.0)
    cost_line, = ax_cost.plot([], [], color=ACCENT_EMERALD, lw=2.0, marker=".", ms=4, label="Best Path Length [m]")
    ax_cost.legend(loc="upper right", fontsize=7.5)

    # Configure Tree Growth Plot
    style_axis(ax_nodes, "Baumwachstum: Knoten & RRT*-Rewiring", "RRT* Iteration", "Count")
    ax_nodes.set_xlim(0, max(float(iters[-1]), 10.0))
    ax_nodes.set_ylim(0, max(float(nodes_arr[-1]), float(rewires_arr[-1]), 10.0) * 1.15)
    nodes_line, = ax_nodes.plot([], [], color=ACCENT_CYAN, lw=1.8, label="Tree Nodes")
    rewire_line, = ax_nodes.plot([], [], color=ACCENT_AMBER, lw=1.6, ls="--", label="RRT* Rewires")
    ax_nodes.legend(loc="upper left", fontsize=7.5)

    hud_box = ax_hud.text(
        0.02,
        0.96,
        "",
        transform=ax_hud.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.8,
        color=TEXT_PRIMARY,
        bbox=dict(
            boxstyle="round,pad=0.50",
            facecolor=CARD_COLOR,
            edgecolor=GRID_COLOR,
            linewidth=1.2,
        ),
    )

    fig.suptitle(
        f"Non-Holonomic RRT* Tree Growth & Obstacle Inflation — {name}",
        fontsize=13,
        fontweight="bold",
        color=TEXT_PRIMARY,
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.07, top=0.91)

    def _update_tree(frame_idx: int):
        is_final_hold = frame_idx >= len(step_indices)
        h_idx = int(step_indices[-1] if is_final_hold else step_indices[frame_idx])
        snap = history[h_idx]

        # Collect forward and reverse segments up to h_idx
        fwd_segs: List[np.ndarray] = []
        rev_segs: List[np.ndarray] = []
        for k in range(h_idx + 1):
            seg_k = history[k].added_segment
            pts_k = np.column_stack((seg_k.x, seg_k.y))
            if np.any(seg_k.direction < 0):
                rev_segs.append(pts_k)
            else:
                fwd_segs.append(pts_k)

        fwd_collection.set_segments(fwd_segs)
        rev_collection.set_segments(rev_segs)
        nodes_scatter.set_offsets(np.asarray(all_node_pts[:h_idx + 1], dtype=float))

        if not is_final_hold:
            frontier_line.set_data(snap.added_segment.x, snap.added_segment.y)
            sample_marker.set_data([snap.sampled_xy[0]], [snap.sampled_xy[1]])
            for g_art in ghost_vehicles:
                g_art.update(-999.0, -999.0, 0.0, 0.0, velocity=0.0)
        else:
            frontier_line.set_data([], [])
            sample_marker.set_data([], [])
            # Reveal 2D planar vehicle poses progressively during the hold frames
            n_reveal = min(len(ghost_vehicles), max(1, int((frame_idx - len(step_indices) + 1) * len(ghost_vehicles) / max(1, hold_frames - 2))))
            for g_i, g_art in enumerate(ghost_vehicles):
                if g_i < n_reveal:
                    gx, gy, gyaw, gsteer, gvel = ghost_poses[g_i]
                    g_art.update(gx, gy, gyaw, gsteer, velocity=gvel)
                else:
                    g_art.update(-999.0, -999.0, 0.0, 0.0, velocity=0.0)

        if is_final_hold and final_path is not None:
            best_path_line.set_data(final_path.x, final_path.y)
        elif snap.best_path_xy is not None:
            best_path_line.set_data(snap.best_path_xy[0], snap.best_path_xy[1])
        else:
            best_path_line.set_data([], [])

        cost_line.set_data(iters[:h_idx + 1], costs_arr[:h_idx + 1])
        nodes_line.set_data(iters[:h_idx + 1], nodes_arr[:h_idx + 1])
        rewire_line.set_data(iters[:h_idx + 1], rewires_arr[:h_idx + 1])

        cost_str = f"{snap.best_cost:6.2f} m" if math.isfinite(snap.best_cost) else "Searching..."
        if is_final_hold and final_path is not None:
            cost_str = f"{final_path.length:6.2f} m (Optimal)"
        status_str = "CONVERGED — 2D PLANAR VEHICLE PATH" if is_final_hold else "GROWING RRT* TREE..."

        hud_lines = [
            "=== RRT* TREE GROWTH & INFLATION HUD ===",
            f"Status          : {status_str}",
            f"RRT* Iteration  : {snap.iteration:4d} / {int(iters[-1]):4d}",
            f"Tree Nodes      : {snap.num_nodes:4d} (FWD: {len(fwd_segs)}, REV: {len(rev_segs)})",
            f"RRT* Rewires    : {snap.rewire_count:4d}",
            f"Sampled Target  : ({snap.sampled_xy[0]:5.1f}, {snap.sampled_xy[1]:5.1f}) m",
            f"Best Path Cost  : {cost_str}",
            f"Hard Inflation  : r_hard = {scenario.inflater.hard_inflation_radius:.2f} m (3-Disk)",
            f"Soft Safety Halo: r_soft = {scenario.inflater.soft_inflation_radius:.2f} m",
        ]
        hud_box.set_text("\n".join(hud_lines))
        return []

    anim = FuncAnimation(fig, _update_tree, frames=total_frames, interval=1000.0 / fps, blit=False)
    out = Path(output_path) if output_path is not None else Path("plots") / f"{name}_RRT_Tree.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out

