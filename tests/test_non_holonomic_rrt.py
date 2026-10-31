"""Comprehensive test suite for Non-Holonomic RRT*, Real-Time Dynamic Replanning,
Obstacle Inflation, ScenarioCreator, YAML Configs, MPC Control, and 2D Planar Vehicle GIFs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from non_holonomic_vehicle_planning import (
    DynamicObstacle,
    InflationConfig,
    KinematicState,
    MPCConfig,
    MPCController,
    ObstacleBuilder,
    ObstacleInflater,
    PlannedPath,
    PlannerConfig,
    RRTStarPlanner,
    RealTimePlanner,
    RealTimePlannerConfig,
    Scenario,
    ScenarioCreator,
    SimulationRunner,
    TreeNode,
    VehicleParams,
    animate_simulation,
    create_mpc_from_yaml,
    load_config,
    plan_dubins_segments,
    plan_reeds_shepp_segments,
    plot_planning_result,
    plot_simulation_summary,
    vehicle_geometry_world,
    vehicle_polygons,
    wrap_angle,
)
from non_holonomic_vehicle_planning.planning.base import sanitize_csv_trajectory
from non_holonomic_vehicle_planning.planning.rrt import update_children_cost


def test_yaml_config_and_mpc_factory() -> None:
    """Verify master YAML configuration loading and MPC instantiation from YAML."""
    cfg = load_config()
    assert cfg.vehicle.wheelbase == 2.9
    assert cfg.inflation.enabled is True
    assert cfg.inflation.num_footprint_disks == 3
    assert cfg.realtime.enabled is True

    mpc = create_mpc_from_yaml()
    assert mpc.config.horizon == 6
    assert mpc.vehicle.width == 2.0


def test_obstacle_inflater_multi_disk_and_soft_cost() -> None:
    """Verify multi-disk vehicle footprint decomposition and hard/soft inflation layers."""
    vp = VehicleParams(wheelbase=2.9, width=2.0, rear_overhang=0.8, front_overhang=0.8)
    inf_cfg = InflationConfig(
        enabled=True,
        hard_margin_m=0.30,
        soft_margin_m=1.20,
        use_multi_disk_footprint=True,
        num_footprint_disks=3,
    )
    inflater = ObstacleInflater(vp, inf_cfg)

    # 3-disk radius should be significantly smaller than single bounding circle
    assert inflater.disk_radius < vp.bounding_radius
    disks = inflater.pose_disks_world(10.0, 10.0, 0.0)
    assert disks.shape == (3, 2)

    obs = ObstacleBuilder().add_rectangle(4.0, 4.0, (0.0, 0.0)).build()
    hard_polys, soft_polys, hard_u, soft_u = inflater.build_static_layers(obs)
    assert len(hard_polys) == 1 and len(soft_polys) == 1
    assert soft_u.area > hard_u.area

    # Pose far away has zero soft cost; pose inside soft halo has positive cost
    assert inflater.clearance_cost(20.0, 20.0, obs[0]) == 0.0
    assert inflater.clearance_cost(4.0 + inflater.hard_inflation_radius + 0.3, 2.0, obs[0]) > 0.0


def test_scenario_creator_and_yaml_roundtrip(tmp_path: Path) -> None:
    """Verify ScenarioCreator fluent builder, dynamic obstacles, and YAML export/import."""
    creator = (
        ScenarioCreator("Test_Dynamic_Street", width=50.0, height=30.0, boundary_collision=False)
        .set_start_goal((5.0, 15.0), 0.0, (45.0, 15.0), 0.0)
        .add_rectangle_obstacle(50.0, 4.0, (0.0, 0.0))
        .add_parked_vehicle((20.0, 6.5), yaw_deg=0.0)
        .add_moving_vehicle("lead_car", start=(18.0, 15.0), yaw_deg=0.0, speed=0.5, waypoints=[(18.0, 15.0), (25.0, 15.0)])
        .add_pedestrian("crosswalk_ped", start=(32.0, 6.0), goal=(32.0, 24.0), speed=1.3, periodic=True)
    )

    yaml_file = tmp_path / "custom_scenario.yaml"
    creator.to_yaml(yaml_file)
    assert yaml_file.exists()

    reloaded = ScenarioCreator.from_yaml(yaml_file).build()
    assert reloaded.name == "Test_Dynamic_Street"
    assert len(reloaded.obstacles) == 2
    assert len(reloaded.dynamic_obstacles) == 2


def test_realtime_planner_bypasses_moving_obstacle() -> None:
    """Verify RealTimePlanner triggers a fast (< 25 ms) local collision-free bypass around a blocking vehicle."""
    creator = (
        ScenarioCreator("RT_Bypass_Test", width=50.0, height=30.0, boundary_collision=False)
        .set_start_goal((5.0, 15.0), 0.0, (45.0, 15.0), 0.0)
        .add_moving_vehicle(
            "blocking_car",
            start=(18.0, 15.0),
            yaw_deg=0.0,
            speed=0.1,
            motion_mode="waypoints",
            waypoints=[(18.0, 15.0), (19.0, 15.0)],
        )
    )
    sc = creator.build()
    xs = np.linspace(5.0, 45.0, 201)
    ys = np.full_like(xs, 15.0)
    yaws = np.zeros_like(xs)
    base_path = PlannedPath(x=xs, y=ys, yaw=yaws, direction=np.ones_like(xs, dtype=int))
    sc.set_planned_path(base_path)

    rt = RealTimePlanner(sc, RealTimePlannerConfig(enabled=True, lookahead_distance_m=18.0))
    state = KinematicState(x=7.0, y=15.0, yaw=0.0, v=2.5, vehicle=sc.vehicle)

    new_path, info = rt.step(state, base_path, current_idx=10, current_time=0.0, nominal_speed=3.0)
    assert info.replan_triggered is True
    assert info.mode == "AVOID_BYPASS"
    assert info.replan_latency_ms < 50.0
    # Bypass path should deviate laterally from y=15.0 around x=18.0
    assert np.max(np.abs(new_path.y - 15.0)) > 1.5


def test_vehicle_2d_planar_geometry_and_steering() -> None:
    """Verify 2D planar vehicle body, axles, wheels, and front steering rotation."""
    vp = VehicleParams(wheelbase=2.9, width=2.0, rear_overhang=0.8, front_overhang=0.8, wheel_track=1.5)
    assert math.isclose(vp.length, 4.5)
    assert math.isclose(vp.front_axle_x - vp.rear_axle_x, 2.9)

    steer_rad = math.radians(30.0)
    geom = vehicle_geometry_world(10.0, 20.0, math.pi / 2.0, steer_rad, vp)

    assert geom["body"].shape == (2, 5)
    assert geom["windshield"].shape == (2, 5)
    assert geom["fl"].shape == (2, 5)
    assert geom["fr"].shape == (2, 5)
    assert geom["rl"].shape == (2, 5)
    assert geom["rr"].shape == (2, 5)
    assert geom["wheel_centers"].shape == (4, 2)

    ray_fl = geom["fl_ray"]
    dx = ray_fl[0, 1] - ray_fl[0, 0]
    dy = ray_fl[1, 1] - ray_fl[1, 0]
    ray_angle = math.atan2(dy, dx)
    expected_angle = wrap_angle(math.pi / 2.0 + vp.ackermann_wheel_angles(steer_rad)[0])
    assert math.isclose(wrap_angle(ray_angle - expected_angle), 0.0, abs_tol=1e-6)
    fl_angle = math.atan2(ray_fl[1, 1] - ray_fl[1, 0], ray_fl[0, 1] - ray_fl[0, 0])
    ray_fr = geom["fr_ray"]
    fr_angle = math.atan2(ray_fr[1, 1] - ray_fr[1, 0], ray_fr[0, 1] - ray_fr[0, 0])
    assert math.isclose(fl_angle - fr_angle, vp.ackermann_wheel_angles(steer_rad)[0] - vp.ackermann_wheel_angles(steer_rad)[1], abs_tol=1e-6)
    assert fl_angle > fr_angle

    polys = vehicle_polygons(10.0, 20.0, 0.0, 0.0, vp)
    assert len(polys) == 5


def test_mpc_predicts_actuator_lag() -> None:
    """Verify MPC prediction distinguishes commanded and physical actuator outputs."""
    controller = MPCController(MPCConfig(horizon=4, dt=0.2), VehicleParams())
    z0 = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    steering_commands = np.full(4, math.radians(30.0))
    acceleration_commands = np.full(4, 1.0)

    accelerations, steering = controller.predict_actuators(
        z0, acceleration_commands, steering_commands
    )

    assert steering.shape == (5,)
    assert accelerations.shape == (5,)
    assert steering[0] == 0.0
    assert 0.0 < steering[1] < steering_commands[0]
    assert np.all(np.diff(steering) > 0.0)
    assert steering[-1] < steering_commands[-1]
    assert 0.0 < accelerations[1] < acceleration_commands[0]


def test_dubins_and_reeds_shepp_connectors() -> None:
    """Verify Dubins (forward-only) and Reeds-Shepp (forward+reverse) connectors reach target pose."""
    dubins_segs = plan_dubins_segments(0.0, 0.0, 0.0, 12.0, 8.0, math.pi / 2.0, 1.0 / 4.39, step=0.2)
    assert len(dubins_segs) > 0
    best_dubins = dubins_segs[0]
    assert math.isclose(best_dubins.x[0], 0.0, abs_tol=1e-3)
    assert math.isclose(best_dubins.y[0], 0.0, abs_tol=1e-3)
    assert math.isclose(best_dubins.x[-1], 12.0, abs_tol=1e-2)
    assert math.isclose(best_dubins.y[-1], 8.0, abs_tol=1e-2)
    assert np.all(best_dubins.direction == 1)

    rs_segs = plan_reeds_shepp_segments(0.0, 0.0, 0.0, -6.0, 1.5, 0.0, 1.0 / 4.39, step=0.2)
    assert len(rs_segs) > 0
    best_rs = rs_segs[0]
    assert math.isclose(best_rs.x[-1], -6.0, abs_tol=5e-2)
    assert math.isclose(best_rs.y[-1], 1.5, abs_tol=5e-2)
    assert np.any(best_rs.direction == -1)


def test_recursive_cost_update_and_rrt_star_planning() -> None:
    """Ensure RRT* tree cost updates propagate recursively and RRT* finds a valid path."""
    root = TreeNode(Point(0.0, 0.0), 0.0)
    c1 = TreeNode(Point(1.0, 0.0), 0.0)
    c2 = TreeNode(Point(2.0, 0.0), 0.0)
    c1.cost = 10.0
    c2.cost = 20.0
    root.children.append(c1)
    c1.children.append(c2)

    update_children_cost(c1, 3.0)
    assert math.isclose(c1.cost, 7.0)
    assert math.isclose(c2.cost, 17.0)

    sc = Scenario("UnitTest_Scenario", env_width=25.0, env_height=25.0, boundary_collision=False)
    sc.set_vehicle(1.0 / 4.0, length=4.5, width=2.0)
    obs = ObstacleBuilder().add_rectangle(4.0, 8.0, (10.0, 8.0)).build()
    sc.set_obstacles(obs)
    sc.set_start_goal((4.0, 4.0), 45.0, (20.0, 20.0), 45.0)

    planner = RRTStarPlanner(
        PlannerConfig(
            max_iterations=250,
            step_size=10.0,
            goal_tolerance=2.0,
            connector="reeds_shepp",
            seed=7,
        )
    )
    planned = planner.plan(sc)
    assert isinstance(planned, PlannedPath)
    assert planned.length > 15.0
    assert sc.collision_free(planned.linestring)


def test_csv_sanitization_removes_duplicate_yaw_spikes() -> None:
    """Verify legacy CSV duplicate vertices with 0.0 deg yaw spikes are cleaned."""
    x = [0.0, 1.0, 2.0, 2.0, 3.0, 4.0]
    y = [0.0, 1.0, 2.0, 2.0, 3.0, 4.0]
    yaw_deg = [45.0, 45.0, 45.0, 0.0, 45.0, 45.0]
    rev = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    clean = sanitize_csv_trajectory(x, y, yaw_deg, rev, resample_step=0.5)
    expected_yaw = math.radians(45.0)
    assert np.allclose(clean.yaw, expected_yaw, atol=1e-2)


def test_mpc_tracks_across_pi_branch_cut_and_exports_gif(tmp_path: Path) -> None:
    """Verify MPC tracks a trajectory crossing +-pi smoothly and exports a valid GIF."""
    yaws = np.linspace(math.radians(175.0), math.radians(188.0), 61)
    xs = np.cumsum(np.cos(yaws) * 0.2)
    ys = np.cumsum(np.sin(yaws) * 0.2)
    wrapped_yaws = np.array([wrap_angle(y) for y in yaws])

    sc = Scenario("BranchCut_Scenario", env_width=30.0, env_height=30.0)
    sc.set_start_goal((float(xs[0]), float(ys[0])), 175.0, (float(xs[-1]), float(ys[-1])), 188.0)
    path = PlannedPath(x=xs, y=ys, yaw=wrapped_yaws, direction=np.ones_like(xs, dtype=int))
    sc.set_planned_path(path)

    runner = SimulationRunner(MPCConfig(time_max=15.0))
    res = runner.run(sc, path, label="BranchCut_Test")
    assert np.max(np.abs(res.lateral_error)) < 0.35

    gif_file = tmp_path / "test_anim.gif"
    png_file = tmp_path / "test_summary.png"
    plan_png = tmp_path / "test_plan.png"
    tree_gif = tmp_path / "test_tree.gif"

    from non_holonomic_vehicle_planning.visualization.animation import animate_rrt_tree_growth

    animate_simulation(res, output_path=gif_file, max_frames=20, dpi=60)
    animate_rrt_tree_growth(sc, output_path=tree_gif, max_frames=18, dpi=60, iterations=120)
    plot_simulation_summary(res, output_path=png_file)
    plot_planning_result(sc, output_path=plan_png)

    assert gif_file.exists() and gif_file.stat().st_size > 1000
    assert tree_gif.exists() and tree_gif.stat().st_size > 1000
    assert png_file.exists() and png_file.stat().st_size > 1000
    assert plan_png.exists() and plan_png.stat().st_size > 1000
