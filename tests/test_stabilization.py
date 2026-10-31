"""Regressions for state, actuator and failure handling found in the audit."""
import math
import numpy as np
import pytest
from non_holonomic_vehicle_planning.control.kinematic_model import KinematicState
from non_holonomic_vehicle_planning.control.mpc import MPCConfig, MPCController, ReferencePath
from non_holonomic_vehicle_planning.vehicle import VehicleParams


def test_rollout_preserves_measured_acceleration():
    cfg = MPCConfig(horizon=2)
    ctl = MPCController(cfg)
    z0 = [0, 0, 2, 0, .1, .2, 1.2]
    a = np.zeros(2); d = np.full(2, .1)
    predicted = ctl.predict_states(z0, a, d)
    state = KinematicState(v=2, steer=.1)
    state.steer_rate = .2; state.accel = 1.2
    state.update(0, .1, dt=cfg.dt)
    np.testing.assert_allclose(predicted[:, 1], state.vector, atol=1e-12)


def test_terminal_servo_rate_is_not_period_average():
    state = KinematicState()
    state.update(0, .5, dt=.2)
    # At this early point acceleration saturation is active throughout:
    # terminal rate = max_accel * dt, versus a smaller average over the interval.
    assert state.steer_rate == pytest.approx(state.vehicle.max_steer_accel * .2)


def test_failed_solve_brakes_instead_of_reusing_stale_acceleration(monkeypatch):
    ctl = MPCController(MPCConfig(horizon=3))
    monkeypatch.setattr(ctl, 'solve_qp', lambda *args: (None,) * 6)
    a, d, *_ = ctl.step(np.zeros((4, 4)), [0, 0, 2, 0, .2, 0, 0], np.ones(3), np.full(3, -.5))
    assert np.all(a < 0)
    assert d[0] == pytest.approx(.2)


def test_cusp_cannot_switch_gear_while_moving():
    ref = ReferencePath([0, 1, 2, 1, 0], [0]*5, [0]*5, [0, 0, 0, 1, 1])
    ref.ind_old = 1
    ref.nearest_index(KinematicState(x=1.9, v=1))
    assert ref.active_cusp_ptr == 0


def test_cte_uses_reference_normal():
    ref = ReferencePath([0, 1, 2], [0]*3, [0]*3)
    _, cte = ref.nearest_index(KinematicState(x=.2, y=1, yaw=math.pi/2))
    assert cte == pytest.approx(1)


def test_full_discrete_linearization_reproduces_nominal_and_small_perturbation():
    from non_holonomic_vehicle_planning.control.kinematic_model import discrete_transition, linearize_transition
    veh = VehicleParams()
    z = np.array([1,2,2,.3,.1,.05,.2]);u=np.array([.3,.12])
    A,B,C = linearize_transition(z,u,veh,.2)
    expected=discrete_transition(z,u,veh,.2)
    np.testing.assert_allclose(A@z+B@u+C,expected,atol=1e-10)
    dz=np.array([.001,-.001,.001,.0001,.0001,.0001,.001])
    np.testing.assert_allclose(A@(z+dz)+B@u+C,discrete_transition(z+dz,u,veh,.2),atol=2e-7)


def test_reference_arrays_do_not_alias_caller():
    x=np.arange(4.)
    ref=ReferencePath(x,np.zeros(4),np.zeros(4))
    x[1]=99
    assert ref.cx[1] == 1


def test_fractional_reference_progress_and_interpolation():
    cfg=MPCConfig(horizon=2,dt=.1,target_speed=1)
    ref=ReferencePath([0,1,2],[0]*3,[0]*3)
    z,_,_=ref.calc_ref_trajectory(KinematicState(x=.45,v=1),np.ones(3),cfg)
    assert ref.progress_s == pytest.approx(.45)
    np.testing.assert_allclose(z[0],[.45,.55,.65])


def test_speed_envelope_obeys_acceleration_and_lateral_limit():
    from non_holonomic_vehicle_planning.trajectory import profile_speed,geometry
    theta=np.linspace(0,1,101)
    x=5*np.sin(theta);y=5*(1-np.cos(theta))
    speed=profile_speed(x,y,theta,np.zeros(101),4,max_accel=1,max_lateral_accel=.8)
    s,k=geometry(x,y,theta,np.zeros(101))
    assert np.max(speed**2*abs(k)) <= .8+1e-9
    assert np.max(abs(np.diff(speed**2))-2*np.diff(s)) < 1e-8
    assert speed[-1] == 0


def test_reverse_speed_profile_stops_at_cusp_and_preserves_direction():
    from non_holonomic_vehicle_planning.trajectory import profile_speed
    speed=profile_speed([0,1,2,1,0],[0]*5,[0]*5,[0,0,0,1,1],2,max_accel=1)
    assert speed[2] == 0 and speed[3]<0 and speed[-1] == 0
    assert abs(speed[3]) <= math.sqrt(2)


@pytest.mark.parametrize('kwargs',[{'dt':0},{'horizon':0},{'Q':np.eye(3)},{'Rd':np.diag([1,-1])}])
def test_invalid_mpc_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError): MPCConfig(**kwargs)


def test_impossible_planner_curvature_is_rejected():
    with pytest.raises(ValueError): VehicleParams(explicit_max_curvature=1.)


def test_rear_axle_geometry_and_center_obstacle_geometry():
    veh=VehicleParams()
    assert veh.rear_axle_x == 0
    assert veh.front_axle_x == veh.wheelbase
    footprint=veh.footprint_polygon(0,0,0)
    assert footprint.bounds[0] == pytest.approx(-veh.rear_overhang)
    assert footprint.bounds[2] == pytest.approx(veh.wheelbase+veh.front_overhang)
    from non_holonomic_vehicle_planning.dynamic_obstacles import DynamicObstacle
    obs=DynamicObstacle('car',kind='vehicle',start=(10,10),speed=0)
    assert obs.physical_polygon_at(0).centroid.x == pytest.approx(10)


def test_first_command_is_penalized_and_warm_start_shifted(monkeypatch):
    ctl=MPCController(MPCConfig(horizon=3,iter_max=1))
    captured=[]
    def fail(*args):
        captured.append(ctl._nominal_commands.copy())
        return (None,)*6
    monkeypatch.setattr(ctl,'solve_qp',fail)
    ctl.step(np.zeros((4,4)),[0,0,0,0,0,0,0],[.1,.2,.3],[.01,.02,.03])
    np.testing.assert_allclose(captured[0][0],[.2,.3,.3])
    np.testing.assert_allclose(captured[0][1],[.02,.03,.03])


def test_solver_exceptions_trigger_observable_fallback(monkeypatch):
    import cvxpy
    monkeypatch.setattr(cvxpy.Problem,'solve',lambda *args,**kwargs: (_ for _ in ()).throw(cvxpy.SolverError('forced')))
    ctl=MPCController(MPCConfig(horizon=2,iter_max=1))
    a,d,*_=ctl.step(np.zeros((4,3)),[0,0,2,0,.2,0,0])
    assert ctl.last_fallback and ctl.last_solver_status.startswith('fallback:')
    assert a[0]<0 and d[0]==pytest.approx(.2)


def test_quintic_boundaries_have_zero_slope_and_curvature():
    from non_holonomic_vehicle_planning.trajectory import quintic_blend
    h=1e-4
    for u in (0.,1.):
        slope=(quintic_blend(u+h)-quintic_blend(u-h))/(2*h)
        second=(quintic_blend(u+h)-2*quintic_blend(u)+quintic_blend(u-h))/h**2
        assert abs(slope)<1e-6 and abs(second)<.02


def test_urgent_bypass_is_rejected_and_yields():
    from non_holonomic_vehicle_planning import ScenarioCreator,PlannedPath,RealTimePlanner,RealTimePlannerConfig
    sc=(ScenarioCreator('urgent',width=50,height=30,boundary_collision=False)
        .set_start_goal((5,15),0,(45,15),0)
        .add_moving_vehicle('blocker',start=(18,15),speed=.1,yaw_deg=0,
                            waypoints=[(18,15),(19,15)]).build())
    x=np.linspace(5,45,201)
    path=PlannedPath(x,np.full_like(x,15),np.zeros_like(x),np.ones_like(x,dtype=int))
    rt=RealTimePlanner(sc,RealTimePlannerConfig())
    _,info=rt.step(KinematicState(x=7,y=15,v=2.5,vehicle=sc.vehicle),path,10,0,3)
    assert info.mode=='YIELD_STOP' and not info.replan_triggered


def test_telemetry_export_is_aligned_strict_json_and_input_path_unchanged(tmp_path):
    import csv,json
    from non_holonomic_vehicle_planning import Scenario,PlannedPath,SimulationRunner,RealTimePlannerConfig
    sc=Scenario('telemetry',env_width=30,env_height=30)
    x=np.linspace(5,15,51)
    path=PlannedPath(x,np.full_like(x,10),np.zeros_like(x),np.ones_like(x,dtype=int))
    sc.set_planned_path(path)
    result=SimulationRunner(MPCConfig(horizon=3,time_max=.4,iter_max=1),RealTimePlannerConfig(enabled=False)).run(sc)
    csv_path,json_path=result.export(tmp_path/'run')
    rows=list(csv.DictReader(csv_path.open()))
    assert len(rows)==len(result.t)
    assert 'steer_cmd_rad' in rows[0] and 'solver_status' in rows[0]
    payload=json.loads(json_path.read_text(),parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert payload['metrics']['min_dynamic_clearance_m'] is None
    assert sc.planned_path is path
    assert np.all(np.diff(result.telemetry['progress_m'])>=0)


def test_fallback_accounts_for_longitudinal_lag(monkeypatch):
    ctl=MPCController(MPCConfig(horizon=3))
    monkeypatch.setattr(ctl,'solve_qp',lambda *args:(None,)*6)
    state=KinematicState(v=.05)
    state.accel=-.8
    a,d,*_=ctl.step(np.zeros((4,4)),state.augmented_vector)
    state.update(a[0],d[0],dt=.2)
    assert abs(state.v)<1e-6
    assert abs(a[0])<=state.vehicle.max_accel


def test_curvature_candidates_respect_small_physical_limit():
    vehicle=VehicleParams(explicit_max_curvature=.05)
    assert max(vehicle.curvature_candidates)<=.05


def test_path_owns_readonly_arrays():
    from non_holonomic_vehicle_planning import PlannedPath
    x=np.arange(4.)
    path=PlannedPath(x,np.zeros(4),np.zeros(4),np.ones(4,dtype=int))
    x[0]=100
    assert path.x[0]==0
    with pytest.raises(ValueError):path.x[1]=100


def test_static_collision_checks_front_footprint_not_only_reference_point():
    from non_holonomic_vehicle_planning import Scenario,ObstacleBuilder,TrajectorySegment
    sc=Scenario('front',env_width=30,env_height=30)
    sc.set_obstacles(ObstacleBuilder().add_rectangle(1,2,(3,9)).build())
    seg=TrajectorySegment(np.array([0.,.2]),np.array([10.,10.]),np.zeros(2),np.ones(2,dtype=int))
    assert not sc.collision_free(seg)


@pytest.mark.parametrize('mode',['S','L','R'])
@pytest.mark.parametrize('sign',[-1,1])
def test_vectorized_connector_samples_equal_scalar_geometry(mode,sign):
    from non_holonomic_vehicle_planning.trajectory import sample_primitive
    from non_holonomic_vehicle_planning.planning.reeds_shepp import _interpolate_rs
    distances=sign*np.linspace(0,2,41)
    xyz=sample_primitive(distances,mode,.2,(1.,2.,.3))
    expected=np.array([_interpolate_rs(float(d),float(sign),mode,.2,1,2,.3)[:3] for d in distances]).T
    np.testing.assert_allclose(np.array(xyz),expected,atol=1e-12)


def test_configured_vehicle_is_bound_before_global_planning(tmp_path,monkeypatch):
    from non_holonomic_vehicle_planning.scenarios.definitions import load_or_plan_scenario
    from non_holonomic_vehicle_planning.planning.rrt import RRTStarPlanner
    from non_holonomic_vehicle_planning import PlannedPath
    custom=tmp_path/'vehicle.yaml'
    custom.write_text('vehicle:\n  wheelbase: 3.0\n  width: 1.2\n  max_curvature: 0.15\n')
    seen=[]
    def plan(self,scenario):
        seen.append((scenario.vehicle.wheelbase,scenario.vehicle.width,scenario.inflater.vehicle is scenario.vehicle))
        x=np.array([18.,52.5]);y=np.array([5.,45.])
        path=PlannedPath(x,y,np.zeros(2),np.ones(2,dtype=int))
        scenario.set_planned_path(path)
        return path
    monkeypatch.setattr(RRTStarPlanner,'plan',plan)
    load_or_plan_scenario('street_reeds_shepp',config_path=custom,replan=True)
    assert seen==[(3.,1.2,True)]


def test_runner_rejects_path_to_wrong_goal_pose():
    from non_holonomic_vehicle_planning import Scenario,PlannedPath,SimulationRunner
    sc=Scenario('wrong_goal',env_width=30,env_height=30)
    sc.set_start_goal((0,0),0,(10,0),90)
    path=PlannedPath(np.array([0.,10.]),np.zeros(2),np.zeros(2),np.ones(2,dtype=int))
    with pytest.raises(ValueError,match='goal pose'):SimulationRunner().run(sc,path)


def test_large_steering_sqp_change_cannot_converge_using_accel_tolerance(monkeypatch):
    ctl=MPCController(MPCConfig(horizon=2,iter_max=3,du_res=.08,steer_du_res=.005))
    calls=[]
    def solve(*args):
        calls.append(1)
        a=ctl._nominal_commands[0]
        d=ctl._nominal_commands[1]+.01
        return a,d,*([np.zeros(3)]*4)
    monkeypatch.setattr(ctl,'solve_qp',solve)
    monkeypatch.setattr(ctl,'_merit',lambda *args:0.)
    ctl.step(np.zeros((4,3)),[0,0,0,0,0,0,0])
    assert len(calls)==3
    assert ctl.last_sqp_status=='max_iterations'


def test_steering_brakes_before_hard_stop_without_fake_wheel_velocity():
    state=KinematicState()
    previous_rate=0.
    for _ in range(100):
        state.update(0,state.vehicle.max_steer,dt=.02)
        assert abs(state.steer_rate-previous_rate) <= state.vehicle.max_steer_accel*.02+1e-9
        assert abs(state.steer) <= state.vehicle.max_steer+1e-12
        previous_rate=state.steer_rate
    assert state.steer==pytest.approx(state.vehicle.max_steer,abs=1e-4)
    assert abs(state.steer_rate)<1e-3


def _blocked_lane(obstacle_x):
    from non_holonomic_vehicle_planning.planning.base import PlannedPath
    from non_holonomic_vehicle_planning.scenarios.creator import ScenarioCreator
    scenario = (ScenarioCreator("Blocked_Lane", width=75., height=40., boundary_collision=False)
                .set_start_goal((5., 15.), 0., (65., 15.), 0.)
                .add_moving_vehicle("blocker", start=(obstacle_x, 15.), yaw_deg=0., speed=.1, motion_mode="waypoints",
                                    waypoints=[(obstacle_x, 15.), (obstacle_x+1., 15.)])).build()
    x = np.linspace(5., 65., 301)
    y = np.full_like(x, 15.)
    yaw = np.zeros_like(x)
    path = PlannedPath(x=x, y=y, yaw=yaw, direction=np.ones(len(x), dtype=int))
    scenario.set_planned_path(path)
    return scenario, path


def _plan_bypass(scenario, path, state):
    from non_holonomic_vehicle_planning.config import RealTimePlannerConfig
    from non_holonomic_vehicle_planning.planning.realtime import RealTimePlanner
    planner = RealTimePlanner(scenario, RealTimePlannerConfig(enabled=True, lookahead_distance_m=18.))
    return planner.step(state, path, current_idx=10, current_time=0., nominal_speed=3.)


@pytest.mark.parametrize("offset,heading", [(0., 0.), (.005, 0.), (.02, .02), (-.05, -.03), (.15, .05)])
def test_bypass_survives_realistic_tracking_error(offset, heading):
    """Regression: 5 mm of start error used to inflate kappa past the limit and disable all bypasses."""
    from non_holonomic_vehicle_planning.trajectory import geometry
    scenario, path = _blocked_lane(24.)
    state = KinematicState(x=7., y=15.+offset, yaw=heading, v=2.5, vehicle=scenario.vehicle)
    new_path, info = _plan_bypass(scenario, path, state)
    assert info.mode == "AVOID_BYPASS"
    _, kappa = geometry(new_path.x, new_path.y, new_path.yaw, new_path.reversing)
    assert np.max(abs(kappa[12:])) <= scenario.vehicle.max_curvature+2e-3


def test_bypass_starts_at_measured_pose_and_rejoins_base_path():
    scenario, path = _blocked_lane(24.)
    state = KinematicState(x=7.05, y=15.03, yaw=.02, v=2.5, vehicle=scenario.vehicle)
    new_path, info = _plan_bypass(scenario, path, state)
    assert info.replan_triggered
    x, y = info.selected_bypass
    assert y[0] == pytest.approx(state.y, abs=1e-9)
    # Tangent matches the measured heading on departure, and the base line on return.
    assert math.atan2(y[1]-y[0], x[1]-x[0]) == pytest.approx(state.yaw, abs=.01)
    assert y[-1] == pytest.approx(15., abs=1e-6)
    assert np.max(abs(y-15.)) > 1.5


def test_unreachable_close_obstacle_yields_instead_of_infeasible_bypass():
    scenario, path = _blocked_lane(14.)
    state = KinematicState(x=7., y=15., yaw=0., v=2.5, vehicle=scenario.vehicle)
    _, info = _plan_bypass(scenario, path, state)
    assert info.mode == "YIELD_STOP" and not info.replan_triggered and info.speed_scale == 0.


def test_quintic_transition_meets_all_prescribed_boundaries():
    from non_holonomic_vehicle_planning.trajectory import quintic_transition
    d, slope, second = quintic_transition(np.array([0., 7.]), 7., .03, -.02, .01, 2.5)
    np.testing.assert_allclose([d[0], slope[0], second[0]], [.03, -.02, .01], atol=1e-12)
    np.testing.assert_allclose([d[1], slope[1], second[1]], [2.5, 0., 0.], atol=1e-9)


def test_bypass_profile_is_c2_at_phase_junctions():
    from non_holonomic_vehicle_planning.trajectory import lateral_bypass_profile
    s = np.linspace(0., 30., 3001)
    d, slope = lateral_bypass_profile(s, 8., 16., 30., 2.5, .02, .01, .005)
    second = np.diff(slope)/np.diff(s)
    assert np.max(abs(np.diff(d))) < .01 and np.max(abs(np.diff(second))) < .01
    assert d[-1] == pytest.approx(0., abs=1e-9)
    for junction in (8., 16.):
        i = int(np.argmin(abs(s-junction)))
        assert abs(slope[i]) < 1e-4 and abs(second[i]) < .01


def test_min_transition_length_respects_curvature_limit():
    from non_holonomic_vehicle_planning.trajectory import lateral_bypass_profile, min_transition_length
    length = min_transition_length(3., .15)
    s = np.linspace(0., length, 2001)
    _, slope = lateral_bypass_profile(s, length, length, 2*length, 3.)
    assert np.max(abs(np.gradient(slope, s))) == pytest.approx(.15, rel=1e-2)


def test_cold_start_warm_start_holds_measured_actuators(monkeypatch):
    ctl = MPCController(MPCConfig(horizon=3, iter_max=1))
    captured = []
    def fail(*args):
        captured.append(ctl._nominal_commands.copy())
        return (None,)*6
    monkeypatch.setattr(ctl, "solve_qp", fail)
    ctl.step(np.zeros((4, 4)), [0, 0, 2, 0, .3, 0, -.4])
    np.testing.assert_allclose(captured[0], [[-.4]*3, [.3]*3])


def test_replan_does_not_cut_reference_speed():
    from non_holonomic_vehicle_planning.config import RealTimePlannerConfig
    from non_holonomic_vehicle_planning.simulation.runner import SimulationRunner
    scenario, _ = _blocked_lane(30.)
    cfg = MPCConfig(horizon=8, time_max=30., target_speed=2.5, iter_max=3, Rd=np.diag([.02, 1.5]))
    result = SimulationRunner(cfg, RealTimePlannerConfig(enabled=True, lookahead_distance_m=18.)).run(scenario)
    assert result.replan_count >= 1 and result.reached_goal
    events = np.flatnonzero(result.telemetry["replan_event"])
    target = result.telemetry["target_speed_m_s"]
    assert np.min(target[events[0]:events[0]+10]) > .8*target[events[0]-1]
    assert np.min(result.min_dynamic_clearances_m) > 0


def test_near_set_is_bounded_in_small_scenes(monkeypatch):
    import non_holonomic_vehicle_planning.planning.rrt as rrt
    from non_holonomic_vehicle_planning.scenarios.creator import ScenarioCreator
    scenario = (ScenarioCreator("Small", width=30., height=30., boundary_collision=False)
                .set_start_goal((5., 15.), 0., (25., 15.), 0.)).build()
    calls = []
    original = rrt.create_connector_segment
    monkeypatch.setattr(rrt, "create_connector_segment", lambda *args: (calls.append(1), original(*args))[1])
    rrt.RRTStarPlanner(rrt.PlannerConfig(max_iterations=120, step_size=8., connector="dubins", seed=42, max_rewire_neighbors=5)).plan(scenario)
    # candidate pre-filter (12) + choose-parent (5) + rewire (5) + goal attempt per iteration
    assert len(calls) <= 120*(12+5+5+1)


def _cusp_path():
    from non_holonomic_vehicle_planning.planning.base import PlannedPath
    forward = np.arange(0., 12.01, .2)
    backward = forward[-2::-1]
    x = np.r_[forward, backward]+5.
    return PlannedPath(x=x, y=np.full(len(x), 15.), yaw=np.zeros(len(x)),
                       direction=np.r_[np.ones(len(forward), dtype=int), -np.ones(len(backward), dtype=int)])


def test_closed_loop_reverse_cusp_stops_before_gear_change_and_reaches_pose():
    """The catalog only runs gear-free paths, so a cusp must be closed-loop tested explicitly."""
    from non_holonomic_vehicle_planning.config import RealTimePlannerConfig
    from non_holonomic_vehicle_planning.scenario import Scenario
    from non_holonomic_vehicle_planning.simulation.runner import SimulationRunner
    scenario = Scenario("Cusp", env_width=40, env_height=30)
    scenario.set_planned_path(_cusp_path())
    cfg = MPCConfig(horizon=10, time_max=40., target_speed=2., iter_max=3, Rd=np.diag([.02, 1.5]))
    result = SimulationRunner(cfg, RealTimePlannerConfig(enabled=False)).run(scenario)
    assert result.reached_goal and result.termination_reason == "goal"
    assert np.min(result.v) < -.5 and np.max(result.v) > .5
    change = int(np.argmax(result.v < -.05))
    assert np.max(result.x[:change+1]) == pytest.approx(17., abs=.4)
    assert np.max(abs(result.lateral_error)) < .1 and not result.telemetry["solver_fallback"].any()


def test_short_reverse_arc_after_cusp_steers_at_standstill_and_reaches_pose():
    """A 0.65 m arc in reverse is shorter than the distance a lagging wheel needs to swing across."""
    from non_holonomic_vehicle_planning.config import RealTimePlannerConfig
    from non_holonomic_vehicle_planning.planning.base import PlannedPath
    from non_holonomic_vehicle_planning.scenario import Scenario
    from non_holonomic_vehicle_planning.simulation.runner import SimulationRunner
    kappa = .2278
    straight = np.arange(0., 8.01, .2)
    bend = np.arange(.2, 6.01, .2)*kappa
    tail = bend[-1]+np.arange(.2, .66, .15)*kappa
    x = np.r_[straight, 8.+np.sin(bend)/kappa, 8.+np.sin(bend[-1])/kappa-(np.sin(tail)-np.sin(bend[-1]))/kappa]+5.
    y = np.r_[np.zeros(len(straight)), (1.-np.cos(bend))/kappa, (1.-np.cos(bend[-1]))/kappa-(np.cos(bend[-1])-np.cos(tail))/kappa]+8.
    yaw = np.r_[np.zeros(len(straight)), bend, tail]
    direction = np.r_[np.ones(len(straight)+len(bend), dtype=int), -np.ones(len(tail), dtype=int)]
    scenario = Scenario("Short_Cusp", env_width=40, env_height=30)
    scenario.set_planned_path(PlannedPath(x, y, yaw, direction))
    cfg = MPCConfig(horizon=10, time_max=30., target_speed=2., iter_max=3, Rd=np.diag([.02, 1.5]))
    result = SimulationRunner(cfg, RealTimePlannerConfig(enabled=False)).run(scenario)
    assert result.reached_goal and abs(result.yaw[-1]-yaw[-1]) < cfg.heading_stop
    # The wheel swings across at standstill: no reverse travel before it has passed centre.
    leaving = int(np.argmax(result.v < -.15))
    assert result.steer[leaving] < -.3


def test_catalog_start_and_goal_poses_are_collision_free():
    """Poses are rear-axle coordinates; a pose left in center convention can make a goal unreachable."""
    from non_holonomic_vehicle_planning.config import get_configs_dir, load_config
    from non_holonomic_vehicle_planning.planning.base import TrajectorySegment
    from non_holonomic_vehicle_planning.scenarios.definitions import SCENARIO_SPECS, _apply_configuration
    for key, spec in SCENARIO_SPECS.items():
        scenario = spec.factory()
        _apply_configuration(scenario, load_config(get_configs_dir()/"scenarios"/spec.yaml_file if spec.yaml_file else None))
        for point, yaw_deg in (scenario.start, scenario.goal):
            yaw = math.radians(yaw_deg)
            seg = TrajectorySegment(x=np.array([point.x, point.x+1e-3*math.cos(yaw)]),
                                    y=np.array([point.y, point.y+1e-3*math.sin(yaw)]),
                                    yaw=np.full(2, yaw), direction=np.ones(2, dtype=int), total_length=1e-3)
            assert scenario.collision_free(seg), (key, point.x, point.y)


def test_unsolved_planning_problem_does_not_abort_suite(monkeypatch, tmp_path):
    from non_holonomic_vehicle_planning import cli
    def fail(*args, **kwargs):
        raise RuntimeError("RRT could not find a collision-free path")
    monkeypatch.setattr(cli, "load_or_plan_scenario", fail)
    assert cli.main(["--scenario", "street_reeds_shepp", "--no-gif", "--output-dir", str(tmp_path)]) == 1
