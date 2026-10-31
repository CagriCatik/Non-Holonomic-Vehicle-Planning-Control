"""Unified scenario catalog (Static & Real-Time Dynamic Scenarios) and YAML builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from non_holonomic_vehicle_planning.config import get_configs_dir
from non_holonomic_vehicle_planning.obstacles import ObstacleBuilder
from non_holonomic_vehicle_planning.planning.rrt import ConnectorType, PlannerConfig, RRTStarPlanner
from non_holonomic_vehicle_planning.scenario import Scenario
from non_holonomic_vehicle_planning.scenarios.creator import ScenarioCreator

CSV_KEYS: Dict[str, str] = {
    "ParallelParking_Scenario_ReedShepp": "ReedsSchepp_43.53_2k",
    "ParallelParking_Scenario_ReedsShepp": "ReedsSchepp_43.53_2k",
    "ParallelParking_Scenario_Dubin": "Dubin_04.26_5k",
    "ParallelParking_Scenario_Dubins": "Dubin_04.26_5k",
    "Parking_Scenario_ReedsShepp": "ReedsShepp_22.59_2k",
    "Parking_Scenario_Dubin": "Dubins_5.25_20k",
    "Parking_Scenario_Dubins": "Dubins_5.25_20k",
    "Street_Scenario_ReedsShepp": "ReedsShepp_12.56_2k",
    "Street_Scenario_Dubin": "Dubins_39.54_20k",
    "Street_Scenario_Dubins": "Dubins_39.54_20k",
    "Urban_Pedestrian_Crossing_RealTime": "ReedsShepp_12.56_2k",
}

CANONICAL_ALIASES: Dict[str, str] = {
    "street_reeds_shepp": "Street_Scenario_ReedsShepp",
    "street_dubins": "Street_Scenario_Dubin",
    "parallel_parking_reeds_shepp": "ParallelParking_Scenario_ReedShepp",
    "parallel_parking_dubins": "ParallelParking_Scenario_Dubin",
    "parking_reeds_shepp": "Parking_Scenario_ReedsShepp",
    "parking_dubins": "Parking_Scenario_Dubin",
    "urban_pedestrian_crossing": "Urban_Pedestrian_Crossing_RealTime",
    "urban_v1": "Urban_Pedestrian_Crossing_RealTime",
    "urban_v2": "Urban_Crossing_V2_Right_Bypass",
    "urban_v3": "Urban_Crossing_V3_Pedestrian_Yield",
    "urban_v4": "Urban_Crossing_V4_Multi_Vehicle_Slalom",
    "urban_v5": "Urban_Crossing_V5_Rush_Hour",
    "dynamic_overtaking_traffic": "Dynamic_Overtaking_Traffic_RealTime",
}


@dataclass(frozen=True)
class ScenarioSpec:
    """Metadata and factory for a benchmark scenario."""

    key: str
    base_name: str
    connector: ConnectorType
    csv_suffix: Optional[str]
    default_iterations: int
    default_step_size: float
    factory: Callable[[], Scenario]
    yaml_file: Optional[str] = None


def build_street_scenario(boundary_collision: bool = False) -> Scenario:
    """Create the Street Intersection Navigation scenario."""
    obs = (
        ObstacleBuilder()
        .add_rectangle(10.5, 22.5, (0.0, 0.0))
        .add_rectangle(10.5, 25.0, (0.0, 37.0))
        .add_rectangle(20.0, 22.5, (25.0, 0.0))
        .add_rectangle(20.0, 25.0, (25.0, 37.0))
        .build()
    )
    sc = Scenario("Street_Scenario", env_width=60.0, env_height=50.0, boundary_collision=boundary_collision)
    sc.set_vehicle(1.0 / 4.39, length=4.5, width=2.0)
    sc.set_obstacles(obs)
    sc.set_start_goal((18.0, 5.0), 90.0, (52.5, 45.0), 90.0)
    return sc


def build_parallel_parking_scenario(reeds_shepp: bool = True) -> Scenario:
    """Create the Parallel Parking maneuver scenario."""
    builder = ObstacleBuilder()
    if reeds_shepp:
        builder.add_rectangle(10.0, 40.0, (16.5, 0.0))
        builder.add_rectangle(2.0, 4.5, (12.0, 1.0))
        builder.add_rectangle(2.0, 4.5, (12.0, 7.0))
        builder.add_rectangle(2.0, 4.5, (12.0, 18.5))
        builder.add_rectangle(2.0, 4.5, (12.0, 33.5))
    else:
        builder.add_rectangle(10.0, 40.0, (16.0, 0.0))
        builder.add_rectangle(2.0, 4.5, (12.0, 1.0))
        builder.add_rectangle(2.0, 4.5, (12.0, 7.0))
        builder.add_rectangle(2.0, 4.5, (12.0, 19.0))
        builder.add_rectangle(2.0, 4.5, (12.0, 33.0))

    sc = Scenario(
        "ParallelParking_Scenario",
        env_width=20.0,
        env_height=40.0,
        boundary_collision=reeds_shepp,
    )
    sc.set_vehicle(1.0 / 4.39, length=4.5, width=2.0)
    sc.set_obstacles(builder.build())
    sc.set_start_goal((5.0, 5.0), 90.0, (13.0, 28.25), 90.0)
    return sc


def build_parking_lot_scenario(boundary_collision: bool = True) -> Scenario:
    """Create the multi-bay Parking Lot navigation scenario."""
    builder = ObstacleBuilder()
    for y in (0.5, 44.5):
        for i in range(14):
            if y == 0.5 and i in (2, 4, 5, 8, 9, 12):
                continue
            if y == 44.5 and i in (0, 1, 2, 5, 6, 8, 11):
                continue
            builder.add_rectangle(2.0, 4.5, (1.5 + i * 3.0, y))

    for y in (20.0, 25.0):
        for i in range(7):
            if y == 25.0 and i in (1, 2, 5):
                continue
            if y == 20.0 and i in (3, 6):
                continue
            builder.add_rectangle(2.0, 4.5, (10.5 + i * 3.0, y))

    for i in range(12):
        if i in (2, 5, 6, 8, 9):
            continue
        builder.add_rectangle(4.5, 2.0, (44.5, 7.0 + i * 3.0))

    sc = Scenario(
        "Parking_Scenario",
        env_width=50.0,
        env_height=50.0,
        boundary_collision=boundary_collision,
    )
    sc.set_vehicle(1.0 / 4.39, length=4.5, width=2.0)
    sc.set_obstacles(builder.build())
    sc.set_start_goal((16.0, 2.75), 90.0, (19.0, 46.75), 90.0)
    return sc


def build_urban_pedestrian_crossing_scenario() -> Scenario:
    """Build the dynamic Urban Pedestrian Crossing & Traffic scenario from YAML."""
    yaml_path = get_configs_dir() / "scenarios" / "urban_pedestrian_crossing.yaml"
    return ScenarioCreator.from_yaml(yaml_path).build()


def build_dynamic_overtaking_scenario() -> Scenario:
    """Build the Dynamic Boulevard Overtaking & Pedestrian Avoidance scenario from YAML."""
    yaml_path = get_configs_dir() / "scenarios" / "dynamic_overtaking_traffic.yaml"
    return ScenarioCreator.from_yaml(yaml_path).build()


SCENARIO_SPECS: Dict[str, ScenarioSpec] = {
    "Street_Scenario_ReedsShepp": ScenarioSpec(
        key="Street_Scenario_ReedsShepp",
        base_name="Street_Scenario",
        connector="reeds_shepp",
        csv_suffix="ReedsShepp_12.56_2k",
        default_iterations=1500,
        default_step_size=12.0,
        factory=lambda: build_street_scenario(boundary_collision=False),
        yaml_file="street_reeds_shepp.yaml",
    ),
    "Street_Scenario_Dubin": ScenarioSpec(
        key="Street_Scenario_Dubin",
        base_name="Street_Scenario",
        connector="dubins",
        csv_suffix="Dubins_39.54_20k",
        default_iterations=2500,
        default_step_size=18.0,
        factory=lambda: build_street_scenario(boundary_collision=False),
    ),
    "ParallelParking_Scenario_ReedShepp": ScenarioSpec(
        key="ParallelParking_Scenario_ReedShepp",
        base_name="ParallelParking_Scenario",
        connector="reeds_shepp",
        csv_suffix="ReedsSchepp_43.53_2k",
        default_iterations=1500,
        default_step_size=10.0,
        factory=lambda: build_parallel_parking_scenario(reeds_shepp=True),
        yaml_file="parallel_parking_reeds_shepp.yaml",
    ),
    "ParallelParking_Scenario_Dubin": ScenarioSpec(
        key="ParallelParking_Scenario_Dubin",
        base_name="ParallelParking_Scenario",
        connector="dubins",
        csv_suffix="Dubin_04.26_5k",
        default_iterations=2500,
        default_step_size=18.0,
        factory=lambda: build_parallel_parking_scenario(reeds_shepp=False),
    ),
    "Parking_Scenario_ReedsShepp": ScenarioSpec(
        key="Parking_Scenario_ReedsShepp",
        base_name="Parking_Scenario",
        connector="reeds_shepp",
        csv_suffix="ReedsShepp_22.59_2k",
        default_iterations=2500,
        default_step_size=15.0,
        factory=lambda: build_parking_lot_scenario(boundary_collision=True),
    ),
    "Parking_Scenario_Dubin": ScenarioSpec(
        key="Parking_Scenario_Dubin",
        base_name="Parking_Scenario",
        connector="dubins",
        csv_suffix="Dubins_5.25_20k",
        default_iterations=3000,
        default_step_size=20.0,
        factory=lambda: build_parking_lot_scenario(boundary_collision=False),
    ),
    "Urban_Pedestrian_Crossing_RealTime": ScenarioSpec(
        key="Urban_Pedestrian_Crossing_RealTime",
        base_name="Urban_Pedestrian_Crossing_RealTime",
        connector="dubins",
        csv_suffix=None,
        default_iterations=600,
        default_step_size=14.0,
        factory=build_urban_pedestrian_crossing_scenario,
        yaml_file="urban_pedestrian_crossing.yaml",
    ),
    "Urban_Crossing_V2_Right_Bypass": ScenarioSpec(
        key="Urban_Crossing_V2_Right_Bypass",
        base_name="Urban_Crossing_V2_Right_Bypass",
        connector="dubins",
        csv_suffix=None,
        default_iterations=600,
        default_step_size=14.0,
        factory=lambda: ScenarioCreator.from_yaml(get_configs_dir() / "scenarios" / "urban_crossing_v2_right_bypass.yaml").build(),
        yaml_file="urban_crossing_v2_right_bypass.yaml",
    ),
    "Urban_Crossing_V3_Pedestrian_Yield": ScenarioSpec(
        key="Urban_Crossing_V3_Pedestrian_Yield",
        base_name="Urban_Crossing_V3_Pedestrian_Yield",
        connector="dubins",
        csv_suffix=None,
        default_iterations=600,
        default_step_size=14.0,
        factory=lambda: ScenarioCreator.from_yaml(get_configs_dir() / "scenarios" / "urban_crossing_v3_pedestrian_yield.yaml").build(),
        yaml_file="urban_crossing_v3_pedestrian_yield.yaml",
    ),
    "Urban_Crossing_V4_Multi_Vehicle_Slalom": ScenarioSpec(
        key="Urban_Crossing_V4_Multi_Vehicle_Slalom",
        base_name="Urban_Crossing_V4_Multi_Vehicle_Slalom",
        connector="dubins",
        csv_suffix=None,
        default_iterations=600,
        default_step_size=14.0,
        factory=lambda: ScenarioCreator.from_yaml(get_configs_dir() / "scenarios" / "urban_crossing_v4_slalom.yaml").build(),
        yaml_file="urban_crossing_v4_slalom.yaml",
    ),
    "Urban_Crossing_V5_Rush_Hour": ScenarioSpec(
        key="Urban_Crossing_V5_Rush_Hour",
        base_name="Urban_Crossing_V5_Rush_Hour",
        connector="dubins",
        csv_suffix=None,
        default_iterations=600,
        default_step_size=14.0,
        factory=lambda: ScenarioCreator.from_yaml(get_configs_dir() / "scenarios" / "urban_crossing_v5_rush_hour.yaml").build(),
        yaml_file="urban_crossing_v5_rush_hour.yaml",
    ),
    "Dynamic_Overtaking_Traffic_RealTime": ScenarioSpec(
        key="Dynamic_Overtaking_Traffic_RealTime",
        base_name="Dynamic_Overtaking_Traffic_RealTime",
        connector="dubins",
        csv_suffix=None,
        default_iterations=400,
        default_step_size=15.0,
        factory=build_dynamic_overtaking_scenario,
        yaml_file="dynamic_overtaking_traffic.yaml",
    ),
}


def resolve_scenario_key(name: str) -> str:
    """Resolve short or canonical scenario names to a registered ScenarioSpec key."""
    if name in SCENARIO_SPECS:
        return name
    low = name.strip().lower()
    if low in CANONICAL_ALIASES:
        return CANONICAL_ALIASES[low]
    for key in SCENARIO_SPECS:
        if key.lower() == low:
            return key
    raise KeyError(
        f"Unknown scenario '{name}'. Available: {list(SCENARIO_SPECS.keys())} or {list(CANONICAL_ALIASES.keys())}"
    )


def load_or_plan_scenario(
    name: str,
    replan: bool = False,
    iterations: Optional[int] = None,
    seed: Optional[int] = 42,
    saved_dir: str | Path = "./Saved_scenarios",
) -> Tuple[Scenario, ScenarioSpec]:
    """Build a scenario (or load a custom YAML file) and compute or load its baseline trajectory."""
    # Support passing a direct path to a YAML file
    if name.endswith((".yaml", ".yml")) or Path(name).exists():
        creator = ScenarioCreator.from_yaml(name)
        scenario = creator.build()
        pl_cfg = creator.raw_config.get("planner", {})
        conn: ConnectorType = pl_cfg.get("connector", "reeds_shepp")
        n_iter = iterations or int(pl_cfg.get("max_iterations", 800))
        step_sz = float(pl_cfg.get("step_size", 14.0))
        spec = ScenarioSpec(
            key=scenario.name,
            base_name=scenario.name,
            connector=conn,
            csv_suffix=None,
            default_iterations=n_iter,
            default_step_size=step_sz,
            factory=creator.build,
            yaml_file=str(name),
        )
        planner = RRTStarPlanner(
            PlannerConfig(
                max_iterations=n_iter,
                step_size=step_sz,
                connector=conn,
                star=bool(pl_cfg.get("star", True)),
                seed=seed,
            )
        )
        planner.plan(scenario)
        return scenario, spec

    key = resolve_scenario_key(name)
    spec = SCENARIO_SPECS[key]
    scenario = spec.factory()

    csv_file = (Path(saved_dir) / f"{spec.base_name}_Path_{spec.csv_suffix}") if spec.csv_suffix else None
    if not replan and csv_file is not None and csv_file.exists():
        orig_name = scenario.name
        scenario.name = spec.base_name
        scenario.read_csv(spec.csv_suffix, set_path=True, directory=saved_dir)
        scenario.name = orig_name
    else:
        planner = RRTStarPlanner(
            PlannerConfig(
                max_iterations=iterations or spec.default_iterations,
                step_size=spec.default_step_size,
                connector=spec.connector,
                star=True,
                seed=seed,
            )
        )
        planner.plan(scenario)

    return scenario, spec
