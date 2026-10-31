"""Command-line interface for non-holonomic vehicle planning and control."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from non_holonomic_vehicle_planning.config import load_config
from non_holonomic_vehicle_planning.mpc import parse_mpc_config
from non_holonomic_vehicle_planning.scenarios.definitions import (
    SCENARIO_SPECS,
    load_or_plan_scenario,
)
from non_holonomic_vehicle_planning.simulation.runner import SimulationResult, run_simulation


def run_benchmark_suite(
    scenario_names: Optional[List[str]] = None,
    replan: bool = False,
    animate: bool = True,
    tree_gif: bool = True,
    output_dir: str | Path = "./plots",
    iterations: Optional[int] = None,
    config_path: Optional[str | Path] = None,
) -> List[SimulationResult]:
    """Execute one or more scenarios (or YAML scenario files), generating 2D planar vehicle GIFs, RRT* tree GIFs & PNGs."""
    from non_holonomic_vehicle_planning.visualization.animation import animate_rrt_tree_growth

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    app_cfg = load_config(config_path)
    mpc_cfg = parse_mpc_config(app_cfg.raw)

    keys = scenario_names if scenario_names else list(SCENARIO_SPECS.keys())
    results: List[SimulationResult] = []
    metrics_lines: List[str] = []

    for name in keys:
        scenario, spec = load_or_plan_scenario(name, replan=replan, iterations=iterations)
        # If scenario has its own YAML file and no explicit config override was passed, load its settings
        eff_app = load_config( Path("configs/scenarios") / spec.yaml_file ) if (config_path is None and spec.yaml_file) else app_cfg
        eff_mpc = parse_mpc_config(eff_app.raw)

        if animate and tree_gif:
            tree_gif_path = out_dir / f"{spec.key}_RRT_Tree.gif"
            animate_rrt_tree_growth(
                scenario,
                scenario_name=spec.key,
                output_path=tree_gif_path,
                connector=spec.connector,
                iterations=min(spec.default_iterations, 750),
                step_size=spec.default_step_size,
            )
            print(f"[{spec.key}] Saved RRT* Tree Growth GIF animation -> {tree_gif_path}")

        gif_path = out_dir / f"{spec.key}.gif"
        png_path = out_dir / f"{spec.key}.png"
        res = run_simulation(
            scenario,
            spec.key,
            animate=animate,
            gif_output_path=gif_path,
            plot_output_path=png_path,
            mpc_config=eff_mpc,
            realtime_config=eff_app.realtime,
            vis_config=eff_app.visualization,
        )
        results.append(res)
        metrics_lines.extend([
            f"{spec.key}:",
            f"  Simulation Time [s]:               {res.wall_time_s:.4f}",
            f"  Length of Planned Trajectory [m]:  {res.planned_length_m:.4f}",
            f"  Length of Simulated Trajectory [m]:{res.simulated_length_m:.4f}",
            f"  Max Cross-Track Error [m]:         {float(np.max(np.abs(res.lateral_error))):.4f}",
            f"  Real-Time Replans Triggered:       {res.replan_count}",
            f"  Goal Reached:                      {res.reached_goal}",
            "",
        ])

    if len(results) >= 6:
        metrics_file = out_dir / "metrics.txt"
        metrics_file.write_text("\n".join(metrics_lines), encoding="utf-8")

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nh-rrt",
        description=(
            "Non-Holonomic Vehicle Planning & Control: RRT*, real-time obstacle avoidance, "
            "linear MPC, and 2D simulation."
        ),
    )
    parser.add_argument(
        "--scenario",
        "-s",
        default="all",
        help=(
            "Scenario name or YAML file to run (e.g., urban_pedestrian_crossing, "
            "dynamic_overtaking_traffic, street_reeds_shepp, parallel_parking_reeds_shepp, "
            "configs/scenarios/urban_pedestrian_crossing.yaml, or 'all')."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to custom YAML configuration file (merged with configs/default.yaml).",
    )
    parser.add_argument(
        "--replan",
        action="store_true",
        help="Run RRT* path planner from scratch instead of loading saved trajectories.",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=None,
        help="Override RRT* iteration count when --replan is active.",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip GIF animation generation (only produce static PNG summary plots).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./plots",
        help="Output directory for GIF animations and PNG summary plots (default: ./plots).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.scenario.strip().lower() == "all":
        names = list(SCENARIO_SPECS.keys())
    elif args.scenario.strip().lower() in ("urban_variants", "urban_all"):
        names = [
            "Urban_Pedestrian_Crossing_RealTime",
            "Urban_Crossing_V2_Right_Bypass",
            "Urban_Crossing_V3_Pedestrian_Yield",
            "Urban_Crossing_V4_Multi_Vehicle_Slalom",
            "Urban_Crossing_V5_Rush_Hour",
        ]
    else:
        names = [args.scenario]

    results = run_benchmark_suite(
        scenario_names=names,
        replan=args.replan,
        animate=not args.no_gif,
        output_dir=args.output_dir,
        iterations=args.iterations,
        config_path=args.config,
    )
    if args.scenario.strip().lower() in ("urban_variants", "urban_all") and len(results) >= 2:
        from non_holonomic_vehicle_planning.visualization.plots import plot_urban_variants_comparison

        comp_path = Path(args.output_dir) / "Urban_Crossing_Variants_Comparison.png"
        plot_urban_variants_comparison(results, output_path=comp_path)
        print(f"[Urban_Variants_Study] Saved comparative analysis plot -> {comp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
