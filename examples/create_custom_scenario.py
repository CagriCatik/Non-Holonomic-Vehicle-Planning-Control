"""Example: Build a custom dynamic scenario with ScenarioCreator, export to YAML, and generate GIF."""

from __future__ import annotations

from non_holonomic_vehicle_planning import (
    PlannerConfig,
    RRTStarPlanner,
    ScenarioCreator,
    run_simulation,
)


def main() -> None:
    # 1. Build a custom scenario with static obstacles, moving vehicles, and crossing pedestrians
    creator = (
        ScenarioCreator(name="Custom_Crosswalk_Scenario", width=55.0, height=36.0, boundary_collision=False)
        .set_inflation(
            enabled=True,
            hard_margin_m=0.30,
            soft_margin_m=1.15,
            pedestrian_safety_margin_m=1.15,
            dynamic_vehicle_margin_m=0.60,
        )
        .set_start_goal(start=(6.5, 13.5), start_yaw_deg=0.0, goal=(49.0, 14.5), goal_yaw_deg=0.0)
        .add_rectangle_obstacle(55.0, 5.5, bottom_left=(0.0, 0.0))
        .add_rectangle_obstacle(55.0, 5.5, bottom_left=(0.0, 30.5))
        .add_moving_vehicle(
            id="parked_delivery_truck",
            start=(20.0, 13.5),
            yaw_deg=0.0,
            speed=0.25,
            motion_mode="waypoints",
            waypoints=[(20.0, 13.5), (24.0, 13.5)],
        )
        .add_pedestrian(
            id="pedestrian_crosswalk",
            start=(35.0, 8.0),
            goal=(35.0, 27.0),
            speed=1.30,
            periodic=True,
        )
    )

    # 2. Export to YAML configuration
    yaml_path = creator.to_yaml("configs/scenarios/custom_crosswalk.yaml")
    print(f"Exported custom scenario YAML -> {yaml_path}")

    # 3. Build scenario, run global RRT* planner, and execute Real-Time MPC + GIF generation
    scenario = creator.build()
    RRTStarPlanner(PlannerConfig(max_iterations=400, connector="dubins", seed=42)).plan(scenario)

    run_simulation(
        scenario,
        "Custom_Crosswalk_Scenario",
        animate=True,
        gif_output_path="plots/Custom_Crosswalk_Scenario.gif",
        plot_output_path="plots/Custom_Crosswalk_Scenario.png",
    )


if __name__ == "__main__":
    main()

