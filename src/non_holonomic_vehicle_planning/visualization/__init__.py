"""2D Planar Vehicle Visualization & GIF Animation Package."""

from non_holonomic_vehicle_planning.visualization.animation import animate_rrt_tree_growth, animate_simulation
from non_holonomic_vehicle_planning.visualization.plan_view import (
    PlanarVehicleArtist,
    draw_heading_arrow,
    draw_vehicle,
    vehicle_geometry_world,
    vehicle_polygons,
)
from non_holonomic_vehicle_planning.visualization.plots import (
    draw_scenario_environment,
    plot_planning_result,
    plot_simulation_summary,
    plot_urban_variants_comparison,
)
from non_holonomic_vehicle_planning.visualization.theme import apply_dark_theme, style_axis

__all__ = [
    "PlanarVehicleArtist",
    "animate_simulation",
    "apply_dark_theme",
    "draw_heading_arrow",
    "draw_scenario_environment",
    "draw_vehicle",
    "plot_planning_result",
    "plot_simulation_summary",
    "style_axis",
    "vehicle_geometry_world",
    "vehicle_polygons",
]

