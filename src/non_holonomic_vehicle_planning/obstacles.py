"""Obstacle builder utilities using Shapely polygons."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
from shapely.geometry import Polygon


class ObstacleBuilder:
    """Fluent helper to construct 2D polygonal obstacles for planning scenarios."""

    def __init__(self) -> None:
        self._polygons: List[Polygon] = []

    def add_polygon(self, coords: Sequence[Tuple[float, float]]) -> "ObstacleBuilder":
        """Add an arbitrary polygon specified by its exterior vertex coordinates."""
        poly = Polygon(coords)
        self._polygons.append(poly)
        return self

    def add_rectangle(
        self,
        width: float,
        height: float,
        bottom_left: Tuple[float, float],
    ) -> "ObstacleBuilder":
        """Add an axis-aligned rectangle defined by width, height, and bottom-left corner (x, y)."""
        x, y = bottom_left
        poly = Polygon([
            (x, y),
            (x, y + height),
            (x + width, y + height),
            (x + width, y),
        ])
        self._polygons.append(poly)
        return self

    def build(self) -> List[Polygon]:
        """Return the list of constructed Shapely Polygon obstacles."""
        return list(self._polygons)

    # Legacy compatibility aliases
    def create_polygon(self, coords: Sequence[Tuple[float, float]]) -> Polygon:
        poly = Polygon(coords)
        self._polygons.append(poly)
        return poly

    def create_rectangle(self, width: float, height: float, left_corner: Tuple[float, float]) -> Polygon:
        x, y = left_corner
        poly = Polygon([
            (x, y),
            (x, y + height),
            (x + width, y + height),
            (x + width, y),
        ])
        self._polygons.append(poly)
        return poly

    def return_obstacles(self) -> List[Polygon]:
        return self.build()


# Backward compatibility alias
ObstacleCreator = ObstacleBuilder

