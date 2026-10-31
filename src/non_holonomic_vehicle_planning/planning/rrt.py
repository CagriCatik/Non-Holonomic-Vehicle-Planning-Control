"""RRT and RRT* motion planner with Dubins, Reeds-Shepp, and Euclidean connectors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Literal, Optional, Tuple

import numpy as np
from shapely.geometry import Point

from non_holonomic_vehicle_planning.planning.base import PlannedPath, TrajectorySegment
from non_holonomic_vehicle_planning.planning.dubins import iter_dubins_segments
from non_holonomic_vehicle_planning.planning.reeds_shepp import iter_reeds_shepp_segments
from non_holonomic_vehicle_planning.scenario import Scenario
from non_holonomic_vehicle_planning.vehicle import wrap_angle

ConnectorType = Literal["reeds_shepp", "dubins", "euclidean"]


@dataclass
class PlannerConfig:
    """Configuration parameters for the RRT / RRT* motion planner."""

    max_iterations: int = 1500
    step_size: float = 15.0
    goal_tolerance: float = 1.5
    rewire_radius: float = 20.0
    max_rewire_neighbors: int = 40  # Closest nodes considered; bounds the cost per iteration in small scenes.
    goal_sample_rate: float = 0.10
    goal_orientation_bias: float = 0.15
    star: bool = True
    connector: ConnectorType = "reeds_shepp"
    seed: Optional[int] = None


class TreeNode:
    """Node in the RRT/RRT* search tree.

    Attributes:
        point: Shapely Point(x, y) in world coordinates [m].
        yaw: Heading angle in degrees [0, 360).
        cost: Cumulative path length from start node [m].
        parent: Parent TreeNode in the tree.
        segment_to_parent: TrajectorySegment connecting parent -> self.
        children: Child TreeNodes.
    """

    def __init__(self, point: Point, yaw_deg: float) -> None:
        self.point: Point = point
        self.yaw: float = float(yaw_deg) % 360.0
        self.cost: float = 0.0
        self.parent: Optional["TreeNode"] = None
        self.segment_to_parent: Optional[TrajectorySegment] = None
        self.children: List["TreeNode"] = []

    @property
    def x(self) -> float:
        return float(self.point.x)

    @property
    def y(self) -> float:
        return float(self.point.y)

    @property
    def yaw_rad(self) -> float:
        return wrap_angle(math.radians(self.yaw))

    @property
    def path_to_parent(self):
        return self.segment_to_parent.linestring if self.segment_to_parent is not None else None


def _angle_diff_deg(a_deg: float, b_deg: float) -> float:
    diff = abs((a_deg - b_deg) % 360.0)
    return min(diff, 360.0 - diff)


def _heuristic_cost(n1: TreeNode, n2: TreeNode) -> float:
    dist = math.hypot(n1.x - n2.x, n1.y - n2.y)
    ang_rad = math.radians(_angle_diff_deg(n1.yaw, n2.yaw))
    return dist + 1.5 * ang_rad


def create_connector_segment(
    node_from: TreeNode,
    node_to: TreeNode,
    scenario: Scenario,
    non_holonomic: bool = True,
    backwards: bool = True,
) -> Optional[TrajectorySegment]:
    """Generate the shortest collision-free trajectory segment from ``node_from`` to ``node_to``."""
    sx, sy, syaw = node_from.x, node_from.y, node_from.yaw_rad
    gx, gy, gyaw = node_to.x, node_to.y, node_to.yaw_rad

    if math.hypot(gx - sx, gy - sy) < 1e-3:
        return None

    if not non_holonomic:
        n_pts = max(2, int(math.ceil(math.hypot(gx - sx, gy - sy) / 0.2)) + 1)
        xs = np.linspace(sx, gx, n_pts)
        ys = np.linspace(sy, gy, n_pts)
        heading = math.atan2(gy - sy, gx - sx)
        yaws = np.full(n_pts, wrap_angle(heading), dtype=float)
        dirs = np.ones(n_pts, dtype=int)
        seg = TrajectorySegment(x=xs, y=ys, yaw=yaws, direction=dirs, ctypes=["S"])
        return seg if scenario.collision_free(seg) else None

    maxc = scenario.max_curvature
    if backwards:
        candidates = iter_reeds_shepp_segments(sx, sy, syaw, gx, gy, gyaw, maxc, step=0.2)
    else:
        candidates = iter_dubins_segments(sx, sy, syaw, gx, gy, gyaw, maxc, step=0.2)

    for seg in candidates:
        if scenario.collision_free(seg):
            return seg
    return None


def _truncate_segment(seg: TrajectorySegment, max_length: float) -> Tuple[TrajectorySegment, TreeNode]:
    """Truncate a TrajectorySegment to at most ``max_length`` [m] arc-length."""
    if seg.length <= max_length or len(seg.x) < 2:
        end_yaw_deg = (math.degrees(float(seg.yaw[-1])) + 360.0) % 360.0
        return seg, TreeNode(Point(float(seg.x[-1]), float(seg.y[-1])), end_yaw_deg)

    ds = np.hypot(np.diff(seg.x), np.diff(seg.y))
    s_cum = np.concatenate(([0.0], np.cumsum(ds)))
    idx = int(np.searchsorted(s_cum, max_length, side="right"))
    idx = max(2, min(idx, len(seg.x)))

    sub_x = seg.x[:idx].copy()
    sub_y = seg.y[:idx].copy()
    sub_yaw = seg.yaw[:idx].copy()
    sub_dir = seg.direction[:idx].copy()
    sub_seg = TrajectorySegment(
        x=sub_x,
        y=sub_y,
        yaw=sub_yaw,
        direction=sub_dir,
        ctypes=seg.ctypes,
        total_length=float(s_cum[idx - 1]),
    )
    new_yaw_deg = (math.degrees(float(sub_yaw[-1])) + 360.0) % 360.0
    new_node = TreeNode(Point(float(sub_x[-1]), float(sub_y[-1])), new_yaw_deg)
    return sub_seg, new_node


def update_children_cost(node: TreeNode, cost_difference: float) -> None:
    """Recursively subtract ``cost_difference`` from all descendants of ``node``."""
    node.cost -= cost_difference
    for child in node.children:
        update_children_cost(child, cost_difference)


def collect_all_nodes(root: TreeNode) -> List[TreeNode]:
    """Return all nodes in the tree rooted at ``root``."""
    nodes: List[TreeNode] = []
    stack = [root]
    while stack:
        curr = stack.pop()
        nodes.append(curr)
        stack.extend(curr.children)
    return nodes


def extract_all_edges(start_node: TreeNode) -> List[TrajectorySegment]:
    """Extract all TrajectorySegments in the RRT tree."""
    edges: List[TrajectorySegment] = []
    for node in collect_all_nodes(start_node):
        if node.segment_to_parent is not None:
            edges.append(node.segment_to_parent)
    return edges


def extract_path_segments(final_node: TreeNode) -> List[TrajectorySegment]:
    """Extract TrajectorySegments from ``final_node`` back to root (goal -> start order)."""
    segments: List[TrajectorySegment] = []
    curr: Optional[TreeNode] = final_node
    while curr is not None and curr.parent is not None:
        if curr.segment_to_parent is not None:
            segments.append(curr.segment_to_parent)
        curr = curr.parent
    return segments


@dataclass
class TreeGrowthSnapshot:
    """Single step snapshot of RRT / RRT* tree growth for animation."""

    iteration: int
    sampled_xy: Tuple[float, float]
    added_segment: TrajectorySegment
    node_pose: Tuple[float, float, float]  # (x, y, yaw_rad)
    num_nodes: int
    rewire_count: int
    best_cost: float
    best_path_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None


class RRTStarPlanner:
    """Object-oriented Non-Holonomic RRT / RRT* Motion Planner."""

    def __init__(self, config: Optional[PlannerConfig] = None) -> None:
        self.config = config or PlannerConfig()

    def plan(self, scenario: Scenario) -> PlannedPath:
        """Execute RRT/RRT* on ``scenario`` and populate ``scenario.planned_path`` & ``tree_growth_history``."""
        if scenario.start is None or scenario.goal is None:
            raise ValueError("Scenario start and goal must be configured before planning.")

        rng = np.random.default_rng(self.config.seed)
        non_holonomic = self.config.connector != "euclidean"
        backwards = self.config.connector == "reeds_shepp"

        start_node = TreeNode(scenario.start[0], scenario.start[1])
        goal_node = TreeNode(scenario.goal[0], scenario.goal[1])

        max_candidates_check = 12
        growth_history: List[TreeGrowthSnapshot] = []
        total_rewires = 0
        current_best_cost = float("inf")
        current_best_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None

        for n in range(self.config.max_iterations):
            if n == 0 or (self.config.star and n == self.config.max_iterations - 1):
                sampled_node = TreeNode(goal_node.point, goal_node.yaw)
            elif rng.random() < self.config.goal_sample_rate:
                sampled_node = TreeNode(goal_node.point, goal_node.yaw)
            else:
                rx = float(rng.uniform(0.0, scenario.width))
                ry = float(rng.uniform(0.0, scenario.height))
                # Sample full [0, 360) degrees uniformly (fixes original double deg2rad bug)
                ryaw = float(rng.uniform(0.0, 360.0))
                sampled_node = TreeNode(Point(rx, ry), ryaw)
                if rng.random() < self.config.goal_orientation_bias:
                    ang_goal = math.degrees(
                        math.atan2(goal_node.y - sampled_node.y, goal_node.x - sampled_node.x)
                    )
                    sampled_node.yaw = (ang_goal + 360.0) % 360.0

            raw_sample_xy = (sampled_node.x, sampled_node.y)
            if not scenario.collision_free(sampled_node.point):
                continue

            # Pre-filter candidate tree nodes by lower-bound heuristic distance
            all_nodes = collect_all_nodes(start_node)
            all_nodes.sort(key=lambda nd: _heuristic_cost(nd, sampled_node))

            parent_node: Optional[TreeNode] = None
            best_seg: Optional[TrajectorySegment] = None
            min_len = float("inf")

            for cand in all_nodes[:max_candidates_check]:
                if math.hypot(cand.x - sampled_node.x, cand.y - sampled_node.y) >= min_len:
                    continue
                seg = create_connector_segment(cand, sampled_node, scenario, non_holonomic, backwards)
                if seg is not None and seg.length < min_len:
                    min_len = seg.length
                    parent_node = cand
                    best_seg = seg

            if parent_node is None or best_seg is None:
                continue

            if best_seg.length > self.config.step_size:
                best_seg, sampled_node = _truncate_segment(best_seg, self.config.step_size)
                if not scenario.collision_free(sampled_node.point):
                    continue

            sampled_node.cost = parent_node.cost + best_seg.length

            # RRT* Choose Parent within rewire_radius
            nearby_nodes: List[TreeNode] = []
            if self.config.star:
                distances = [(math.hypot(nd.x - sampled_node.x, nd.y - sampled_node.y), i) for i, nd in enumerate(all_nodes)]
                nearest = sorted(d for d in distances if d[0] < self.config.rewire_radius)
                nearby_nodes = [
                    all_nodes[i] for _, i in nearest[:self.config.max_rewire_neighbors]
                    if all_nodes[i] is not sampled_node
                ]
                min_total_cost = sampled_node.cost
                for near_nd in nearby_nodes:
                    if near_nd.cost + math.hypot(near_nd.x - sampled_node.x, near_nd.y - sampled_node.y) >= min_total_cost:
                        continue
                    cand_seg = create_connector_segment(near_nd, sampled_node, scenario, non_holonomic, backwards)
                    if cand_seg is not None and near_nd.cost + cand_seg.length < min_total_cost:
                        parent_node = near_nd
                        best_seg = cand_seg
                        min_total_cost = near_nd.cost + cand_seg.length

            sampled_node.cost = parent_node.cost + best_seg.length
            parent_node.children.append(sampled_node)
            sampled_node.parent = parent_node
            sampled_node.segment_to_parent = best_seg

            # RRT* Rewire nearby nodes through sampled_node
            rewired_this_step = False
            if self.config.star and nearby_nodes:
                for near_nd in nearby_nodes:
                    if near_nd is parent_node:
                        continue
                    if sampled_node.cost + math.hypot(sampled_node.x - near_nd.x, sampled_node.y - near_nd.y) >= near_nd.cost:
                        continue
                    rewire_seg = create_connector_segment(sampled_node, near_nd, scenario, non_holonomic, backwards)
                    if rewire_seg is None:
                        continue
                    new_cost = sampled_node.cost + rewire_seg.length
                    if new_cost < near_nd.cost:
                        if near_nd.parent is not None:
                            near_nd.parent.children.remove(near_nd)
                        near_nd.parent = sampled_node
                        near_nd.segment_to_parent = rewire_seg
                        sampled_node.children.append(near_nd)
                        cost_diff = near_nd.cost - new_cost
                        update_children_cost(near_nd, cost_diff)
                        total_rewires += 1
                        rewired_this_step = True

            # Opportunistically try direct connection to Goal when close so tree growth shows progressive optimization
            dist_to_goal = math.hypot(sampled_node.x - goal_node.x, sampled_node.y - goal_node.y)
            if dist_to_goal > self.config.goal_tolerance and dist_to_goal <= self.config.step_size * 1.15:
                seg_to_g = create_connector_segment(sampled_node, goal_node, scenario, non_holonomic, backwards)
                if seg_to_g is not None and sampled_node.cost + seg_to_g.length < current_best_cost:
                    g_leaf = TreeNode(goal_node.point, goal_node.yaw)
                    g_leaf.cost = sampled_node.cost + seg_to_g.length
                    g_leaf.parent = sampled_node
                    g_leaf.segment_to_parent = seg_to_g
                    sampled_node.children.append(g_leaf)
                    rewired_this_step = True

            # Update best path snapshot if a goal candidate exists
            if dist_to_goal <= self.config.goal_tolerance or rewired_this_step:
                curr_all = collect_all_nodes(start_node)
                g_cands = [
                    nd for nd in curr_all
                    if math.hypot(nd.x - goal_node.x, nd.y - goal_node.y) <= self.config.goal_tolerance
                    and nd is not start_node
                ]
                if g_cands:
                    best_g = min(
                        g_cands,
                        key=lambda nd: nd.cost + 2.0 * math.hypot(nd.x - goal_node.x, nd.y - goal_node.y)
                        + 0.1 * _angle_diff_deg(nd.yaw, goal_node.yaw),
                    )
                    if best_g.cost < current_best_cost - 1e-3 or current_best_xy is None:
                        current_best_cost = best_g.cost
                        segs_rev = extract_path_segments(best_g)
                        p_tmp = PlannedPath.from_segments(list(reversed(segs_rev)))
                        current_best_xy = (p_tmp.x.copy(), p_tmp.y.copy())

            growth_history.append(
                TreeGrowthSnapshot(
                    iteration=n + 1,
                    sampled_xy=raw_sample_xy,
                    added_segment=best_seg,
                    node_pose=(sampled_node.x, sampled_node.y, sampled_node.yaw_rad),
                    num_nodes=len(all_nodes) + 1,
                    rewire_count=total_rewires,
                    best_cost=current_best_cost,
                    best_path_xy=current_best_xy,
                )
            )

            # Check if we reached goal region
            if dist_to_goal < self.config.goal_tolerance:
                if not self.config.star:
                    break

        # A position-near node with the wrong heading is not a goal pose.
        all_nodes = collect_all_nodes(start_node)
        goal_candidates = [
            nd for nd in all_nodes
            if math.hypot(nd.x-goal_node.x,nd.y-goal_node.y) <= 1e-5
            and _angle_diff_deg(nd.yaw,goal_node.yaw) <= 1e-4
            and nd is not start_node
        ]

        # If no node landed directly inside goal_tolerance, try connecting the closest nodes directly to goal_node
        if not goal_candidates:
            closest_to_goal = sorted(
                all_nodes,
                key=lambda nd: math.hypot(nd.x - goal_node.x, nd.y - goal_node.y),
            )[:10]
            for cand in closest_to_goal:
                seg_to_goal = create_connector_segment(cand, goal_node, scenario, non_holonomic, backwards)
                if seg_to_goal is not None:
                    g_leaf = TreeNode(goal_node.point, goal_node.yaw)
                    g_leaf.cost = cand.cost + seg_to_goal.length
                    g_leaf.parent = cand
                    g_leaf.segment_to_parent = seg_to_goal
                    cand.children.append(g_leaf)
                    goal_candidates.append(g_leaf)

        total_edges = extract_all_edges(start_node)
        scenario.set_totaltree(total_edges)
        scenario.tree_segments = list(total_edges)
        scenario.tree_growth_history = growth_history

        if not goal_candidates:
            raise RuntimeError(
                f"RRT could not find a collision-free path within {self.config.max_iterations} iterations."
            )

        best_goal_node = min(
            goal_candidates,
            key=lambda nd: nd.cost + 2.0 * math.hypot(nd.x - goal_node.x, nd.y - goal_node.y)
            + 0.1 * _angle_diff_deg(nd.yaw, goal_node.yaw),
        )
        segments_rev = extract_path_segments(best_goal_node)
        scenario.set_path(segments_rev)
        assert scenario.planned_path is not None
        if growth_history:
            growth_history[-1].best_cost = scenario.planned_path.length
            growth_history[-1].best_path_xy = (scenario.planned_path.x.copy(), scenario.planned_path.y.copy())
        return scenario.planned_path


def RRT(
    N_iter: int,
    scenario: Scenario,
    step_size: float = float("inf"),
    dist_tolerance: float = 1.0,
    star: bool = True,
    non_holonomic: bool = True,
    force_return_tree: bool = False,
    backwards: bool = True,
    seed: Optional[int] = None,
) -> Optional[PlannedPath]:
    """Functional interface for RRT / RRT* planning (compatible with original API)."""
    connector: ConnectorType = (
        "euclidean" if not non_holonomic else ("reeds_shepp" if backwards else "dubins")
    )
    cfg = PlannerConfig(
        max_iterations=N_iter,
        step_size=step_size,
        goal_tolerance=dist_tolerance,
        star=star,
        connector=connector,
        seed=seed,
    )
    planner = RRTStarPlanner(cfg)
    try:
        return planner.plan(scenario)
    except RuntimeError:
        if force_return_tree:
            return None
        raise

