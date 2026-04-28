from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from .models import AtlasProject, TileRecord
from .refinement_diagnostics import summarize_neighbor_constraints, summarize_refined_positions


@dataclass
class NeighborConstraint:
    tile_a_id: str
    tile_b_id: str
    dx: float
    dy: float
    confidence: float
    direction: str = ""


def solve_refined_tile_positions(project: AtlasProject, constraints: list[NeighborConstraint]) -> AtlasProject:
    solved_project = AtlasProject.from_dict(project.to_dict())
    tile_by_id = {tile.tile_id: tile for tile in solved_project.tiles}
    indexed_tiles = [tile for tile in solved_project.tiles if tile.tile_id]
    if not indexed_tiles:
        return solved_project

    anchor_tile = choose_anchor_tile(indexed_tiles)
    if anchor_tile is None:
        return solved_project

    valid_constraints = [
        constraint
        for constraint in constraints
        if constraint.tile_a_id in tile_by_id and constraint.tile_b_id in tile_by_id
    ]
    if not valid_constraints:
        _mark_project_refinement_state(
            solved_project,
            placement_mode="nominal",
            refinement_status="no_constraints",
            refined_tile_count=0,
        )
        return solved_project

    matrix, rhs = build_constraint_system(indexed_tiles, valid_constraints, anchor_tile)
    solution = lsqr(matrix, rhs)[0]
    tile_count = len(indexed_tiles)
    for index, tile in enumerate(indexed_tiles):
        tile.transform.refined_x = _stable_coordinate(solution[index])
        tile.transform.refined_y = _stable_coordinate(solution[tile_count + index])

    for component_anchor in choose_component_anchors(indexed_tiles, valid_constraints, preferred_anchor=anchor_tile):
        anchor_x, anchor_y = get_nominal_position(component_anchor)
        component_anchor.transform.refined_x = anchor_x
        component_anchor.transform.refined_y = anchor_y

    _mark_project_refinement_state(
        solved_project,
        constraints=valid_constraints,
        placement_mode="refined_translation",
        refinement_status="solved",
        refined_tile_count=len(indexed_tiles),
    )
    return solved_project


def build_constraint_system(
    tiles: list[TileRecord],
    constraints: Iterable[NeighborConstraint],
    anchor_tile: TileRecord,
) -> tuple[coo_matrix, np.ndarray]:
    constraint_list = list(constraints)
    tile_to_index = {tile.tile_id: index for index, tile in enumerate(tiles)}
    tile_count = len(tiles)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row_index = 0

    for constraint in constraint_list:
        index_a = tile_to_index.get(constraint.tile_a_id)
        index_b = tile_to_index.get(constraint.tile_b_id)
        if index_a is None or index_b is None:
            continue
        weight = _constraint_weight(constraint.confidence)

        rows.extend([row_index, row_index])
        cols.extend([index_a, index_b])
        data.extend([-weight, weight])
        rhs.append(weight * float(constraint.dx))
        row_index += 1

        rows.extend([row_index, row_index])
        cols.extend([tile_count + index_a, tile_count + index_b])
        data.extend([-weight, weight])
        rhs.append(weight * float(constraint.dy))
        row_index += 1

    anchor_weight = max(1.0, np.sqrt(max(1, tile_count)))
    for component_anchor in choose_component_anchors(tiles, constraint_list, preferred_anchor=anchor_tile):
        anchor_index = tile_to_index[component_anchor.tile_id]
        anchor_x, anchor_y = get_nominal_position(component_anchor)

        rows.append(row_index)
        cols.append(anchor_index)
        data.append(anchor_weight)
        rhs.append(anchor_weight * anchor_x)
        row_index += 1

        rows.append(row_index)
        cols.append(tile_count + anchor_index)
        data.append(anchor_weight)
        rhs.append(anchor_weight * anchor_y)
        row_index += 1

    matrix = coo_matrix((data, (rows, cols)), shape=(row_index, tile_count * 2), dtype=float)
    return matrix.tocsr(), np.asarray(rhs, dtype=float)


def choose_anchor_tile(tiles: list[TileRecord]) -> TileRecord | None:
    for tile in tiles:
        if tile.start_x is not None and tile.start_y is not None:
            return tile
    return tiles[0] if tiles else None


def choose_component_anchors(
    tiles: list[TileRecord],
    constraints: Iterable[NeighborConstraint],
    *,
    preferred_anchor: TileRecord | None = None,
) -> list[TileRecord]:
    tile_by_id = {tile.tile_id: tile for tile in tiles}
    adjacency: dict[str, set[str]] = {tile.tile_id: set() for tile in tiles}
    for constraint in constraints:
        if constraint.tile_a_id not in adjacency or constraint.tile_b_id not in adjacency:
            continue
        adjacency[constraint.tile_a_id].add(constraint.tile_b_id)
        adjacency[constraint.tile_b_id].add(constraint.tile_a_id)

    anchors: list[TileRecord] = []
    visited: set[str] = set()
    preferred_id = preferred_anchor.tile_id if preferred_anchor is not None else ""
    for tile in tiles:
        if tile.tile_id in visited:
            continue
        stack = [tile.tile_id]
        component_ids: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component_ids.append(current)
            stack.extend(neighbor for neighbor in adjacency[current] if neighbor not in visited)
        component_tiles = [tile_by_id[tile_id] for tile_id in component_ids if tile_id in tile_by_id]
        if not component_tiles:
            continue
        if preferred_id and preferred_id in component_ids:
            anchors.append(tile_by_id[preferred_id])
            continue
        anchors.append(choose_anchor_tile(component_tiles) or component_tiles[0])
    return anchors


def get_nominal_position(tile: TileRecord) -> tuple[float, float]:
    return float(tile.transform.nominal_x), float(tile.transform.nominal_y)


def _constraint_weight(confidence: float) -> float:
    value = float(confidence)
    if not np.isfinite(value) or value <= 0:
        return 1.0
    return value


def _mark_project_refinement_state(
    project: AtlasProject,
    *,
    constraints: Iterable[NeighborConstraint] | None = None,
    placement_mode: str,
    refinement_status: str,
    refined_tile_count: int,
) -> None:
    project.metadata.extra_metadata["atlas_stitch_placement_mode"] = placement_mode
    project.metadata.extra_metadata["atlas_stitch_refinement_status"] = refinement_status
    project.metadata.extra_metadata["atlas_stitch_refined_tile_count"] = refined_tile_count
    if constraints is not None:
        constraint_summary = summarize_neighbor_constraints(project, list(constraints))
        refined_summary = summarize_refined_positions(project)
        project.metadata.extra_metadata["atlas_stitch_constraint_count"] = constraint_summary["constraint_count"]
        project.metadata.extra_metadata["atlas_stitch_constrained_tile_count"] = constraint_summary["constrained_tile_count"]
        project.metadata.extra_metadata["atlas_stitch_isolated_tile_count"] = constraint_summary["isolated_tile_count"]
        project.metadata.extra_metadata["atlas_stitch_anchor_component_count"] = constraint_summary["anchor_component_count"]
        project.metadata.extra_metadata["atlas_stitch_refined_tile_count"] = refined_summary["refined_tile_count"]
        project.metadata.extra_metadata["atlas_stitch_max_abs_shift_x"] = refined_summary["max_abs_shift_x"]
        project.metadata.extra_metadata["atlas_stitch_max_abs_shift_y"] = refined_summary["max_abs_shift_y"]
        project.metadata.extra_metadata["atlas_stitch_mean_shift_x"] = refined_summary["mean_shift_x"]
        project.metadata.extra_metadata["atlas_stitch_mean_shift_y"] = refined_summary["mean_shift_y"]
        project.metadata.extra_metadata["atlas_stitch_std_shift_x"] = refined_summary["std_shift_x"]
        project.metadata.extra_metadata["atlas_stitch_std_shift_y"] = refined_summary["std_shift_y"]
        project.metadata.extra_metadata["atlas_stitch_zero_like_refined_tile_count"] = refined_summary[
            "zero_like_refined_tile_count"
        ]


def _stable_coordinate(value: float) -> float:
    return float(round(float(value), 9))
