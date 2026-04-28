from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .models import AtlasProject, TileRecord


def summarize_neighbor_constraints(project: AtlasProject, constraints: Iterable[Any]) -> dict[str, Any]:
    tiles = [tile for tile in project.tiles if tile.tile_id]
    tile_ids = {tile.tile_id for tile in tiles}
    constraint_list = [
        constraint
        for constraint in constraints
        if getattr(constraint, "tile_a_id", "") in tile_ids and getattr(constraint, "tile_b_id", "") in tile_ids
    ]
    constrained_ids: set[str] = set()
    for constraint in constraint_list:
        constrained_ids.add(str(getattr(constraint, "tile_a_id", "")))
        constrained_ids.add(str(getattr(constraint, "tile_b_id", "")))
    component_count = _connected_component_count(tiles, constraint_list)
    return {
        "tile_count": len(tiles),
        "constraint_count": len(constraint_list),
        "constrained_tile_count": len(constrained_ids),
        "isolated_tile_count": len(tiles) - len(constrained_ids),
        "anchor_component_count": component_count,
        "neighbor_pairs_total": int(project.metadata.extra_metadata.get("atlas_stitch_neighbor_pairs_total", 0) or 0),
        "neighbor_pairs_accepted": int(project.metadata.extra_metadata.get("atlas_stitch_neighbor_pairs_accepted", 0) or 0),
        "skip_reasons": dict(project.metadata.extra_metadata.get("atlas_stitch_neighbor_skip_reasons") or {}),
        "fallback_reasons": dict(project.metadata.extra_metadata.get("atlas_stitch_neighbor_fallback_reasons") or {}),
    }


def summarize_refined_positions(project: AtlasProject) -> dict[str, Any]:
    refined_tiles = [tile for tile in project.tiles if tile.transform.refined_x is not None and tile.transform.refined_y is not None]
    shifts_x: list[float] = []
    shifts_y: list[float] = []
    zero_like_refined_tile_count = 0
    for tile in refined_tiles:
        nominal_x = float(tile.transform.nominal_x)
        nominal_y = float(tile.transform.nominal_y)
        refined_x = float(tile.transform.refined_x)
        refined_y = float(tile.transform.refined_y)
        shifts_x.append(refined_x - nominal_x)
        shifts_y.append(refined_y - nominal_y)
        if abs(refined_x) < 1e-9 and abs(refined_y) < 1e-9 and (abs(nominal_x) > 1e-9 or abs(nominal_y) > 1e-9):
            zero_like_refined_tile_count += 1
    return {
        "refined_tile_count": len(refined_tiles),
        "max_abs_shift_x": _safe_stat(shifts_x, lambda values: np.max(np.abs(values))),
        "max_abs_shift_y": _safe_stat(shifts_y, lambda values: np.max(np.abs(values))),
        "mean_shift_x": _safe_stat(shifts_x, np.mean),
        "mean_shift_y": _safe_stat(shifts_y, np.mean),
        "std_shift_x": _safe_stat(shifts_x, np.std),
        "std_shift_y": _safe_stat(shifts_y, np.std),
        "zero_like_refined_tile_count": zero_like_refined_tile_count,
    }


def _connected_component_count(tiles: list[TileRecord], constraints: list[Any]) -> int:
    adjacency: dict[str, set[str]] = {tile.tile_id: set() for tile in tiles}
    for constraint in constraints:
        tile_a_id = str(getattr(constraint, "tile_a_id", ""))
        tile_b_id = str(getattr(constraint, "tile_b_id", ""))
        if tile_a_id not in adjacency or tile_b_id not in adjacency:
            continue
        adjacency[tile_a_id].add(tile_b_id)
        adjacency[tile_b_id].add(tile_a_id)
    visited: set[str] = set()
    component_count = 0
    for tile in tiles:
        if tile.tile_id in visited:
            continue
        component_count += 1
        stack = [tile.tile_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(neighbor for neighbor in adjacency[current] if neighbor not in visited)
    return component_count


def _safe_stat(values: list[float], fn) -> float:
    if not values:
        return 0.0
    return float(fn(np.asarray(values, dtype=float)))
