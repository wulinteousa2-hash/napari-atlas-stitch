from __future__ import annotations

import numpy as np
from tifffile import imwrite

from atlas_stitch.models import AtlasMetadata, AtlasProject, TileRecord, TileTransform
from atlas_stitch.refinement_overlap import (
    ROBUST_ALIGNMENT_METHOD,
    build_neighbor_constraints,
    estimate_translation_phasecorr,
    extract_overlap_strip,
)
from atlas_stitch.refinement_solver import solve_refined_tile_positions


def test_extract_overlap_strip_returns_expected_slices():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)

    assert extract_overlap_strip(image, "left", fraction=0.2).shape == (10, 2)
    assert extract_overlap_strip(image, "right", fraction=0.2).shape == (10, 2)
    assert extract_overlap_strip(image, "top", fraction=0.2).shape == (2, 10)
    assert extract_overlap_strip(image, "bottom", fraction=0.2).shape == (2, 10)


def test_estimate_translation_phasecorr_returns_overlap_correction_for_right_neighbor(tmp_path):
    rng = np.random.default_rng(0)
    height = 80
    width = 100
    nominal_dx = 90
    correction_dx = 3
    correction_dy = 2
    canvas = rng.normal(size=(height + correction_dy + 10, width + nominal_dx + correction_dx + 10)).astype(np.float32)
    tile_a = canvas[:height, :width]
    tile_b = canvas[correction_dy : correction_dy + height, nominal_dx + correction_dx : nominal_dx + correction_dx + width]

    path_a = tmp_path / "tile_a.tif"
    path_b = tmp_path / "tile_b.tif"
    imwrite(path_a, tile_a)
    imwrite(path_b, tile_b)

    estimate = estimate_translation_phasecorr(path_a, path_b, "right_neighbor")

    assert estimate is not None
    dx, dy, confidence = estimate
    assert abs(dx - correction_dx) <= 0.25
    assert abs(dy - correction_dy) <= 0.25
    assert 0.0 <= confidence <= 1.0


def test_build_neighbor_constraints_and_solver_recover_small_grid_offsets(tmp_path):
    rng = np.random.default_rng(1)
    height = 80
    width = 100
    nominal_dx = 90
    nominal_dy = 70
    actual_positions = {
        "t00": (5, 7),
        "t01": (5 + nominal_dx + 2, 7 + 1),
        "t10": (5 + 1, 7 + nominal_dy + 1),
        "t11": (5 + nominal_dx + 2, 7 + nominal_dy + 2),
    }
    max_x = max(x for x, _y in actual_positions.values()) + width + 10
    max_y = max(y for _x, y in actual_positions.values()) + height + 10
    canvas = rng.normal(size=(max_y, max_x)).astype(np.float32)

    tiles: list[TileRecord] = []
    for tile_id, row, col, nominal_x, nominal_y in (
        ("t00", 0, 0, 5.0, 7.0),
        ("t01", 0, 1, 5.0 + nominal_dx, 7.0),
        ("t10", 1, 0, 5.0, 7.0 + nominal_dy),
        ("t11", 1, 1, 5.0 + nominal_dx, 7.0 + nominal_dy),
    ):
        x, y = actual_positions[tile_id]
        image = canvas[y : y + height, x : x + width]
        path = tmp_path / f"{tile_id}.tif"
        imwrite(path, image)
        tiles.append(
            TileRecord(
                tile_id=tile_id,
                resolved_path=str(path),
                row=row,
                col=col,
                start_x=nominal_x,
                start_y=nominal_y,
                exists=True,
                transform=TileTransform(nominal_x=nominal_x, nominal_y=nominal_y),
            )
        )

    project = AtlasProject(metadata=AtlasMetadata(atlas_name="synthetic"), tiles=tiles)

    constraints = build_neighbor_constraints(project)
    solved = solve_refined_tile_positions(project, constraints)
    solved_positions = {tile.tile_id: (tile.transform.refined_x, tile.transform.refined_y) for tile in solved.tiles}

    assert len(constraints) == 4
    assert abs(solved_positions["t00"][0] - 5.0) <= 0.5
    assert abs(solved_positions["t00"][1] - 7.0) <= 0.5
    assert abs(solved_positions["t01"][0] - actual_positions["t01"][0]) <= 1.0
    assert abs(solved_positions["t01"][1] - actual_positions["t01"][1]) <= 1.0
    assert abs(solved_positions["t10"][0] - actual_positions["t10"][0]) <= 1.0
    assert abs(solved_positions["t10"][1] - actual_positions["t10"][1]) <= 1.0


def test_build_neighbor_constraints_records_skip_reasons_for_missing_pair(tmp_path):
    tile_path = tmp_path / "tile_a.tif"
    imwrite(tile_path, np.ones((16, 16), dtype=np.float32))
    project = AtlasProject(
        metadata=AtlasMetadata(atlas_name="missing"),
        tiles=[
            TileRecord(
                tile_id="a",
                resolved_path=str(tile_path),
                row=0,
                col=0,
                start_x=0.0,
                start_y=0.0,
                exists=True,
                transform=TileTransform(nominal_x=0.0, nominal_y=0.0),
            ),
            TileRecord(
                tile_id="b",
                resolved_path=str(tmp_path / "missing.tif"),
                row=0,
                col=1,
                start_x=10.0,
                start_y=0.0,
                exists=False,
                transform=TileTransform(nominal_x=10.0, nominal_y=0.0),
            ),
        ],
    )

    constraints = build_neighbor_constraints(project)

    assert constraints == []
    assert project.metadata.extra_metadata["atlas_stitch_neighbor_pairs_total"] == 1
    assert project.metadata.extra_metadata["atlas_stitch_neighbor_pairs_accepted"] == 0
    assert project.metadata.extra_metadata["atlas_stitch_neighbor_skip_reasons"]["missing_file_b"] == 1


def test_build_neighbor_constraints_falls_back_to_nominal_for_low_variance_pair(tmp_path):
    tile_a_path = tmp_path / "tile_a.tif"
    tile_b_path = tmp_path / "tile_b.tif"
    imwrite(tile_a_path, np.zeros((32, 32), dtype=np.float32))
    imwrite(tile_b_path, np.zeros((32, 32), dtype=np.float32))
    project = AtlasProject(
        metadata=AtlasMetadata(atlas_name="fallback"),
        tiles=[
            TileRecord(
                tile_id="a",
                resolved_path=str(tile_a_path),
                row=0,
                col=0,
                start_x=0.0,
                start_y=0.0,
                exists=True,
                transform=TileTransform(nominal_x=0.0, nominal_y=0.0),
            ),
            TileRecord(
                tile_id="b",
                resolved_path=str(tile_b_path),
                row=0,
                col=1,
                start_x=20.0,
                start_y=0.0,
                exists=True,
                transform=TileTransform(nominal_x=20.0, nominal_y=0.0),
            ),
        ],
    )

    constraints = build_neighbor_constraints(project)

    assert len(constraints) == 1
    assert constraints[0].dx == 20.0
    assert constraints[0].dy == 0.0
    assert project.metadata.extra_metadata["atlas_stitch_neighbor_fallback_reasons"]["fallback_low_variance_a"] == 1


def test_estimate_translation_phasecorr_handles_mixed_tile_sizes_by_common_crop(tmp_path):
    rng = np.random.default_rng(2)
    canvas = rng.normal(size=(120, 180)).astype(np.float32)
    tile_a = canvas[:80, :100]
    tile_b = canvas[1:79, 92:180]
    path_a = tmp_path / "tile_a.tif"
    path_b = tmp_path / "tile_b.tif"
    imwrite(path_a, tile_a)
    imwrite(path_b, tile_b)

    estimate = estimate_translation_phasecorr(path_a, path_b, "right_neighbor")

    assert estimate is not None
    dx, dy, confidence = estimate
    assert abs(dx - 2.0) <= 0.5
    assert abs(dy) <= 1.0
    assert confidence >= 0.1


def test_build_neighbor_constraints_records_selected_alignment_method(tmp_path):
    rng = np.random.default_rng(3)
    height = 64
    width = 80
    nominal_dx = 72
    canvas = rng.normal(size=(height + 8, width + nominal_dx + 8)).astype(np.float32)
    tile_a = canvas[:height, :width]
    tile_b = canvas[1 : 1 + height, nominal_dx + 2 : nominal_dx + 2 + width]
    path_a = tmp_path / "tile_a.tif"
    path_b = tmp_path / "tile_b.tif"
    imwrite(path_a, tile_a)
    imwrite(path_b, tile_b)

    project = AtlasProject(
        metadata=AtlasMetadata(atlas_name="robust"),
        tiles=[
            TileRecord(
                tile_id="a",
                resolved_path=str(path_a),
                row=0,
                col=0,
                start_x=0.0,
                start_y=0.0,
                exists=True,
                transform=TileTransform(nominal_x=0.0, nominal_y=0.0),
            ),
            TileRecord(
                tile_id="b",
                resolved_path=str(path_b),
                row=0,
                col=1,
                start_x=float(nominal_dx),
                start_y=0.0,
                exists=True,
                transform=TileTransform(nominal_x=float(nominal_dx), nominal_y=0.0),
            ),
        ],
    )

    constraints = build_neighbor_constraints(project, method=ROBUST_ALIGNMENT_METHOD)

    assert len(constraints) == 1
    assert project.metadata.extra_metadata["atlas_stitch_refinement_method"] == ROBUST_ALIGNMENT_METHOD
    assert constraints[0].confidence > 0.1
    assert abs(constraints[0].dy - 1.0) <= 1.0
