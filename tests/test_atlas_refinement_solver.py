from __future__ import annotations

from atlas_stitch.models import AtlasMetadata, AtlasProject, TileRecord, TileTransform
from atlas_stitch.project_state import load_atlas_project, save_atlas_project
from atlas_stitch.refinement_solver import NeighborConstraint, solve_refined_tile_positions


def test_solve_refined_tile_positions_recovers_synthetic_grid_offsets():
    project = AtlasProject(
        metadata=AtlasMetadata(atlas_name="grid"),
        tiles=[
            TileRecord(tile_id="t00", transform=TileTransform(nominal_x=10.0, nominal_y=20.0)),
            TileRecord(tile_id="t01", transform=TileTransform(nominal_x=110.0, nominal_y=20.0)),
            TileRecord(tile_id="t10", transform=TileTransform(nominal_x=10.0, nominal_y=120.0)),
            TileRecord(tile_id="t11", transform=TileTransform(nominal_x=110.0, nominal_y=120.0)),
        ],
    )
    constraints = [
        NeighborConstraint("t00", "t01", dx=100.0, dy=0.0, confidence=1.0, direction="right"),
        NeighborConstraint("t00", "t10", dx=0.0, dy=100.0, confidence=1.0, direction="down"),
        NeighborConstraint("t01", "t11", dx=0.0, dy=100.0, confidence=1.0, direction="down"),
        NeighborConstraint("t10", "t11", dx=100.0, dy=0.0, confidence=1.0, direction="right"),
    ]

    solved = solve_refined_tile_positions(project, constraints)
    positions = {tile.tile_id: (tile.transform.refined_x, tile.transform.refined_y) for tile in solved.tiles}

    assert positions["t00"] == (10.0, 20.0)
    assert positions["t01"] == (110.0, 20.0)
    assert positions["t10"] == (10.0, 120.0)
    assert positions["t11"] == (110.0, 120.0)
    assert solved.metadata.extra_metadata["atlas_stitch_placement_mode"] == "refined_translation"
    assert solved.metadata.extra_metadata["atlas_stitch_refinement_status"] == "solved"


def test_solve_refined_tile_positions_keeps_anchor_tile_at_nominal_coordinate():
    project = AtlasProject(
        metadata=AtlasMetadata(atlas_name="anchor"),
        tiles=[
            TileRecord(tile_id="anchor", start_x=0.0, start_y=0.0, transform=TileTransform(nominal_x=5.0, nominal_y=7.0)),
            TileRecord(tile_id="neighbor", start_x=100.0, start_y=0.0, transform=TileTransform(nominal_x=105.0, nominal_y=7.0)),
        ],
    )
    constraints = [NeighborConstraint("anchor", "neighbor", dx=100.0, dy=0.0, confidence=2.0, direction="right")]

    solved = solve_refined_tile_positions(project, constraints)
    anchor = next(tile for tile in solved.tiles if tile.tile_id == "anchor")

    assert anchor.transform.refined_x == 5.0
    assert anchor.transform.refined_y == 7.0


def test_refined_positions_round_trip_through_atlas_project_save_load(tmp_path):
    project = AtlasProject(
        metadata=AtlasMetadata(atlas_name="roundtrip"),
        tiles=[
            TileRecord(tile_id="a", transform=TileTransform(nominal_x=0.0, nominal_y=0.0, refined_x=1.5, refined_y=2.5)),
            TileRecord(tile_id="b", transform=TileTransform(nominal_x=10.0, nominal_y=0.0, refined_x=11.5, refined_y=2.5)),
        ],
    )
    destination = tmp_path / "atlas_project.json"

    save_atlas_project(project, str(destination))
    restored = load_atlas_project(destination)

    assert restored.tiles[0].transform.refined_x == 1.5
    assert restored.tiles[0].transform.refined_y == 2.5
    assert restored.tiles[1].transform.refined_x == 11.5
    assert restored.tiles[1].transform.refined_y == 2.5


def test_solve_refined_tile_positions_anchors_each_disconnected_component():
    project = AtlasProject(
        metadata=AtlasMetadata(atlas_name="disconnected"),
        tiles=[
            TileRecord(tile_id="a0", start_x=0.0, start_y=0.0, transform=TileTransform(nominal_x=0.0, nominal_y=0.0)),
            TileRecord(tile_id="a1", start_x=2048.0, start_y=0.0, transform=TileTransform(nominal_x=2048.0, nominal_y=0.0)),
            TileRecord(tile_id="a2", start_x=4096.0, start_y=0.0, transform=TileTransform(nominal_x=4096.0, nominal_y=0.0)),
            TileRecord(tile_id="b0", start_x=0.0, start_y=2048.0, transform=TileTransform(nominal_x=0.0, nominal_y=2048.0)),
            TileRecord(tile_id="b1", start_x=2048.0, start_y=2048.0, transform=TileTransform(nominal_x=2048.0, nominal_y=2048.0)),
            TileRecord(tile_id="b2", start_x=4096.0, start_y=2048.0, transform=TileTransform(nominal_x=4096.0, nominal_y=2048.0)),
        ],
    )
    constraints = [
        NeighborConstraint("a0", "a1", dx=2048.0, dy=0.0, confidence=1.0, direction="right"),
        NeighborConstraint("a1", "a2", dx=2048.0, dy=0.0, confidence=1.0, direction="right"),
        NeighborConstraint("b0", "b1", dx=2048.0, dy=0.0, confidence=1.0, direction="right"),
        NeighborConstraint("b1", "b2", dx=2048.0, dy=0.0, confidence=1.0, direction="right"),
    ]

    solved = solve_refined_tile_positions(project, constraints)
    positions = {tile.tile_id: (tile.transform.refined_x, tile.transform.refined_y) for tile in solved.tiles}

    assert positions["a0"] == (0.0, 0.0)
    assert positions["a1"] == (2048.0, 0.0)
    assert positions["a2"] == (4096.0, 0.0)
    assert positions["b0"] == (0.0, 2048.0)
    assert positions["b1"] == (2048.0, 2048.0)
    assert positions["b2"] == (4096.0, 2048.0)
    assert len({positions["a0"], positions["a1"], positions["a2"], positions["b0"], positions["b1"], positions["b2"]}) == 6
    assert solved.metadata.extra_metadata["atlas_stitch_constraint_count"] == 4
    assert solved.metadata.extra_metadata["atlas_stitch_constrained_tile_count"] == 6
    assert solved.metadata.extra_metadata["atlas_stitch_isolated_tile_count"] == 0
    assert solved.metadata.extra_metadata["atlas_stitch_anchor_component_count"] == 2
    assert solved.metadata.extra_metadata["atlas_stitch_zero_like_refined_tile_count"] == 0
