from __future__ import annotations

import numpy as np
import zarr
from tifffile import imwrite

from atlas_stitch.models import AtlasMetadata, AtlasProject, TileRecord
from atlas_stitch.ome_zarr_export import export_nominal_layout_to_omezarr


def test_nominal_atlas_export_writes_stitched_omezarr_with_metadata(tmp_path):
    tile_a = tmp_path / "tile_a.tif"
    tile_b = tmp_path / "tile_b.tif"
    imwrite(tile_a, np.array([[1, 2], [3, 4]], dtype=np.uint16))
    imwrite(tile_b, np.array([[5, 6], [7, 8]], dtype=np.uint16))

    project = AtlasProject(
        metadata=AtlasMetadata(
            atlas_name="demo_atlas",
            xml_path="/tmp/demo.xml",
            source_directory="/tmp/source_tiles",
            voxel_size_x=2.0,
            voxel_size_y=3.0,
        ),
        tiles=[
            TileRecord(
                tile_id="a",
                resolved_path=str(tile_a),
                start_x=10,
                start_y=20,
                exists=True,
            ),
            TileRecord(
                tile_id="b",
                resolved_path=str(tile_b),
                start_x=12,
                start_y=20,
                exists=True,
            ),
        ],
    )

    destination = export_nominal_layout_to_omezarr(project, str(tmp_path / "atlas"), chunk_size=2, build_pyramid=True)

    assert destination.name == "atlas.ome.zarr"
    root = zarr.open_group(str(destination), mode="r")
    base = np.asarray(zarr.open(str(destination / "0"), mode="r"))
    assert np.array_equal(base, np.array([[1, 2, 5, 6], [3, 4, 7, 8]], dtype=np.uint16))
    assert np.array_equal(np.asarray(zarr.open(str(destination / "1"), mode="r")), np.array([[1, 5]], dtype=np.uint16))
    assert root["0"].chunks == (2, 2)
    assert root.attrs["napari_atlas_stitch"]["atlas_stitch"] == {
        "kind": "nominal_atlas_export",
        "atlas_name": "demo_atlas",
        "xml_path": "/tmp/demo.xml",
        "tile_root": "/tmp/source_tiles",
        "tile_count": 2,
        "placement_mode": "nominal",
        "atlas_project_path": "",
        "pixel_size_x": 2.0,
        "pixel_size_y": 3.0,
        "pixel_size_unit": "",
        "bit_per_sample": "",
        "sample_per_pixel": "",
        "fusion_method": "overwrite",
        "export_version": 1,
    }
    multiscales = root.attrs["multiscales"]
    assert multiscales[0]["axes"][0]["name"] == "y"
    assert multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"] == [3.0, 2.0]


def test_nominal_atlas_export_infers_missing_tile_dimensions_from_tiff(tmp_path):
    tile_a = tmp_path / "tile_a.tif"
    tile_b = tmp_path / "tile_b.tif"
    imwrite(tile_a, np.arange(6, dtype=np.uint8).reshape(2, 3))
    imwrite(tile_b, np.full((2, 3), 9, dtype=np.uint8))

    project = AtlasProject(
        metadata=AtlasMetadata(atlas_name="atlas", xml_path="/tmp/input.xml", tile_root_override="/tmp/override"),
        tiles=[
            TileRecord(
                tile_id="a",
                resolved_path=str(tile_a),
                start_x=0,
                start_y=0,
                exists=True,
            ),
            TileRecord(
                tile_id="b",
                resolved_path=str(tile_b),
                start_x=3,
                start_y=0,
                exists=True,
            ),
        ],
    )

    destination = export_nominal_layout_to_omezarr(project, str(tmp_path / "atlas_out"), build_pyramid=False)

    base = np.asarray(zarr.open(str(destination / "0"), mode="r"))
    assert base.shape == (2, 6)
    assert np.array_equal(base[:, :3], np.arange(6, dtype=np.uint8).reshape(2, 3))
    assert np.array_equal(base[:, 3:], np.full((2, 3), 9, dtype=np.uint8))
    assert not (destination / "1").exists()
    attrs = zarr.open_group(str(destination), mode="r").attrs["napari_atlas_stitch"]["atlas_stitch"]
    assert attrs["tile_root"] == "/tmp/override"


def test_nominal_atlas_export_rejects_projects_without_exportable_tiles(tmp_path):
    project = AtlasProject(
        metadata=AtlasMetadata(atlas_name="empty"),
        tiles=[TileRecord(tile_id="missing", resolved_path="", start_x=None, start_y=None, exists=False)],
    )

    try:
        export_nominal_layout_to_omezarr(project, str(tmp_path / "atlas"))
    except ValueError as exc:
        assert "No exportable tiles" in str(exc)
    else:
        raise AssertionError("Expected nominal export to reject projects without exportable tiles.")


def test_nominal_atlas_export_reports_staged_progress(tmp_path):
    tile_a = tmp_path / "tile_a.tif"
    tile_b = tmp_path / "tile_b.tif"
    imwrite(tile_a, np.ones((2, 2), dtype=np.uint8))
    imwrite(tile_b, np.full((2, 2), 2, dtype=np.uint8))

    project = AtlasProject(
        metadata=AtlasMetadata(atlas_name="atlas", xml_path="/tmp/input.xml"),
        tiles=[
            TileRecord(tile_id="a", resolved_path=str(tile_a), start_x=0, start_y=0, exists=True),
            TileRecord(tile_id="b", resolved_path=str(tile_b), start_x=2, start_y=0, exists=True),
        ],
    )
    progress_events: list[tuple[str, int | None, int | None]] = []

    export_nominal_layout_to_omezarr(
        project,
        str(tmp_path / "atlas_progress"),
        build_pyramid=True,
        progress_callback=lambda stage, current, total: progress_events.append((stage, current, total)),
    )

    stages = [stage for stage, _current, _total in progress_events]
    assert "Preparing tiles" in stages
    assert "Reading tiles" in stages
    assert "Assembling atlas" in stages
    assert "Writing OME-Zarr" in stages
    assert "Finalizing metadata" in stages
    assert stages[-1] == "Export complete"


def test_nominal_atlas_export_writes_project_link_and_xml_metadata_to_attrs(tmp_path):
    tile = tmp_path / "tile.tif"
    imwrite(tile, np.ones((2, 2), dtype=np.uint8))
    project = AtlasProject(
        metadata=AtlasMetadata(
            atlas_name="linked",
            xml_path="/tmp/atlas.xml",
            source_directory="/tmp/source",
            voxel_size_x=0.006,
            voxel_size_y=0.006,
            extra_metadata={"pixel_size_unit": "µm", "bit_per_sample": "8", "sample_per_pixel": "1"},
        ),
        tiles=[TileRecord(tile_id="a", resolved_path=str(tile), start_x=0, start_y=0, exists=True)],
    )

    destination = export_nominal_layout_to_omezarr(
        project,
        str(tmp_path / "linked_atlas"),
        atlas_project_path="/tmp/project.atlas.json",
    )

    attrs = zarr.open_group(str(destination), mode="r").attrs["napari_atlas_stitch"]["atlas_stitch"]
    assert attrs["atlas_project_path"] == "/tmp/project.atlas.json"
    assert attrs["pixel_size_x"] == 0.006
    assert attrs["pixel_size_y"] == 0.006
    assert attrs["pixel_size_unit"] == "µm"
    assert attrs["bit_per_sample"] == "8"
    assert attrs["sample_per_pixel"] == "1"


def test_nominal_atlas_export_caps_pyramid_levels_to_avoid_lexicographic_reopen_issue(tmp_path):
    tile = tmp_path / "large_tile.tif"
    imwrite(tile, np.ones((2048, 2048), dtype=np.uint8))
    project = AtlasProject(
        metadata=AtlasMetadata(atlas_name="large", xml_path="/tmp/large.xml"),
        tiles=[TileRecord(tile_id="a", resolved_path=str(tile), start_x=0, start_y=0, exists=True)],
    )

    destination = export_nominal_layout_to_omezarr(project, str(tmp_path / "large_atlas"), build_pyramid=True)

    root = zarr.open_group(str(destination), mode="r")
    dataset_paths = [entry["path"] for entry in root.attrs["multiscales"][0]["datasets"]]
    shapes = [tuple(int(v) for v in np.asarray(zarr.open(str(destination / path), mode="r")).shape) for path in dataset_paths]

    assert dataset_paths == [str(index) for index in range(len(dataset_paths))]
    assert len(dataset_paths) <= 9
    assert all(shapes[index][0] > shapes[index + 1][0] for index in range(len(shapes) - 1))
    assert all(shapes[index][1] > shapes[index + 1][1] for index in range(len(shapes) - 1))
