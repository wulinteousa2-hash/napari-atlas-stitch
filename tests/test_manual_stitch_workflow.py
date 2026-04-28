import numpy as np
import pytest
from pathlib import Path
from tifffile import imwrite
import zarr


from atlas_stitch.models import (
    AtlasMetadata,
    AtlasProject,
    TileRecord,
    TileTransform,
)
from atlas_stitch.ome_zarr_export import (
    export_nominal_layout_to_omezarr,
)
from atlas_stitch.widget import _preview_position


def _make_test_tile(
    tmp_path: Path,
    tile_id: str,
    file_name: str,
    data: np.ndarray,
    *,
    nominal_x: float,
    nominal_y: float,
    manual_x: float | None = None,
    manual_y: float | None = None,
) -> TileRecord:
    path = tmp_path / file_name
    imwrite(path, data)

    return TileRecord(
        tile_id=tile_id,
        file_name=file_name,
        source_path=str(path),
        resolved_path=str(path),
        row=0,
        col=0,
        start_x=nominal_x,
        start_y=nominal_y,
        width=int(data.shape[1]),
        height=int(data.shape[0]),
        exists=True,
        transform=TileTransform(
            nominal_x=nominal_x,
            nominal_y=nominal_y,
            manual_x=manual_x,
            manual_y=manual_y,
        ),
        metadata={},
    )


def test_tiletransform_manual_roundtrip():
    transform = TileTransform(
        nominal_x=10.0,
        nominal_y=20.0,
        refined_x=11.5,
        refined_y=21.5,
        manual_x=100.0,
        manual_y=200.0,
    )

    payload = transform.to_dict()
    rebuilt = TileTransform.from_dict(payload)

    assert rebuilt.nominal_x == 10.0
    assert rebuilt.nominal_y == 20.0
    assert rebuilt.refined_x == 11.5
    assert rebuilt.refined_y == 21.5
    assert rebuilt.manual_x == 100.0
    assert rebuilt.manual_y == 200.0


def test_preview_position_prefers_manual_then_nominal():
    tile = TileRecord(
        tile_id="tile_001",
        file_name="tile_001.tif",
        source_path="tile_001.tif",
        resolved_path="tile_001.tif",
        start_x=5.0,
        start_y=7.0,
        exists=False,
        transform=TileTransform(
            nominal_x=5.0,
            nominal_y=7.0,
            manual_x=50.0,
            manual_y=70.0,
        ),
    )

    manual_pos = _preview_position(tile, "manual")
    nominal_pos = _preview_position(tile, "nominal")
    refined_fallback = _preview_position(tile, "refined")

    assert manual_pos == (50.0, 70.0)
    assert nominal_pos == (5.0, 7.0)
    # No refined set, so refined mode should fall back to nominal
    assert refined_fallback == (5.0, 7.0)


def test_preview_position_manual_falls_back_to_nominal_when_missing():
    tile = TileRecord(
        tile_id="tile_002",
        file_name="tile_002.tif",
        source_path="tile_002.tif",
        resolved_path="tile_002.tif",
        start_x=12.0,
        start_y=14.0,
        exists=False,
        transform=TileTransform(
            nominal_x=12.0,
            nominal_y=14.0,
            manual_x=None,
            manual_y=None,
        ),
    )

    pos = _preview_position(tile, "manual")
    assert pos == (12.0, 14.0)


def test_export_manual_placement_creates_omezarr(tmp_path: Path):
    tile1 = _make_test_tile(
        tmp_path,
        "tile_001",
        "tile_001.tif",
        np.full((16, 16), 10, dtype=np.uint16),
        nominal_x=0.0,
        nominal_y=0.0,
        manual_x=0.0,
        manual_y=0.0,
    )
    tile2 = _make_test_tile(
        tmp_path,
        "tile_002",
        "tile_002.tif",
        np.full((16, 16), 20, dtype=np.uint16),
        nominal_x=100.0,
        nominal_y=100.0,
        manual_x=12.0,
        manual_y=0.0,
    )

    project = AtlasProject(
        metadata=AtlasMetadata(
            atlas_name="manual_test",
            xml_path="dummy.ve-mif",
            source_directory=str(tmp_path),
            tile_count=2,
            voxel_size_x=1.0,
            voxel_size_y=1.0,
        ),
        tiles=[tile1, tile2],
    )

    out_path = tmp_path / "manual_test_export"
    exported = export_nominal_layout_to_omezarr(
        project,
        str(out_path),
        chunk_size=8,
        build_pyramid=False,
        placement_mode="manual",
    )

    assert exported.exists()
    assert exported.suffix == ".zarr"

    root = zarr.open_group(str(exported), mode="r")
    assert "napari_atlas_stitch" in root.attrs
    atlas_meta = root.attrs["napari_atlas_stitch"]["atlas_stitch"]

    assert atlas_meta["placement_mode"] == "manual"
    assert atlas_meta["tile_count"] == 2
    assert atlas_meta["atlas_name"] == "manual_test"


def test_export_manual_subset_only_exports_worked_tiles(tmp_path: Path):
    tile1 = _make_test_tile(
        tmp_path,
        "tile_001",
        "tile_001.tif",
        np.full((16, 16), 111, dtype=np.uint16),
        nominal_x=0.0,
        nominal_y=0.0,
        manual_x=0.0,
        manual_y=0.0,
    )
    tile2 = _make_test_tile(
        tmp_path,
        "tile_002",
        "tile_002.tif",
        np.full((16, 16), 222, dtype=np.uint16),
        nominal_x=50.0,
        nominal_y=0.0,
        manual_x=8.0,
        manual_y=0.0,
    )

    project = AtlasProject(
        metadata=AtlasMetadata(
            atlas_name="subset_test",
            xml_path="dummy.ve-mif",
            source_directory=str(tmp_path),
            tile_count=2,
            voxel_size_x=1.0,
            voxel_size_y=1.0,
        ),
        tiles=[tile1, tile2],
    )

    out_path = tmp_path / "subset_export"
    exported = export_nominal_layout_to_omezarr(
        project,
        str(out_path),
        chunk_size=8,
        build_pyramid=False,
        placement_mode="manual",
        tile_ids=["tile_002"],
    )

    assert exported.exists()

    root = zarr.open_group(str(exported), mode="r")
    atlas_meta = root.attrs["napari_atlas_stitch"]["atlas_stitch"]

    assert atlas_meta["placement_mode"] == "manual"
    assert atlas_meta["tile_count"] == 1
    assert atlas_meta["tile_ids"] == ["tile_002"]


def test_export_manual_fails_if_requested_tile_has_no_manual_position(tmp_path: Path):
    tile1 = _make_test_tile(
        tmp_path,
        "tile_001",
        "tile_001.tif",
        np.full((16, 16), 10, dtype=np.uint16),
        nominal_x=0.0,
        nominal_y=0.0,
        manual_x=None,
        manual_y=None,
    )

    project = AtlasProject(
        metadata=AtlasMetadata(
            atlas_name="manual_fail_test",
            xml_path="dummy.ve-mif",
            source_directory=str(tmp_path),
            tile_count=1,
        ),
        tiles=[tile1],
    )

    with pytest.raises(ValueError, match="No exportable tiles were found"):
        export_nominal_layout_to_omezarr(
            project,
            str(tmp_path / "fail_export"),
            build_pyramid=False,
            placement_mode="manual",
            tile_ids=["tile_001"],
        )
