from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zarr
from ome_zarr.writer import write_multiscale
from tifffile import imread

from .models import AtlasProject, TileRecord
from .seam_repair import preferred_tile_path

ATLAS_EXPORT_VERSION = 1
MAX_SAFE_PYRAMID_LEVELS = 9
FUSION_OVERWRITE = "overwrite"
FUSION_LINEAR_BLEND = "linear_blend"
FUSION_AVERAGE = "average"
FUSION_MAX = "max_intensity"
FUSION_MIN = "min_intensity"


def export_nominal_layout_to_omezarr(
    project: AtlasProject,
    output_path: str,
    *,
    chunk_size: int = 256,
    build_pyramid: bool = True,
    fusion_method: str = FUSION_OVERWRITE,
    atlas_project_path: str | None = None,
    progress_callback: Callable[[str, int | None, int | None], None] | None = None,
    placement_mode: str = "nominal",
    tile_ids: list[str] | None = None,
) -> Path:
    destination = Path(output_path).expanduser()
    if destination.suffix != ".zarr":
        destination = destination.with_suffix(".ome.zarr")
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)

    selected_ids = {str(tile_id).strip() for tile_id in (tile_ids or []) if str(tile_id).strip()}
    candidate_tiles = [
        tile for tile in project.tiles
        if (not selected_ids or tile.tile_id in selected_ids)
    ]
    valid_tiles = [tile for tile in candidate_tiles if _tile_is_exportable(tile, placement_mode=placement_mode)]
    if not valid_tiles:
        raise ValueError(
            f"No exportable tiles were found for placement mode '{placement_mode}'. "
            "Tiles need existing files and valid placement coordinates."
        )
    
    _notify_progress(progress_callback, "Preparing tiles", 0, len(valid_tiles))
    tile_specs: list[dict[str, Any]] = []
    for index, tile in enumerate(valid_tiles, start=1):
    
        tile_specs.append(_tile_spec(tile, placement_mode=placement_mode))
        _notify_progress(progress_callback, "Preparing tiles", index, len(valid_tiles))
    min_x = min(spec["start_x"] for spec in tile_specs)
    min_y = min(spec["start_y"] for spec in tile_specs)
    max_x = max(spec["start_x"] + spec["width"] for spec in tile_specs)
    max_y = max(spec["start_y"] + spec["height"] for spec in tile_specs)
    mosaic_width = max_x - min_x
    mosaic_height = max_y - min_y
    if mosaic_width <= 0 or mosaic_height <= 0:
        raise ValueError("Computed stitched atlas size is invalid.")

    dtype = np.result_type(*[spec["dtype"] for spec in tile_specs])
    fusion_method = _normalize_fusion_method(fusion_method)
    _notify_progress(progress_callback, "Assembling atlas", None, None)
    mosaic_dtype = np.float32 if fusion_method in {FUSION_LINEAR_BLEND, FUSION_AVERAGE} else dtype
    mosaic = np.zeros((mosaic_height, mosaic_width), dtype=mosaic_dtype)
    weight_sum = np.zeros((mosaic_height, mosaic_width), dtype=np.float32) if fusion_method in {FUSION_LINEAR_BLEND, FUSION_AVERAGE} else None
    initialized = np.zeros((mosaic_height, mosaic_width), dtype=bool) if fusion_method == FUSION_MIN else None
    for index, spec in enumerate(tile_specs, start=1):
        _notify_progress(progress_callback, "Reading tiles", index, len(tile_specs))
        tile_data = _load_tile_pixels(spec["path"], dtype=dtype)
        y0 = spec["start_y"] - min_y
        x0 = spec["start_x"] - min_x
        y1 = min(mosaic_height, y0 + tile_data.shape[0])
        x1 = min(mosaic_width, x0 + tile_data.shape[1])
        if y1 <= y0 or x1 <= x0:
            continue
        patch = tile_data[: y1 - y0, : x1 - x0]
        _fuse_patch(
            mosaic,
            patch,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            fusion_method=fusion_method,
            weight_sum=weight_sum,
            initialized=initialized,
        )

    if weight_sum is not None:
        valid = weight_sum > 0
        if np.any(valid):
            mosaic = mosaic.astype(np.float32, copy=False)
            mosaic[valid] = mosaic[valid] / weight_sum[valid]
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                mosaic = np.clip(np.rint(mosaic), info.min, info.max).astype(dtype)
            else:
                mosaic = mosaic.astype(dtype, copy=False)

    pyramid = _build_pyramid(mosaic, enabled=build_pyramid, progress_callback=progress_callback)
    root = zarr.open_group(str(destination), mode="w")
    _notify_progress(progress_callback, "Writing OME-Zarr", None, None)
    write_multiscale(
        pyramid,
        root,
        chunks=_normalize_chunks(chunk_size, mosaic.shape),
        axes=_ome_axes(project),
        coordinate_transformations=_coordinate_transformations(project, pyramid),
        compute=True,
        name=project.metadata.atlas_name or destination.stem,
        omero={"name": project.metadata.atlas_name or destination.stem},
    )
    _notify_progress(progress_callback, "Finalizing metadata", None, None)
    root.attrs["napari_atlas_stitch"] = {
        "atlas_stitch": _atlas_export_metadata(
            project,
            tile_count=len(valid_tiles),
            atlas_project_path=atlas_project_path,
            fusion_method=fusion_method,
            placement_mode=placement_mode,
            tile_ids=sorted(selected_ids),
        )
    }
    _notify_progress(progress_callback, "Export complete", 1, 1)
    return destination

def _tile_position(tile: TileRecord, placement_mode: str) -> tuple[float, float] | None:
    mode = str(placement_mode or "nominal").strip().lower()

    if mode == "manual":
        if tile.transform.manual_x is not None and tile.transform.manual_y is not None:
            return float(tile.transform.manual_x), float(tile.transform.manual_y)
        return None

    if mode == "refined":
        if tile.transform.refined_x is not None and tile.transform.refined_y is not None:
            return float(tile.transform.refined_x), float(tile.transform.refined_y)
        if tile.start_x is not None and tile.start_y is not None:
            return float(tile.start_x), float(tile.start_y)
        return float(tile.transform.nominal_x), float(tile.transform.nominal_y)

    if tile.start_x is not None and tile.start_y is not None:
        return float(tile.start_x), float(tile.start_y)
    return float(tile.transform.nominal_x), float(tile.transform.nominal_y)


def _tile_is_exportable(tile: TileRecord, *, placement_mode: str = "nominal") -> bool:
    path = preferred_tile_path(tile)
    return bool(path and Path(path).expanduser().exists() and _tile_position(tile, placement_mode) is not None)


def _tile_spec(tile: TileRecord, *, placement_mode: str = "nominal") -> dict[str, Any]:
    width = int(tile.width) if tile.width is not None else None
    height = int(tile.height) if tile.height is not None else None
    path = Path(preferred_tile_path(tile))
    sample = None
    if width is None or height is None:
        sample = _load_tile_pixels(path)
        height, width = sample.shape[:2]
    if width is None or height is None:
        raise ValueError(f"Could not infer tile size for {path}")

    position = _tile_position(tile, placement_mode)
    if position is None:
        raise ValueError(f"Tile {tile.tile_id} does not have valid placement for mode '{placement_mode}'.")

    start_x, start_y = position
    return {
        "path": path,
        "start_x": int(round(float(start_x))),
        "start_y": int(round(float(start_y))),
        "width": int(width),
        "height": int(height),
        "dtype": (sample.dtype if sample is not None else _load_tile_pixels(path).dtype),
    }

def _load_tile_pixels(path: Path, dtype: np.dtype | None = None) -> np.ndarray:
    data = imread(path)
    array = np.asarray(data)
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Only 2D atlas tiles are currently supported: {path}")
    if dtype is not None and array.dtype != dtype:
        return array.astype(dtype, copy=False)
    return array


def _fuse_patch(
    mosaic: np.ndarray,
    patch: np.ndarray,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    fusion_method: str,
    weight_sum: np.ndarray | None,
    initialized: np.ndarray | None,
) -> None:
    target = mosaic[y0:y1, x0:x1]
    if fusion_method == FUSION_OVERWRITE:
        target[...] = patch
        return
    if fusion_method == FUSION_MAX:
        target[...] = np.maximum(target, patch)
        return
    if fusion_method == FUSION_MIN:
        assert initialized is not None
        init_view = initialized[y0:y1, x0:x1]
        target[...] = np.where(init_view, np.minimum(target, patch), patch)
        init_view[...] = True
        return
    assert weight_sum is not None
    weights = np.ones(patch.shape, dtype=np.float32)
    if fusion_method == FUSION_LINEAR_BLEND:
        weights = _feather_weights(patch.shape)
    target[...] = target.astype(np.float32, copy=False) + patch.astype(np.float32, copy=False) * weights
    weight_sum[y0:y1, x0:x1] += weights


def _feather_weights(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    y_dist = np.minimum(yy + 1.0, height - yy)
    x_dist = np.minimum(xx + 1.0, width - xx)
    weights = np.minimum(y_dist, x_dist)
    max_value = float(np.max(weights)) if weights.size else 1.0
    if max_value <= 0:
        return np.ones(shape, dtype=np.float32)
    return np.clip(weights / max_value, 1e-3, 1.0).astype(np.float32)


def _normalize_fusion_method(method: str) -> str:
    value = str(method or "").strip().lower()
    if value in {FUSION_OVERWRITE, FUSION_LINEAR_BLEND, FUSION_AVERAGE, FUSION_MAX, FUSION_MIN}:
        return value
    return FUSION_OVERWRITE


def _build_pyramid(
    base: np.ndarray,
    *,
    enabled: bool,
    progress_callback: Callable[[str, int | None, int | None], None] | None = None,
) -> list[np.ndarray]:
    levels = [base]
    if not enabled:
        return levels
    _notify_progress(progress_callback, "Building pyramid", 1, 1)
    current = base
    # Napari builtin OME-Zarr loading can encounter lexicographic dataset-path
    # ordering issues once level names reach "10", "11", etc. Keep the pyramid
    # below that threshold so levels remain reopenable as a normal multiscale image.
    while min(current.shape) > 1 and len(levels) < MAX_SAFE_PYRAMID_LEVELS:
        next_level = current[::2, ::2]
        if next_level.shape == current.shape:
            break
        levels.append(next_level)
        current = next_level
        _notify_progress(progress_callback, "Building pyramid", len(levels), len(levels))
    return levels


def _normalize_chunks(chunk_size: int, shape: tuple[int, int]) -> tuple[int, int]:
    size = max(1, int(chunk_size))
    return tuple(min(int(dim), size) for dim in shape)


def _ome_axes(project: AtlasProject) -> list[dict[str, str]]:
    unit = "micrometer"
    return [
        {"name": "y", "type": "space", "unit": unit},
        {"name": "x", "type": "space", "unit": unit},
    ]


def _coordinate_transformations(project: AtlasProject, pyramid: list[np.ndarray]) -> list[list[dict[str, Any]]]:
    scale_y = float(project.metadata.voxel_size_y or 1.0)
    scale_x = float(project.metadata.voxel_size_x or 1.0)
    transforms: list[list[dict[str, Any]]] = []
    for level in pyramid:
        downsample_y = pyramid[0].shape[0] / max(1, level.shape[0])
        downsample_x = pyramid[0].shape[1] / max(1, level.shape[1])
        transforms.append(
            [
                {
                    "type": "scale",
                    "scale": [scale_y * downsample_y, scale_x * downsample_x],
                }
            ]
        )
    return transforms


def _notify_progress(
    progress_callback: Callable[[str, int | None, int | None], None] | None,
    stage: str,
    current: int | None,
    total: int | None,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, current, total)


def _atlas_export_metadata(
    project: AtlasProject,
    *,
    tile_count: int,
    atlas_project_path: str | None,
    fusion_method: str,
    placement_mode: str,
    tile_ids: list[str] | None = None,
) -> dict[str, Any]:
    extra = project.metadata.extra_metadata
    return {
        "kind": "nominal_atlas_export",
        "atlas_name": project.metadata.atlas_name,
        "xml_path": project.metadata.xml_path,
        "tile_root": project.metadata.tile_root_override or project.metadata.source_directory,
        "tile_count": tile_count,
        
        "placement_mode": str(placement_mode or "nominal").strip() or "nominal",
        "atlas_project_path": str(atlas_project_path or "").strip(),
        "tile_ids": list(tile_ids or []),        
        "pixel_size_x": project.metadata.voxel_size_x,
        "pixel_size_y": project.metadata.voxel_size_y,
        "pixel_size_unit": _metadata_text(
            extra,
            "pixel_size_unit",
            "pixel_unit",
            "voxel_size_unit",
            "voxel_unit",
            "physicalsizeunitx",
            "physicalsizeunity",
            "unit",
            "units",
        ),
        "bit_per_sample": _metadata_text(
            extra,
            "bit_per_sample",
            "bitpersample",
            "bits_per_sample",
            "bit_depth",
            "bitdepth",
        ),
        "sample_per_pixel": _metadata_text(
            extra,
            "sample_per_pixel",
            "sampleperpixel",
            "samples_per_pixel",
            "samplesperpixel",
        ),
        "fusion_method": fusion_method,
        "export_version": ATLAS_EXPORT_VERSION,
    }


def _metadata_text(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""
