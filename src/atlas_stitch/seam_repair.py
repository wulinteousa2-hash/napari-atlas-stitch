from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tifffile import imread, imwrite

from .models import AtlasProject, TileRecord
from .refinement_overlap import _estimate_translation_phasecorr_detailed
from .refinement_solver import get_nominal_position

DEFAULT_REPAIR_OVERLAP = 128
REPAIR_MODE_FULL_OVERLAP = "full_overlap"
REPAIR_MODE_ROI_GUIDED = "roi_guided"
BLEND_MODE_HARD_REPLACE = "hard_replace"
BLEND_MODE_FEATHER = "feather_blend"
DONOR_DIRECTIONS = {"left", "right", "top", "bottom"}


@dataclass
class RepairDonorSpec:
    tile_id: str
    direction: str
    priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"tile_id": self.tile_id, "direction": self.direction, "priority": self.priority}


@dataclass
class TileRepairRequest:
    target_tile_id: str
    donors: list[RepairDonorSpec]
    overlap_width: int = DEFAULT_REPAIR_OVERLAP
    repair_mode: str = REPAIR_MODE_FULL_OVERLAP
    blend_mode: str = BLEND_MODE_HARD_REPLACE
    roi_bounds: tuple[float, float, float, float] | None = None


@dataclass
class TileRepairResult:
    target_tile_id: str
    repaired_tile: np.ndarray
    confidence_map: np.ndarray
    attribution_map: np.ndarray
    roi_bounds: tuple[int, int, int, int] | None
    donors: list[RepairDonorSpec] = field(default_factory=list)
    output_dtype: str = "float32"


def preferred_tile_path(tile: TileRecord) -> str:
    repaired_path = str(tile.repaired_path or "").strip()
    if repaired_path and Path(repaired_path).expanduser().exists():
        return repaired_path
    return str(tile.resolved_path or "").strip()


def load_preferred_tile_pixels(tile: TileRecord, dtype: np.dtype | None = None) -> np.ndarray:
    path = preferred_tile_path(tile)
    if not path:
        raise FileNotFoundError(f"Tile {tile.tile_id} does not have a readable source path.")
    array = _read_2d_tile(Path(path))
    if dtype is not None and array.dtype != dtype:
        return array.astype(dtype, copy=False)
    return array


def reconstruct_tile_from_donors(project: AtlasProject, request: TileRepairRequest) -> TileRepairResult:
    if not request.target_tile_id.strip():
        raise ValueError("Select a target tile before previewing repair.")
    target_tile = _tile_by_id(project, request.target_tile_id)
    if target_tile is None:
        raise ValueError(f"Unknown target tile: {request.target_tile_id}")
    if not request.donors:
        raise ValueError("Select at least one donor tile.")

    target = _read_2d_tile(Path(target_tile.resolved_path))
    repaired = np.asarray(target, copy=True)
    confidence = np.full(target.shape, 0.5, dtype=np.float32)
    attribution = np.zeros(target.shape, dtype=np.uint16)
    overlap_width = max(1, int(request.overlap_width or DEFAULT_REPAIR_OVERLAP))

    donor_specs = sorted(
        [_normalize_donor_spec(spec) for spec in request.donors],
        key=lambda spec: (spec.priority, spec.tile_id),
    )
    requested_mask = _repair_mask(target.shape, donor_specs, overlap_width, request.repair_mode, request.roi_bounds)
    if not np.any(requested_mask):
        raise ValueError("Repair region is empty. Draw an ROI or increase overlap width.")
    confidence[requested_mask] = 0.0
    filled_mask = np.zeros(target.shape, dtype=bool)

    for donor_index, donor_spec in enumerate(donor_specs, start=1):
        donor_tile = _tile_by_id(project, donor_spec.tile_id)
        if donor_tile is None:
            continue
        donor = _read_2d_tile(Path(donor_tile.resolved_path))
        offset_y, offset_x = _target_to_donor_offset(target_tile, donor_tile, donor_spec.direction)
        target_slices, donor_slices = _mapped_overlap_slices(
            target_shape=target.shape,
            donor_shape=donor.shape,
            offset_y=offset_y,
            offset_x=offset_x,
        )
        if target_slices is None or donor_slices is None:
            continue
        target_view = repaired[target_slices]
        donor_view = donor[donor_slices]
        target_mask_view = requested_mask[target_slices]
        fill_mask_view = filled_mask[target_slices]
        replace_mask = target_mask_view & ~fill_mask_view
        if not np.any(replace_mask):
            continue
        if request.blend_mode == BLEND_MODE_FEATHER:
            alpha = _feather_alpha(replace_mask, overlap_width=overlap_width)
            target_view[:] = np.where(
                replace_mask,
                (1.0 - alpha) * target_view + alpha * donor_view,
                target_view,
            )
            confidence[target_slices] = np.where(
                replace_mask,
                np.maximum(confidence[target_slices], 0.5 + 0.5 * alpha),
                confidence[target_slices],
            )
        else:
            target_view[replace_mask] = donor_view[replace_mask]
            confidence[target_slices][replace_mask] = 1.0
        attribution[target_slices][replace_mask] = donor_index
        filled_mask[target_slices][replace_mask] = True

    unresolved = requested_mask & ~filled_mask
    repaired[unresolved] = target[unresolved]
    confidence[unresolved] = 0.0
    return TileRepairResult(
        target_tile_id=target_tile.tile_id,
        repaired_tile=repaired.astype(target.dtype, copy=False),
        confidence_map=confidence,
        attribution_map=attribution,
        roi_bounds=_normalized_roi_bounds(target.shape, request.roi_bounds),
        donors=donor_specs,
        output_dtype=str(target.dtype),
    )


def save_repair_outputs(
    tile: TileRecord,
    result: TileRepairResult,
    output_dir: str,
    *,
    repair_mode: str,
    blend_mode: str,
    overlap_width: int,
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    base = _safe_tile_stem(tile.tile_id)
    repaired_path = destination / f"{base}_repaired.tif"
    confidence_path = destination / f"{base}_confidence.tif"
    attribution_path = destination / f"{base}_attribution.tif"
    imwrite(repaired_path, result.repaired_tile)
    imwrite(confidence_path, result.confidence_map.astype(np.float32, copy=False))
    imwrite(attribution_path, result.attribution_map.astype(np.uint16, copy=False))
    history_entry = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "target_tile_id": tile.tile_id,
        "repaired_path": str(repaired_path),
        "confidence_path": str(confidence_path),
        "attribution_path": str(attribution_path),
        "repair_mode": repair_mode,
        "blend_mode": blend_mode,
        "overlap_width": int(overlap_width),
        "roi_bounds": list(result.roi_bounds) if result.roi_bounds is not None else [],
        "donors": [spec.to_dict() for spec in result.donors],
    }
    return {
        "repaired_path": str(repaired_path),
        "confidence_path": str(confidence_path),
        "attribution_path": str(attribution_path),
        "history_entry": history_entry,
    }


def _tile_by_id(project: AtlasProject, tile_id: str) -> TileRecord | None:
    wanted = str(tile_id or "").strip()
    if not wanted:
        return None
    for tile in project.tiles:
        if tile.tile_id == wanted:
            return tile
    return None


def _normalize_donor_spec(spec: RepairDonorSpec) -> RepairDonorSpec:
    direction = str(spec.direction or "").strip().lower()
    if direction not in DONOR_DIRECTIONS:
        raise ValueError(f"Unsupported donor direction: {spec.direction}")
    tile_id = str(spec.tile_id or "").strip()
    if not tile_id:
        raise ValueError("Donor tile ID is required.")
    return RepairDonorSpec(tile_id=tile_id, direction=direction, priority=int(spec.priority))


def _read_2d_tile(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Tile image not found: {path}")
    data = imread(path)
    array = np.asarray(data)
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Only 2D seam repair is supported: {path}")
    return array


def _repair_mask(
    target_shape: tuple[int, int],
    donors: list[RepairDonorSpec],
    overlap_width: int,
    repair_mode: str,
    roi_bounds: tuple[float, float, float, float] | None,
) -> np.ndarray:
    height, width = target_shape
    if repair_mode == REPAIR_MODE_ROI_GUIDED:
        normalized = _normalized_roi_bounds(target_shape, roi_bounds)
        if normalized is None:
            raise ValueError("ROI-guided repair requires a rectangular ROI.")
        y0, x0, y1, x1 = normalized
        mask = np.zeros(target_shape, dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask
    mask = np.zeros(target_shape, dtype=bool)
    strip = max(1, min(int(overlap_width), max(height, width)))
    for donor in donors:
        if donor.direction == "left":
            mask[:, : min(strip, width)] = True
        elif donor.direction == "right":
            mask[:, max(0, width - strip) : width] = True
        elif donor.direction == "top":
            mask[: min(strip, height), :] = True
        elif donor.direction == "bottom":
            mask[max(0, height - strip) : height, :] = True
    return mask


def _normalized_roi_bounds(
    target_shape: tuple[int, int],
    roi_bounds: tuple[float, float, float, float] | None,
) -> tuple[int, int, int, int] | None:
    if roi_bounds is None:
        return None
    height, width = target_shape
    y0, x0, y1, x1 = [int(round(float(value))) for value in roi_bounds]
    y0, y1 = sorted((max(0, y0), min(height, y1)))
    x0, x1 = sorted((max(0, x0), min(width, x1)))
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, x0, y1, x1


def _target_to_donor_offset(target: TileRecord, donor: TileRecord, direction: str) -> tuple[float, float]:
    target_nominal_x, target_nominal_y = get_nominal_position(target)
    donor_nominal_x, donor_nominal_y = get_nominal_position(donor)
    nominal_offset_x = float(donor_nominal_x - target_nominal_x)
    nominal_offset_y = float(donor_nominal_y - target_nominal_y)
    correction_x = 0.0
    correction_y = 0.0
    if target.resolved_path and donor.resolved_path:
        estimate = _directional_estimate(target, donor, direction)
        if estimate is not None:
            correction_x, correction_y = estimate
    if direction in {"left", "top"}:
        correction_x = -correction_x
        correction_y = -correction_y
    return nominal_offset_y + correction_y, nominal_offset_x + correction_x


def _directional_estimate(target: TileRecord, donor: TileRecord, direction: str) -> tuple[float, float] | None:
    if direction == "right":
        estimate = _estimate_translation_phasecorr_detailed(target.resolved_path, donor.resolved_path, "right_neighbor")
    elif direction == "bottom":
        estimate = _estimate_translation_phasecorr_detailed(target.resolved_path, donor.resolved_path, "bottom_neighbor")
    elif direction == "left":
        estimate = _estimate_translation_phasecorr_detailed(donor.resolved_path, target.resolved_path, "right_neighbor")
    elif direction == "top":
        estimate = _estimate_translation_phasecorr_detailed(donor.resolved_path, target.resolved_path, "bottom_neighbor")
    else:
        return None
    if estimate.get("status") != "ok":
        return None
    return float(estimate["dy"]), float(estimate["dx"])


def _mapped_overlap_slices(
    *,
    target_shape: tuple[int, int],
    donor_shape: tuple[int, int],
    offset_y: float,
    offset_x: float,
) -> tuple[tuple[slice, slice] | None, tuple[slice, slice] | None]:
    target_height, target_width = target_shape
    donor_height, donor_width = donor_shape
    y0 = int(round(offset_y))
    x0 = int(round(offset_x))
    y1 = y0 + donor_height
    x1 = x0 + donor_width
    target_y0 = max(0, y0)
    target_x0 = max(0, x0)
    target_y1 = min(target_height, y1)
    target_x1 = min(target_width, x1)
    if target_y1 <= target_y0 or target_x1 <= target_x0:
        return None, None
    donor_y0 = target_y0 - y0
    donor_x0 = target_x0 - x0
    donor_y1 = donor_y0 + (target_y1 - target_y0)
    donor_x1 = donor_x0 + (target_x1 - target_x0)
    return (slice(target_y0, target_y1), slice(target_x0, target_x1)), (
        slice(donor_y0, donor_y1),
        slice(donor_x0, donor_x1),
    )


def _feather_alpha(mask: np.ndarray, *, overlap_width: int) -> np.ndarray:
    alpha = np.zeros(mask.shape, dtype=np.float32)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return alpha
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    feather_radius = max(1.0, min(float(overlap_width) / 4.0, float(max(y1 - y0 + 1, x1 - x0 + 1))))
    ys, xs = np.indices(mask.shape, dtype=np.float32)
    distance = np.minimum.reduce(
        [
            ys - float(y0) + 1.0,
            float(y1) - ys + 1.0,
            xs - float(x0) + 1.0,
            float(x1) - xs + 1.0,
        ]
    )
    alpha[mask] = np.clip(distance[mask] / feather_radius, 0.0, 1.0)
    return alpha


def _safe_tile_stem(tile_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(tile_id or "").strip())
    return safe or "tile"
