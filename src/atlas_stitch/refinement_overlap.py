from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path

import numpy as np
from skimage.registration import phase_cross_correlation
from tifffile import imread

from .models import AtlasProject, TileRecord
from .refinement_solver import NeighborConstraint, get_nominal_position

logger = logging.getLogger(__name__)

OVERLAP_FRACTION = 0.1
MIN_CONFIDENCE = 0.1
MIN_STRIP_STD = 1e-6
FALLBACK_CONSTRAINT_CONFIDENCE = 0.2
DEFAULT_ALIGNMENT_METHOD = "light_translation"
ROBUST_ALIGNMENT_METHOD = "robust_translation"
ALIGNMENT_METHODS = {DEFAULT_ALIGNMENT_METHOD, ROBUST_ALIGNMENT_METHOD}


def extract_overlap_strip(image: np.ndarray, side: str, fraction: float = OVERLAP_FRACTION) -> np.ndarray:
    data = np.asarray(image)
    if data.ndim != 2:
        raise ValueError("Overlap strip extraction requires a 2D image.")
    fraction = float(fraction)
    if not np.isfinite(fraction) or fraction <= 0:
        raise ValueError("fraction must be positive.")
    height, width = data.shape
    if side in {"left", "right"}:
        strip_width = max(1, int(round(width * min(fraction, 1.0))))
        return data[:, :strip_width] if side == "left" else data[:, width - strip_width :]
    if side in {"top", "bottom"}:
        strip_height = max(1, int(round(height * min(fraction, 1.0))))
        return data[:strip_height, :] if side == "top" else data[height - strip_height :, :]
    raise ValueError(f"Unsupported strip side: {side}")


def estimate_translation_phasecorr(
    tile_A_path: str | Path,
    tile_B_path: str | Path,
    direction: str,
    *,
    method: str = DEFAULT_ALIGNMENT_METHOD,
    overlap_fraction: float = OVERLAP_FRACTION,
) -> tuple[float, float, float] | None:
    result = _estimate_translation_phasecorr_detailed(
        tile_A_path,
        tile_B_path,
        direction,
        method=method,
        overlap_fraction=overlap_fraction,
    )
    if result["status"] != "ok":
        return None
    return float(result["dx"]), float(result["dy"]), float(result["confidence"])


def build_neighbor_constraints(
    project: AtlasProject,
    *,
    method: str = DEFAULT_ALIGNMENT_METHOD,
    overlap_fraction: float = OVERLAP_FRACTION,
) -> list[NeighborConstraint]:
    method = _normalize_alignment_method(method)
    overlap_fraction = _normalize_overlap_fraction(overlap_fraction)
    tiles_by_grid = {
        (tile.row, tile.col): tile
        for tile in project.tiles
        if tile.row is not None and tile.col is not None
    }
    constraints: list[NeighborConstraint] = []
    skip_reasons: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    pairs_total = 0
    for (row, col), tile_a in sorted(tiles_by_grid.items()):
        right_neighbor = tiles_by_grid.get((row, col + 1))
        if right_neighbor is not None:
            pairs_total += 1
            constraint, reason = _build_constraint_for_pair(
                tile_a,
                right_neighbor,
                direction="right_neighbor",
                method=method,
                overlap_fraction=overlap_fraction,
            )
            if constraint is not None:
                constraints.append(constraint)
                if reason.startswith("fallback_"):
                    fallback_reasons[reason] += 1
            else:
                skip_reasons[reason] += 1
        bottom_neighbor = tiles_by_grid.get((row + 1, col))
        if bottom_neighbor is not None:
            pairs_total += 1
            constraint, reason = _build_constraint_for_pair(
                tile_a,
                bottom_neighbor,
                direction="bottom_neighbor",
                method=method,
                overlap_fraction=overlap_fraction,
            )
            if constraint is not None:
                constraints.append(constraint)
                if reason.startswith("fallback_"):
                    fallback_reasons[reason] += 1
            else:
                skip_reasons[reason] += 1

    project.metadata.extra_metadata["atlas_stitch_refinement_method"] = method
    project.metadata.extra_metadata["atlas_stitch_overlap_fraction"] = overlap_fraction
    project.metadata.extra_metadata["atlas_stitch_neighbor_pairs_total"] = pairs_total
    project.metadata.extra_metadata["atlas_stitch_neighbor_pairs_accepted"] = len(constraints)
    project.metadata.extra_metadata["atlas_stitch_neighbor_skip_reasons"] = dict(skip_reasons)
    project.metadata.extra_metadata["atlas_stitch_neighbor_fallback_reasons"] = dict(fallback_reasons)
    return constraints


def _build_constraint_for_pair(
    tile_a: TileRecord,
    tile_b: TileRecord,
    *,
    direction: str,
    method: str,
    overlap_fraction: float,
) -> tuple[NeighborConstraint | None, str]:
    usable_reason = _tile_pair_usable_reason(tile_a, tile_b)
    if usable_reason is not None:
        return None, usable_reason

    estimate = _estimate_translation_phasecorr_detailed(
        tile_a.resolved_path,
        tile_b.resolved_path,
        direction,
        method=method,
        overlap_fraction=overlap_fraction,
    )
    if estimate["status"] != "ok":
        return _fallback_nominal_constraint(tile_a, tile_b, direction=direction, reason=str(estimate["status"]))

    correction_dx = float(estimate["dx"])
    correction_dy = float(estimate["dy"])
    confidence = float(estimate["confidence"])
    nominal_a = get_nominal_position(tile_a)
    nominal_b = get_nominal_position(tile_b)
    nominal_dx = nominal_b[0] - nominal_a[0]
    nominal_dy = nominal_b[1] - nominal_a[1]
    absolute_dx = nominal_dx + correction_dx
    absolute_dy = nominal_dy + correction_dy
    if not _translation_is_plausible(
        direction=direction,
        nominal_dx=nominal_dx,
        nominal_dy=nominal_dy,
        absolute_dx=absolute_dx,
        absolute_dy=absolute_dy,
        strip_shape=estimate["strip_shape"],
    ):
        logger.info(
            "%s ↔ %s rejected as implausible | nominal=(%.3f, %.3f) solved=(%.3f, %.3f)",
            tile_a.tile_id,
            tile_b.tile_id,
            nominal_dx,
            nominal_dy,
            absolute_dx,
            absolute_dy,
        )
        return _fallback_nominal_constraint(tile_a, tile_b, direction=direction, reason="implausible_translation")
    if confidence < MIN_CONFIDENCE:
        return _fallback_nominal_constraint(tile_a, tile_b, direction=direction, reason="poor_confidence")

    constraint = NeighborConstraint(
        tile_a_id=tile_a.tile_id,
        tile_b_id=tile_b.tile_id,
        dx=absolute_dx,
        dy=absolute_dy,
        confidence=confidence,
        direction=direction,
    )
    logger.info(
        "%s ↔ %s | correction dx=%.3f dy=%.3f confidence=%.3f",
        tile_a.tile_id,
        tile_b.tile_id,
        correction_dx,
        correction_dy,
        confidence,
    )
    return constraint, "accepted"


def _fallback_nominal_constraint(
    tile_a: TileRecord,
    tile_b: TileRecord,
    *,
    direction: str,
    reason: str,
) -> tuple[NeighborConstraint, str]:
    nominal_a = get_nominal_position(tile_a)
    nominal_b = get_nominal_position(tile_b)
    nominal_dx = nominal_b[0] - nominal_a[0]
    nominal_dy = nominal_b[1] - nominal_a[1]
    logger.info(
        "%s ↔ %s fallback to nominal spacing | reason=%s nominal=(%.3f, %.3f)",
        tile_a.tile_id,
        tile_b.tile_id,
        reason,
        nominal_dx,
        nominal_dy,
    )
    return (
        NeighborConstraint(
            tile_a_id=tile_a.tile_id,
            tile_b_id=tile_b.tile_id,
            dx=nominal_dx,
            dy=nominal_dy,
            confidence=FALLBACK_CONSTRAINT_CONFIDENCE,
            direction=direction,
        ),
        f"fallback_{reason}",
    )


def _estimate_translation_phasecorr_detailed(
    tile_A_path: str | Path,
    tile_B_path: str | Path,
    direction: str,
    *,
    method: str = DEFAULT_ALIGNMENT_METHOD,
    overlap_fraction: float = OVERLAP_FRACTION,
) -> dict[str, object]:
    method = _normalize_alignment_method(method)
    overlap_fraction = _normalize_overlap_fraction(overlap_fraction)
    try:
        image_a = _load_tile_image(tile_A_path)
        image_b = _load_tile_image(tile_B_path)
    except Exception:
        return {"status": "load_error"}

    if method == ROBUST_ALIGNMENT_METHOD:
        return _estimate_translation_phasecorr_robust(image_a, image_b, direction, overlap_fraction=overlap_fraction)
    return _estimate_translation_phasecorr_light(image_a, image_b, direction, overlap_fraction=overlap_fraction)


def _estimate_translation_phasecorr_light(
    image_a: np.ndarray,
    image_b: np.ndarray,
    direction: str,
    *,
    overlap_fraction: float,
) -> dict[str, object]:
    pair = _prepare_overlap_pair(image_a, image_b, direction, fraction=overlap_fraction)
    if pair["status"] != "ok":
        return pair
    strip_a = pair["strip_a"]
    strip_b = pair["strip_b"]
    dy_sign = pair["dy_sign"]
    try:
        shift, _error, _diffphase = phase_cross_correlation(strip_a, strip_b, upsample_factor=10)
    except Exception:
        return {"status": "phasecorr_error"}
    return _estimate_result_from_shift(strip_a, strip_b, shift, dy_sign=dy_sign)


def _estimate_translation_phasecorr_robust(
    image_a: np.ndarray,
    image_b: np.ndarray,
    direction: str,
    *,
    overlap_fraction: float,
) -> dict[str, object]:
    candidate_fractions = tuple(sorted({_normalize_overlap_fraction(overlap_fraction * scale) for scale in (0.8, 1.0, 1.5, 2.0)}))
    best: dict[str, object] | None = None
    statuses: Counter[str] = Counter()
    for fraction in candidate_fractions:
        pair = _prepare_overlap_pair(image_a, image_b, direction, fraction=fraction)
        if pair["status"] != "ok":
            statuses[str(pair["status"])] += 1
            continue
        strip_a = pair["strip_a"]
        strip_b = pair["strip_b"]
        dy_sign = float(pair["dy_sign"])
        prepared_a = _prepare_strip_for_phasecorr(strip_a)
        prepared_b = _prepare_strip_for_phasecorr(strip_b)
        if _strip_has_low_variance(prepared_a):
            statuses["low_variance_a"] += 1
            continue
        if _strip_has_low_variance(prepared_b):
            statuses["low_variance_b"] += 1
            continue
        try:
            shift, _error, _diffphase = phase_cross_correlation(prepared_a, prepared_b, upsample_factor=20)
        except Exception:
            statuses["phasecorr_error"] += 1
            continue
        candidate = _estimate_result_from_shift(strip_a, strip_b, shift, dy_sign=dy_sign)
        candidate["confidence"] = min(1.0, float(candidate["confidence"]) * _robust_fraction_bonus(fraction))
        candidate["fraction"] = fraction
        if best is None or float(candidate["confidence"]) > float(best["confidence"]):
            best = candidate
    if best is not None:
        return best
    if statuses:
        return {"status": statuses.most_common(1)[0][0]}
    return {"status": "phasecorr_error"}


def _prepare_overlap_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    direction: str,
    *,
    fraction: float,
) -> dict[str, object]:
    if direction == "right_neighbor":
        strip_a = extract_overlap_strip(image_a, "right", fraction=fraction)
        strip_b = extract_overlap_strip(image_b, "left", fraction=fraction)
        dy_sign = 1.0
    elif direction == "bottom_neighbor":
        strip_a = extract_overlap_strip(image_a, "bottom", fraction=fraction)
        strip_b = extract_overlap_strip(image_b, "top", fraction=fraction)
        dy_sign = -1.0
    else:
        raise ValueError(f"Unsupported neighbor direction: {direction}")

    common = _crop_to_common_shape(strip_a, strip_b)
    if common is None:
        return {"status": "dimension_mismatch"}
    strip_a, strip_b = common
    if _strip_has_low_variance(strip_a):
        return {"status": "low_variance_a"}
    if _strip_has_low_variance(strip_b):
        return {"status": "low_variance_b"}
    return {
        "status": "ok",
        "strip_a": strip_a,
        "strip_b": strip_b,
        "dy_sign": dy_sign,
    }


def _estimate_result_from_shift(
    strip_a: np.ndarray,
    strip_b: np.ndarray,
    shift: np.ndarray,
    *,
    dy_sign: float,
) -> dict[str, object]:
    dy = float(dy_sign * shift[0])
    dx = float(shift[1])
    confidence = _confidence_from_aligned_overlap(strip_a, strip_b, shift)
    return {
        "status": "ok",
        "dx": dx,
        "dy": dy,
        "confidence": confidence,
        "strip_shape": tuple(int(v) for v in strip_a.shape),
    }


def _tile_pair_usable_reason(tile_a: TileRecord, tile_b: TileRecord) -> str | None:
    if not _tile_is_usable(tile_a):
        return "missing_file_a"
    if not _tile_is_usable(tile_b):
        return "missing_file_b"
    return None


def _tile_is_usable(tile: TileRecord) -> bool:
    return bool(tile.resolved_path and tile.exists and Path(tile.resolved_path).exists())


def _load_tile_image(path: str | Path) -> np.ndarray:
    data = imread(Path(path))
    array = np.asarray(data)
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Only 2D atlas tiles are currently supported: {path}")
    return np.asarray(array, dtype=np.float32)


def _crop_to_common_shape(image_a: np.ndarray, image_b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    min_height = min(image_a.shape[0], image_b.shape[0])
    min_width = min(image_a.shape[1], image_b.shape[1])
    if min_height <= 1 or min_width <= 1:
        return None
    return _center_crop(image_a, min_height, min_width), _center_crop(image_b, min_height, min_width)


def _center_crop(image: np.ndarray, height: int, width: int) -> np.ndarray:
    start_y = max(0, (image.shape[0] - height) // 2)
    start_x = max(0, (image.shape[1] - width) // 2)
    return image[start_y : start_y + height, start_x : start_x + width]


def _prepare_strip_for_phasecorr(strip: np.ndarray) -> np.ndarray:
    data = np.asarray(strip, dtype=np.float32)
    if data.size == 0:
        return data
    lower = float(np.nanpercentile(data, 1.0))
    upper = float(np.nanpercentile(data, 99.0))
    if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
        data = np.clip(data, lower, upper)
    data = data - float(np.nanmedian(data))
    std = float(np.nanstd(data))
    if std > 0:
        data = data / std
    window_y = np.hanning(max(2, data.shape[0])).astype(np.float32)
    window_x = np.hanning(max(2, data.shape[1])).astype(np.float32)
    window = np.outer(window_y, window_x)
    return data * window


def _strip_has_low_variance(strip: np.ndarray) -> bool:
    return bool(np.nanstd(strip) < MIN_STRIP_STD)


def _translation_is_plausible(
    *,
    direction: str,
    nominal_dx: float,
    nominal_dy: float,
    absolute_dx: float,
    absolute_dy: float,
    strip_shape: tuple[int, int],
) -> bool:
    strip_height, strip_width = strip_shape
    if direction == "right_neighbor":
        parallel_tolerance = max(64.0, strip_width * 0.75)
        orthogonal_tolerance = max(32.0, strip_height * 0.25)
        return abs(absolute_dx - nominal_dx) <= parallel_tolerance and abs(absolute_dy - nominal_dy) <= orthogonal_tolerance
    if direction == "bottom_neighbor":
        parallel_tolerance = max(64.0, strip_height * 0.75)
        orthogonal_tolerance = max(32.0, strip_width * 0.25)
        return abs(absolute_dy - nominal_dy) <= parallel_tolerance and abs(absolute_dx - nominal_dx) <= orthogonal_tolerance
    return False


def _confidence_from_aligned_overlap(reference: np.ndarray, moving: np.ndarray, shift: np.ndarray) -> float:
    if not np.all(np.isfinite(shift)):
        return 0.0
    dy = int(round(float(shift[0])))
    dx = int(round(float(shift[1])))

    ref_y0 = max(0, dy)
    mov_y0 = max(0, -dy)
    ref_x0 = max(0, dx)
    mov_x0 = max(0, -dx)
    overlap_h = min(reference.shape[0] - ref_y0, moving.shape[0] - mov_y0)
    overlap_w = min(reference.shape[1] - ref_x0, moving.shape[1] - mov_x0)
    if overlap_h <= 1 or overlap_w <= 1:
        return 0.0

    ref_patch = reference[ref_y0 : ref_y0 + overlap_h, ref_x0 : ref_x0 + overlap_w]
    mov_patch = moving[mov_y0 : mov_y0 + overlap_h, mov_x0 : mov_x0 + overlap_w]
    if _strip_has_low_variance(ref_patch) or _strip_has_low_variance(mov_patch):
        return 0.0
    corr = np.corrcoef(ref_patch.ravel(), mov_patch.ravel())[0, 1]
    if not np.isfinite(corr):
        return 0.0
    overlap_ratio = float((overlap_h * overlap_w) / max(1, reference.shape[0] * reference.shape[1]))
    corr_score = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
    return float(np.clip(corr_score * np.sqrt(max(0.0, overlap_ratio)), 0.0, 1.0))


def _robust_fraction_bonus(fraction: float) -> float:
    return 1.0 + max(0.0, float(fraction) - OVERLAP_FRACTION)


def _normalize_alignment_method(method: str) -> str:
    value = str(method or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"", "phasecorr", "default", "light", "light_phasecorr", "light_translation"}:
        return DEFAULT_ALIGNMENT_METHOD
    if value in {"robust", "robust_phasecorr", "robust_translation"}:
        return ROBUST_ALIGNMENT_METHOD
    raise ValueError(f"Unsupported alignment method: {method}")


def _normalize_overlap_fraction(value: float) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return OVERLAP_FRACTION
    if not np.isfinite(fraction):
        return OVERLAP_FRACTION
    return float(np.clip(fraction, 0.01, 1.0))
