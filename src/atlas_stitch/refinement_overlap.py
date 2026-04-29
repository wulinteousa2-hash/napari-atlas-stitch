from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from skimage.registration import phase_cross_correlation
from tifffile import imread

from .models import AtlasProject, TileRecord
from .refinement_solver import NeighborConstraint, get_nominal_position, solve_refined_tile_positions

logger = logging.getLogger(__name__)

OVERLAP_FRACTION = 0.1
MIN_CONFIDENCE = 0.20
DEFAULT_MAX_CORRECTION_PX = 50.0
MIN_STRIP_STD = 1e-6
FALLBACK_CONSTRAINT_CONFIDENCE = 0.2
DEFAULT_ALIGNMENT_METHOD = "light_translation"
ROBUST_ALIGNMENT_METHOD = "robust_translation"
ROBUST_OUTLIER_ALIGNMENT_METHOD = "robust_translation_outlier_rejection"
FEATURE_MATCHING_METHOD = "feature_matching"
CENTER_OUT_CLOCKWISE_METHOD = "center_out_clockwise"
CENTER_OUT_COUNTERCLOCKWISE_METHOD = "center_out_counterclockwise"
ECC_TRANSLATION_METHOD = "ecc_translation"
ALIGNMENT_METHODS = {
    DEFAULT_ALIGNMENT_METHOD,
    ROBUST_ALIGNMENT_METHOD,
    ROBUST_OUTLIER_ALIGNMENT_METHOD,
    FEATURE_MATCHING_METHOD,
    ECC_TRANSLATION_METHOD,
    CENTER_OUT_CLOCKWISE_METHOD,
    CENTER_OUT_COUNTERCLOCKWISE_METHOD,
}


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
    min_confidence: float = MIN_CONFIDENCE,
    max_correction_px: float | None = DEFAULT_MAX_CORRECTION_PX,
    progress_callback: Callable[[str, int | None, int | None], None] | None = None,
) -> list[NeighborConstraint]:
    method = _normalize_alignment_method(method)
    overlap_fraction = _normalize_overlap_fraction(overlap_fraction)
    min_confidence = _normalize_min_confidence(min_confidence)
    max_correction_px = _normalize_max_correction_px(max_correction_px)
    tiles_by_grid = {
        (tile.row, tile.col): tile
        for tile in project.tiles
        if tile.row is not None and tile.col is not None
    }
    neighbor_pairs = _neighbor_pairs_for_method(tiles_by_grid, method)

    constraints: list[NeighborConstraint] = []
    skip_reasons: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    accepted_count = 0
    pairs_total = len(neighbor_pairs)
    _emit_progress(
        progress_callback,
        f"Auto-registration: found {pairs_total} neighboring tile pair(s)",
        0,
        pairs_total,
    )
    for pair_index, (tile_a, tile_b, direction) in enumerate(neighbor_pairs, start=1):
        _emit_progress(
            progress_callback,
            (
                f"Auto-registration: pair {pair_index} / {pairs_total} "
                f"{tile_a.tile_id} -> {tile_b.tile_id} ({_direction_label(direction)})"
            ),
            pair_index,
            pairs_total,
        )
        constraint, reason = _build_constraint_for_pair(
            tile_a,
            tile_b,
            direction=direction,
            method=method,
            overlap_fraction=overlap_fraction,
            min_confidence=min_confidence,
            max_correction_px=max_correction_px,
        )
        if constraint is not None:
            constraints.append(constraint)
            if reason.startswith("fallback_"):
                fallback_reasons[reason] += 1
            else:
                accepted_count += 1
        else:
            skip_reasons[reason] += 1

    outlier_summary: dict[str, object] = {}
    if method == ROBUST_OUTLIER_ALIGNMENT_METHOD and constraints:
        _emit_progress(
            progress_callback,
            "Auto-registration: rejecting globally inconsistent overlap pairs",
            None,
            None,
        )
        constraints, outlier_summary = _reject_residual_outliers(project, constraints)

    project.metadata.extra_metadata["atlas_stitch_refinement_method"] = method
    project.metadata.extra_metadata["atlas_stitch_overlap_fraction"] = overlap_fraction
    project.metadata.extra_metadata["atlas_stitch_min_confidence"] = min_confidence
    project.metadata.extra_metadata["atlas_stitch_max_correction_px"] = max_correction_px if max_correction_px is not None else ""
    project.metadata.extra_metadata["atlas_stitch_neighbor_pairs_total"] = pairs_total
    project.metadata.extra_metadata["atlas_stitch_neighbor_pairs_accepted"] = accepted_count
    project.metadata.extra_metadata["atlas_stitch_neighbor_skip_reasons"] = dict(skip_reasons)
    project.metadata.extra_metadata["atlas_stitch_neighbor_fallback_reasons"] = dict(fallback_reasons)
    project.metadata.extra_metadata["atlas_stitch_outlier_rejection_enabled"] = method == ROBUST_OUTLIER_ALIGNMENT_METHOD
    project.metadata.extra_metadata["atlas_stitch_outlier_rejected_pair_count"] = int(outlier_summary.get("rejected_count", 0) or 0) if outlier_summary else 0
    project.metadata.extra_metadata["atlas_stitch_outlier_rejection_threshold_px"] = float(outlier_summary.get("threshold_px", 0.0) or 0.0) if outlier_summary else 0.0
    project.metadata.extra_metadata["atlas_stitch_outlier_rejection_iterations"] = int(outlier_summary.get("iterations", 0) or 0) if outlier_summary else 0
    project.metadata.extra_metadata["atlas_stitch_outlier_rejected_pairs"] = list(outlier_summary.get("rejected_pairs", []) or []) if outlier_summary else []
    _emit_progress(
        progress_callback,
        (
            "Auto-registration: pair registration finished "
            f"(accepted={accepted_count}, fallback={sum(fallback_reasons.values())}, skipped={sum(skip_reasons.values())})"
        ),
        pairs_total,
        pairs_total,
    )
    return constraints


def _emit_progress(
    progress_callback: Callable[[str, int | None, int | None], None] | None,
    message: str,
    current: int | None,
    total: int | None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(message, current, total)


def _direction_label(direction: str) -> str:
    if direction == "right_neighbor":
        return "right"
    if direction == "bottom_neighbor":
        return "bottom"
    return direction


def _neighbor_pairs_for_method(
    tiles_by_grid: dict[tuple[int | None, int | None], TileRecord],
    method: str,
) -> list[tuple[TileRecord, TileRecord, str]]:
    if method == CENTER_OUT_CLOCKWISE_METHOD:
        return _center_out_neighbor_pairs(tiles_by_grid, clockwise=True)
    if method == CENTER_OUT_COUNTERCLOCKWISE_METHOD:
        return _center_out_neighbor_pairs(tiles_by_grid, clockwise=False)
    return _all_neighbor_pairs(tiles_by_grid)


def _all_neighbor_pairs(
    tiles_by_grid: dict[tuple[int | None, int | None], TileRecord],
) -> list[tuple[TileRecord, TileRecord, str]]:
    neighbor_pairs: list[tuple[TileRecord, TileRecord, str]] = []
    for (row, col), tile_a in sorted(tiles_by_grid.items()):
        right_neighbor = tiles_by_grid.get((row, col + 1))
        if right_neighbor is not None:
            neighbor_pairs.append((tile_a, right_neighbor, "right_neighbor"))
        bottom_neighbor = tiles_by_grid.get((row + 1, col))
        if bottom_neighbor is not None:
            neighbor_pairs.append((tile_a, bottom_neighbor, "bottom_neighbor"))
    return neighbor_pairs


def _center_out_neighbor_pairs(
    tiles_by_grid: dict[tuple[int | None, int | None], TileRecord],
    *,
    clockwise: bool,
) -> list[tuple[TileRecord, TileRecord, str]]:
    if not tiles_by_grid:
        return []
    valid_positions = [(int(row), int(col)) for row, col in tiles_by_grid if row is not None and col is not None]
    if not valid_positions:
        return []
    center_row = (min(row for row, _col in valid_positions) + max(row for row, _col in valid_positions)) / 2.0
    center_col = (min(col for _row, col in valid_positions) + max(col for _row, col in valid_positions)) / 2.0
    ordered_starts = sorted(valid_positions, key=lambda rc: ((rc[0] - center_row) ** 2 + (rc[1] - center_col) ** 2, rc[0], rc[1]))
    visited: set[tuple[int, int]] = set()
    queued: set[tuple[int, int]] = set()
    pairs: list[tuple[TileRecord, TileRecord, str]] = []

    for start in ordered_starts:
        if start in visited or start in queued:
            continue
        queued.add(start)
        queue: list[tuple[int, int]] = [start]
        while queue:
            parent = queue.pop(0)
            visited.add(parent)
            for child in _ordered_neighbor_positions(parent, clockwise=clockwise):
                if child in visited or child in queued or child not in tiles_by_grid:
                    continue
                pair = _canonical_neighbor_pair(tiles_by_grid[parent], tiles_by_grid[child])
                if pair is None:
                    continue
                pairs.append(pair)
                queued.add(child)
                queue.append(child)
    return pairs


def _ordered_neighbor_positions(position: tuple[int, int], *, clockwise: bool) -> list[tuple[int, int]]:
    row, col = position
    offsets = ((0, 1), (1, 0), (0, -1), (-1, 0)) if clockwise else ((0, -1), (1, 0), (0, 1), (-1, 0))
    return [(row + d_row, col + d_col) for d_row, d_col in offsets]


def _canonical_neighbor_pair(tile_a: TileRecord, tile_b: TileRecord) -> tuple[TileRecord, TileRecord, str] | None:
    if tile_a.row is None or tile_a.col is None or tile_b.row is None or tile_b.col is None:
        return None
    row_a = int(tile_a.row)
    col_a = int(tile_a.col)
    row_b = int(tile_b.row)
    col_b = int(tile_b.col)
    if row_a == row_b and abs(col_a - col_b) == 1:
        return (tile_a, tile_b, "right_neighbor") if col_b > col_a else (tile_b, tile_a, "right_neighbor")
    if col_a == col_b and abs(row_a - row_b) == 1:
        return (tile_a, tile_b, "bottom_neighbor") if row_b > row_a else (tile_b, tile_a, "bottom_neighbor")
    return None


def _build_constraint_for_pair(
    tile_a: TileRecord,
    tile_b: TileRecord,
    *,
    direction: str,
    method: str,
    overlap_fraction: float,
    min_confidence: float,
    max_correction_px: float | None,
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
    if max_correction_px is not None and np.hypot(correction_dx, correction_dy) > max_correction_px:
        logger.info(
            "%s ↔ %s rejected as excessive correction | correction=(%.3f, %.3f) max=%.3f",
            tile_a.tile_id,
            tile_b.tile_id,
            correction_dx,
            correction_dy,
            max_correction_px,
        )
        return _fallback_nominal_constraint(tile_a, tile_b, direction=direction, reason="excessive_correction")
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
    if confidence < min_confidence:
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

    if method in {ROBUST_ALIGNMENT_METHOD, ROBUST_OUTLIER_ALIGNMENT_METHOD}:
        return _estimate_translation_phasecorr_robust(image_a, image_b, direction, overlap_fraction=overlap_fraction)
    if method == FEATURE_MATCHING_METHOD:
        return _estimate_translation_feature_matching(image_a, image_b, direction, overlap_fraction=overlap_fraction)
    if method == ECC_TRANSLATION_METHOD:
        return _estimate_translation_ecc(image_a, image_b, direction, overlap_fraction=overlap_fraction)
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


def _estimate_translation_feature_matching(
    image_a: np.ndarray,
    image_b: np.ndarray,
    direction: str,
    *,
    overlap_fraction: float,
) -> dict[str, object]:
    pair = _prepare_overlap_pair(image_a, image_b, direction, fraction=overlap_fraction)
    if pair["status"] != "ok":
        return pair
    strip_a = np.asarray(pair["strip_a"], dtype=np.float32)
    strip_b = np.asarray(pair["strip_b"], dtype=np.float32)
    dy_sign = float(pair["dy_sign"])
    try:
        import cv2
    except Exception:
        return {"status": "opencv_unavailable"}

    image_a_u8 = _normalize_for_cv_features(strip_a)
    image_b_u8 = _normalize_for_cv_features(strip_b)
    detector_name = "orb"
    try:
        if hasattr(cv2, "SIFT_create"):
            detector = cv2.SIFT_create(nfeatures=1000)
            detector_name = "sift"
            norm_type = cv2.NORM_L2
        else:
            detector = cv2.ORB_create(nfeatures=1500, fastThreshold=5)
            norm_type = cv2.NORM_HAMMING
        keypoints_a, descriptors_a = detector.detectAndCompute(image_a_u8, None)
        keypoints_b, descriptors_b = detector.detectAndCompute(image_b_u8, None)
    except Exception:
        return {"status": "feature_detection_error"}
    if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 4 or len(keypoints_b) < 4:
        return {"status": "not_enough_features"}

    try:
        matcher = cv2.BFMatcher(norm_type, crossCheck=True)
        matches = sorted(matcher.match(descriptors_a, descriptors_b), key=lambda match: match.distance)
    except Exception:
        return {"status": "feature_matching_error"}
    if len(matches) < 4:
        return {"status": "not_enough_matches"}

    # Use the best matches but avoid allowing many weak outliers to dominate.
    matches = matches[: min(len(matches), 80)]
    shifts_x: list[float] = []
    shifts_y: list[float] = []
    for match in matches:
        ref_x, ref_y = keypoints_a[match.queryIdx].pt
        mov_x, mov_y = keypoints_b[match.trainIdx].pt
        shifts_x.append(float(ref_x - mov_x))
        shifts_y.append(float(ref_y - mov_y))
    if not shifts_x or not shifts_y:
        return {"status": "not_enough_matches"}

    median_dx = float(np.median(np.asarray(shifts_x, dtype=float)))
    median_dy_raw = float(np.median(np.asarray(shifts_y, dtype=float)))
    residual = np.hypot(np.asarray(shifts_x) - median_dx, np.asarray(shifts_y) - median_dy_raw)
    inlier_threshold = max(3.0, min(strip_a.shape) * 0.03)
    inliers = residual <= inlier_threshold
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < 4:
        return {"status": "not_enough_inliers"}
    dx = float(np.median(np.asarray(shifts_x)[inliers]))
    dy_raw = float(np.median(np.asarray(shifts_y)[inliers]))
    confidence = float(np.clip((inlier_count / max(1, len(matches))) * min(1.0, inlier_count / 20.0), 0.0, 1.0))
    return {
        "status": "ok",
        "dx": dx,
        "dy": float(dy_sign * dy_raw),
        "confidence": confidence,
        "strip_shape": tuple(int(v) for v in strip_a.shape),
        "detector": detector_name,
        "match_count": len(matches),
        "inlier_count": inlier_count,
    }


def _estimate_translation_ecc(
    image_a: np.ndarray,
    image_b: np.ndarray,
    direction: str,
    *,
    overlap_fraction: float,
) -> dict[str, object]:
    pair = _prepare_overlap_pair(image_a, image_b, direction, fraction=overlap_fraction)
    if pair["status"] != "ok":
        return pair
    strip_a = np.asarray(pair["strip_a"], dtype=np.float32)
    strip_b = np.asarray(pair["strip_b"], dtype=np.float32)
    dy_sign = float(pair["dy_sign"])
    try:
        import cv2
    except Exception:
        return {"status": "opencv_unavailable"}
    try:
        reference = _normalize_for_cv_ecc(strip_a)
        moving = _normalize_for_cv_ecc(strip_b)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
        cc, warp = cv2.findTransformECC(reference, moving, warp, cv2.MOTION_TRANSLATION, criteria)
    except Exception:
        return {"status": "ecc_error"}
    dx = float(warp[0, 2])
    dy_raw = float(warp[1, 2])
    confidence = float(np.clip((float(cc) + 1.0) / 2.0, 0.0, 1.0))
    return {
        "status": "ok",
        "dx": dx,
        "dy": float(dy_sign * dy_raw),
        "confidence": confidence,
        "strip_shape": tuple(int(v) for v in strip_a.shape),
        "ecc_score": float(cc),
    }


def _normalize_for_cv_features(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    lower = float(np.percentile(finite, 1.0))
    upper = float(np.percentile(finite, 99.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
    if upper <= lower:
        return np.zeros(data.shape, dtype=np.uint8)
    scaled = (np.clip(data, lower, upper) - lower) / (upper - lower)
    return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)


def _normalize_for_cv_ecc(image: np.ndarray) -> np.ndarray:
    data = _normalize_for_cv_features(image).astype(np.float32) / 255.0
    return data


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


def _reject_residual_outliers(
    project: AtlasProject,
    constraints: list[NeighborConstraint],
    *,
    max_iterations: int = 5,
    threshold_px: float = 40.0,
) -> tuple[list[NeighborConstraint], dict[str, object]]:
    working = list(constraints)
    rejected_pairs: list[dict[str, object]] = []
    iterations = 0
    while len(working) > 1 and iterations < max_iterations:
        solved = solve_refined_tile_positions(project, working)
        tile_by_id = {tile.tile_id: tile for tile in solved.tiles}
        residuals: list[tuple[float, NeighborConstraint]] = []
        for constraint in working:
            tile_a = tile_by_id.get(constraint.tile_a_id)
            tile_b = tile_by_id.get(constraint.tile_b_id)
            if tile_a is None or tile_b is None:
                continue
            if tile_a.transform.refined_x is None or tile_a.transform.refined_y is None:
                continue
            if tile_b.transform.refined_x is None or tile_b.transform.refined_y is None:
                continue
            solved_dx = float(tile_b.transform.refined_x) - float(tile_a.transform.refined_x)
            solved_dy = float(tile_b.transform.refined_y) - float(tile_a.transform.refined_y)
            residual = float(np.hypot(solved_dx - float(constraint.dx), solved_dy - float(constraint.dy)))
            residuals.append((residual, constraint))
        if not residuals:
            break
        worst_residual, worst_constraint = max(residuals, key=lambda item: item[0])
        if worst_residual <= threshold_px:
            break
        working.remove(worst_constraint)
        rejected_pairs.append({
            "tile_a_id": worst_constraint.tile_a_id,
            "tile_b_id": worst_constraint.tile_b_id,
            "direction": worst_constraint.direction,
            "residual_px": worst_residual,
        })
        iterations += 1
    return working, {
        "rejected_count": len(rejected_pairs),
        "rejected_pairs": rejected_pairs,
        "iterations": iterations,
        "threshold_px": threshold_px,
    }


def _normalize_alignment_method(method: str) -> str:
    value = str(method or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"", "phasecorr", "default", "light", "light_phasecorr", "light_translation"}:
        return DEFAULT_ALIGNMENT_METHOD
    if value in {"robust", "robust_phasecorr", "robust_translation"}:
        return ROBUST_ALIGNMENT_METHOD
    if value in {"robust_outlier", "robust_translation_outlier_rejection", "robust_translation_plus_outlier_rejection"}:
        return ROBUST_OUTLIER_ALIGNMENT_METHOD
    if value in {"feature", "features", "feature_matching", "sift", "orb", "sift_orb"}:
        return FEATURE_MATCHING_METHOD
    if value in {"ecc", "ecc_translation", "enhanced_correlation", "enhanced_correlation_translation"}:
        return ECC_TRANSLATION_METHOD
    if value in {"center_out", "center_out_clockwise", "spiral_clockwise", "clockwise_spiral"}:
        return CENTER_OUT_CLOCKWISE_METHOD
    if value in {"center_out_counterclockwise", "center_out_anticlockwise", "spiral_counterclockwise", "spiral_anticlockwise", "counterclockwise_spiral", "anticlockwise_spiral"}:
        return CENTER_OUT_COUNTERCLOCKWISE_METHOD
    raise ValueError(f"Unsupported alignment method: {method}")


def _normalize_overlap_fraction(value: float) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return OVERLAP_FRACTION
    if not np.isfinite(fraction):
        return OVERLAP_FRACTION
    return float(np.clip(fraction, 0.01, 1.0))


def _normalize_min_confidence(value: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return MIN_CONFIDENCE
    if not np.isfinite(confidence):
        return MIN_CONFIDENCE
    return float(np.clip(confidence, 0.0, 1.0))


def _normalize_max_correction_px(value: float | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        correction = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CORRECTION_PX
    if not np.isfinite(correction) or correction <= 0:
        return None
    return float(correction)
