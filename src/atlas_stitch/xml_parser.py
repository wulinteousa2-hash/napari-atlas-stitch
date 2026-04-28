from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any
import xml.etree.ElementTree as ET

from .models import AtlasMetadata, AtlasProject, TileRecord, TileTransform


_PATH_KEYS = (
    "path",
    "filepath",
    "file_path",
    "filename",
    "file",
    "url",
    "tilepath",
    "tile_path",
    "image",
    "imagefile",
    "image_path",
    "imagepath",
    "src",
    "source",
)


def parse_atlas_source(source_path: str, tile_root_override: str | None = None) -> AtlasProject:
    source_file = Path(source_path).expanduser().resolve()
    suffix = source_file.suffix.lower()
    if suffix == ".ve-mif":
        return parse_atlas_vemif(str(source_file), tile_root_override=tile_root_override)
    return parse_atlas_xml(str(source_file), tile_root_override=tile_root_override)


def parse_atlas_xml(xml_path: str, tile_root_override: str | None = None) -> AtlasProject:
    xml_file = Path(xml_path).expanduser().resolve()
    tree = ET.parse(xml_file)
    root = tree.getroot()
    override = str(tile_root_override or "").strip()

    metadata_values = _collect_metadata(root)
    pixel_size_value, pixel_size_unit = _pixel_size_fields(root)
    voxel_size_x = _pick_first_float(metadata_values, ("voxel_size_x", "pixel_size_x", "pixelsizex", "spacingx"))
    voxel_size_y = _pick_first_float(metadata_values, ("voxel_size_y", "pixel_size_y", "pixelsizey", "spacingy"))
    voxel_size_z = _pick_first_float(metadata_values, ("voxel_size_z", "pixel_size_z", "pixelsizez", "spacingz"))
    if pixel_size_value is not None:
        if voxel_size_x is None:
            voxel_size_x = pixel_size_value
        if voxel_size_y is None:
            voxel_size_y = pixel_size_value
    if pixel_size_unit and "pixel_size_unit" not in metadata_values:
        metadata_values["pixel_size_unit"] = pixel_size_unit
    atlas_name = _pick_first_text(metadata_values, ("atlas_name", "name", "dataset_name", "title")) or xml_file.stem
    metadata = AtlasMetadata(
        atlas_name=atlas_name,
        xml_path=str(xml_file),
        source_directory=str(xml_file.parent),
        tile_root_override=override,
        source_software=_pick_first_text(metadata_values, ("software", "application", "generator", "vendor")),
        image_width=_pick_first_int(metadata_values, ("image_width", "width", "sizex")),
        image_height=_pick_first_int(metadata_values, ("image_height", "height", "sizey")),
        image_depth=_pick_first_int(metadata_values, ("image_depth", "depth", "sizez")),
        channel_count=_pick_first_int(metadata_values, ("channel_count", "channels", "sizec")),
        voxel_size_x=voxel_size_x,
        voxel_size_y=voxel_size_y,
        voxel_size_z=voxel_size_z,
        extra_metadata=metadata_values,
    )

    tiles: list[TileRecord] = []
    seen_signatures: set[tuple[int | None, int | None, str, float | None, float | None]] = set()
    warnings: list[str] = []
    duplicate_count = 0
    ignored_element_count = 0
    for index, element in enumerate(root.iter()):
        raw_values = _element_value_map(element)
        row = _pick_first_int(raw_values, ("row", "row_index", "r"))
        col = _pick_first_int(raw_values, ("column", "col", "column_index"))
        source_path = _pick_first_text(raw_values, _PATH_KEYS)
        if row is None or col is None or not source_path:
            ignored_element_count += 1
            continue
        tile = _parse_tile_element(
            element,
            index=index,
            xml_dir=xml_file.parent,
            tile_root_override=override,
            raw_values=raw_values,
        )
        if tile is None:
            continue
        signature = (
            tile.row,
            tile.col,
            tile.file_name,
            tile.start_x,
            tile.start_y,
        )
        if signature in seen_signatures:
            duplicate_count += 1
            continue
        seen_signatures.add(signature)
        tiles.append(tile)

    metadata.tile_count = len(tiles)
    metadata.extra_metadata["ignored_non_tile_elements"] = ignored_element_count
    metadata.extra_metadata["duplicate_tile_elements"] = duplicate_count
    if duplicate_count:
        warnings.append(
            f"Ignored {duplicate_count} duplicate tile XML element(s) while collecting atlas tiles."
        )
    if ignored_element_count:
        warnings.append(
            f"Ignored {ignored_element_count} non-tile XML element(s) during parsing."
        )
    missing_tiles = [tile.file_name or tile.tile_id for tile in tiles if not tile.exists]
    return AtlasProject(metadata=metadata, tiles=tiles, missing_tiles=missing_tiles, warnings=warnings)


def parse_atlas_vemif(mif_path: str, tile_root_override: str | None = None) -> AtlasProject:
    mif_file = Path(mif_path).expanduser().resolve()
    tree = ET.parse(mif_file)
    root = tree.getroot()
    override = str(tile_root_override or "").strip()

    metadata_values = _collect_metadata(root)
    pixel_size_value, pixel_size_unit = _pixel_size_fields(root)
    atlas_name = _pick_first_text(metadata_values, ("name",)) or mif_file.stem
    tile_width = _pick_first_int(metadata_values, ("tilewidth",))
    tile_height = _pick_first_int(metadata_values, ("tileheight",))
    num_tiles_x = _pick_first_int(metadata_values, ("numtilesx",))
    num_tiles_y = _pick_first_int(metadata_values, ("numtilesy",))
    fov_um = _pick_first_float(metadata_values, ("fov",))
    overlap_x_um = _pick_first_float(metadata_values, ("tileoverlapxum",)) or 0.0
    overlap_y_um = _pick_first_float(metadata_values, ("tileoverlapyum",)) or 0.0
    pixel_size_um = _pixel_size_um(pixel_size_value, pixel_size_unit, fov_um, tile_width)
    step_x_px = _step_pixels(fov_um, overlap_x_um, tile_width, pixel_size_um)
    step_y_px = _step_pixels(fov_um, overlap_y_um, tile_height, pixel_size_um)

    updates_path = mif_file.with_suffix(".ve-updates")
    update_positions_px = _load_update_positions_px(updates_path, pixel_size_um)

    tiles: list[TileRecord] = []
    nominal_tiles: list[tuple[TileRecord, float | None, float | None]] = []
    for index, element in enumerate(root.findall("./Tiles/Tile")):
        raw_values = _element_value_map(element)
        row = _pick_first_int(raw_values, ("row",))
        col = _pick_first_int(raw_values, ("col", "column"))
        source_path = _pick_first_text(raw_values, ("filename",))
        if row is None or col is None or not source_path:
            continue
        tile_name = Path(source_path.replace("\\", "/")).stem
        resolved_path = _resolve_tile_path(source_path, xml_dir=mif_file.parent, tile_root_override=override)
        start_x = float(max(0, col - 1) * step_x_px) if step_x_px is not None else None
        start_y = float(max(0, row - 1) * step_y_px) if step_y_px is not None else None
        tile = TileRecord(
            tile_id=tile_name or f"tile_{index:04d}",
            file_name=Path(source_path).name,
            source_path=source_path,
            resolved_path=str(resolved_path) if resolved_path is not None else "",
            row=row,
            col=col,
            start_x=start_x,
            start_y=start_y,
            position_x=_pick_first_float(raw_values, ("stagex", "targetstagex")),
            position_y=_pick_first_float(raw_values, ("stagey", "targetstagey")),
            width=tile_width,
            height=tile_height,
            exists=resolved_path is not None and resolved_path.exists(),
            transform=TileTransform(
                nominal_x=float(start_x or 0.0),
                nominal_y=float(start_y or 0.0),
                nominal_z=_pick_first_float(raw_values, ("stagez", "targetstagez")) or 0.0,
            ),
            metadata={
                **raw_values,
                "file_name": Path(source_path).name,
                "source_path": source_path,
            },
        )
        nominal_tiles.append((tile, start_x, start_y))

    normalized_tiles = _normalize_tile_origins(nominal_tiles)
    _apply_refined_positions(normalized_tiles, update_positions_px)
    tiles.extend(normalized_tiles)

    width_px = _canvas_extent(tiles, "x")
    height_px = _canvas_extent(tiles, "y")
    metadata_values["source_kind"] = "ve_mif"
    metadata_values["source_path"] = str(mif_file)
    metadata_values["pixel_size_unit"] = pixel_size_unit or metadata_values.get("pixel_size_unit") or "um"
    metadata_values["atlas_stitch_update_file"] = str(updates_path) if updates_path.exists() else ""
    metadata_values["atlas_stitch_updates_applied"] = bool(update_positions_px)
    metadata_values["atlas_stitch_nominal_step_x_px"] = step_x_px
    metadata_values["atlas_stitch_nominal_step_y_px"] = step_y_px
    metadata = AtlasMetadata(
        atlas_name=atlas_name,
        xml_path=str(mif_file),
        source_directory=str(mif_file.parent),
        tile_root_override=override,
        source_software=_pick_first_text(metadata_values, ("application", "software", "vendor")),
        tile_count=len(tiles),
        image_width=width_px,
        image_height=height_px,
        channel_count=1,
        voxel_size_x=pixel_size_value,
        voxel_size_y=pixel_size_value,
        voxel_size_z=None,
        extra_metadata=metadata_values,
    )
    warnings: list[str] = []
    if updates_path.exists() and not update_positions_px:
        warnings.append("VE updates file was found but did not provide usable tile transforms.")
    missing_tiles = [tile.file_name or tile.tile_id for tile in tiles if not tile.exists]
    return AtlasProject(metadata=metadata, tiles=tiles, missing_tiles=missing_tiles, warnings=warnings)


def _collect_metadata(root: ET.Element) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for element in root.iter():
        tag = _clean_key(element.tag)
        if tag and element.text and element.text.strip() and tag not in values:
            values[tag] = element.text.strip()
        for key, value in element.attrib.items():
            clean_key = _clean_key(key)
            if clean_key and value.strip() and clean_key not in values:
                values[clean_key] = value.strip()
    return values


def _pixel_size_fields(root: ET.Element) -> tuple[float | None, str]:
    for element in root.iter():
        if _clean_key(element.tag) != "pixelsize":
            continue
        nested = _element_value_map(element)
        value = _pick_first_float(nested, ("value", "pixel_size", "pixelsize"))
        unit = _pick_first_text(nested, ("unit", "pixel_size_unit", "pixelsizeunit"))
        if value is not None or unit:
            return value, unit
    return None, ""


def _parse_tile_element(
    element: ET.Element,
    *,
    index: int,
    xml_dir: Path,
    tile_root_override: str,
    raw_values: dict[str, str] | None = None,
    ) -> TileRecord | None:
    raw_values = raw_values or _element_value_map(element)
    source_path = _pick_first_text(raw_values, _PATH_KEYS)
    if not source_path:
        return None

    resolved_path = _resolve_tile_path(source_path, xml_dir=xml_dir, tile_root_override=tile_root_override)
    tile_id = (
        _pick_first_text(raw_values, ("tile_id", "tileid", "id", "name", "label"))
        or f"tile_{index:04d}"
    )
    file_name = Path(source_path).name
    raw_values.setdefault("file_name", file_name)
    raw_values.setdefault("source_path", source_path)
    row = _pick_first_int(raw_values, ("row", "row_index", "r"))
    col = _pick_first_int(raw_values, ("column", "col", "column_index"))
    start_x = _pick_first_float(raw_values, ("start_x", "startx", "left", "originx"))
    start_y = _pick_first_float(raw_values, ("start_y", "starty", "top", "originy"))
    position_x = _pick_first_float(raw_values, ("position_x", "posx", "tile_position_x", "stagex"))
    position_y = _pick_first_float(raw_values, ("position_y", "posy", "tile_position_y", "stagey"))
    position_z = _pick_first_float(raw_values, ("position_z", "posz", "tile_position_z", "stagez"))
    transform = TileTransform(
        nominal_x=start_x if start_x is not None else position_x or 0.0,
        nominal_y=start_y if start_y is not None else position_y or 0.0,
        nominal_z=position_z or 0.0,
        scale_x=_pick_first_float(raw_values, ("scale_x", "scalex")) or 1.0,
        scale_y=_pick_first_float(raw_values, ("scale_y", "scaley")) or 1.0,
        scale_z=_pick_first_float(raw_values, ("scale_z", "scalez")) or 1.0,
        rotation_degrees=_pick_first_float(raw_values, ("rotation", "angle", "rotation_degrees")) or 0.0,
    )
    return TileRecord(
        tile_id=tile_id,
        file_name=file_name,
        source_path=source_path,
        resolved_path=str(resolved_path) if resolved_path is not None else "",
        row=row,
        col=col,
        start_x=start_x,
        start_y=start_y,
        position_x=position_x,
        position_y=position_y,
        width=_pick_first_int(raw_values, ("width", "sizex")),
        height=_pick_first_int(raw_values, ("height", "sizey")),
        exists=resolved_path is not None and resolved_path.exists(),
        transform=transform,
        metadata=raw_values,
    )


def _element_value_map(element: ET.Element) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in element.attrib.items():
        if value.strip():
            values[_clean_key(key)] = value.strip()
    for child in element:
        if child.text and child.text.strip():
            values.setdefault(_clean_key(child.tag), child.text.strip())
        for key, value in child.attrib.items():
            if value.strip():
                values.setdefault(_clean_key(key), value.strip())
    if element.text and element.text.strip():
        values.setdefault(_clean_key(element.tag), element.text.strip())
    return values


def _resolve_tile_path(source_path: str, *, xml_dir: Path, tile_root_override: str) -> Path | None:
    if not source_path.strip():
        return None
    text = source_path.strip().strip('"').strip("'")
    posix_text = text.replace("\\", "/")

    candidates: list[Path] = []
    if tile_root_override.strip():
        override_root = Path(tile_root_override).expanduser()
        override_candidates = _override_relative_parts(text)
        for part in override_candidates:
            candidates.append(override_root / part)
        candidates.append(override_root / Path(posix_text).name)

    source_path_obj = Path(posix_text).expanduser()
    candidates.append(source_path_obj)
    candidates.append(xml_dir / source_path_obj)
    candidates.append(xml_dir / Path(posix_text).name)

    for candidate in candidates:
        try:
            normalized = candidate.resolve(strict=False)
        except Exception:
            normalized = candidate
        if normalized.exists():
            return normalized

    return candidates[0].resolve(strict=False) if candidates else None


def _override_relative_parts(path_text: str) -> list[Path]:
    win_path = PureWindowsPath(path_text)
    parts = [part for part in win_path.parts if part not in (win_path.anchor, "\\", "/")]
    filtered_parts = [part for part in parts if ":" not in part]
    if not filtered_parts:
        posix_name = Path(path_text.replace("\\", "/")).name
        return [Path(posix_name)] if posix_name else []
    candidates = [Path(*filtered_parts)]
    if len(filtered_parts) > 1:
        candidates.append(Path(*filtered_parts[-2:]))
    candidates.append(Path(filtered_parts[-1]))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _load_update_positions_px(updates_path: Path, pixel_size_um: float | None) -> dict[str, tuple[float | None, float | None]]:
    if not updates_path.exists():
        return {}
    try:
        tree = ET.parse(updates_path)
    except Exception:
        return {}
    positions: dict[str, tuple[float | None, float | None]] = {}
    root = tree.getroot()
    for element in root.findall("./Tile"):
        name = (element.findtext("Name") or "").strip()
        parent = element.find("./ParentTransform")
        if not name or parent is None:
            continue
        x_um = _safe_float(parent.findtext("M41"))
        y_um = _safe_float(parent.findtext("M42"))
        positions[name] = (
            _physical_um_to_px(x_um, pixel_size_um),
            _physical_um_to_px(y_um, pixel_size_um),
        )
    return positions


def _normalize_tile_origins(raw_tiles: list[tuple[TileRecord, float | None, float | None]]) -> list[TileRecord]:
    xs = [float(start_x) for _, start_x, _ in raw_tiles if start_x is not None]
    ys = [float(start_y) for _, _, start_y in raw_tiles if start_y is not None]
    min_x = min(xs) if xs else 0.0
    min_y = min(ys) if ys else 0.0
    tiles: list[TileRecord] = []
    for tile, start_x, start_y in raw_tiles:
        if start_x is not None:
            tile.start_x = float(start_x) - min_x
            tile.transform.nominal_x = tile.start_x
        if start_y is not None:
            tile.start_y = float(start_y) - min_y
            tile.transform.nominal_y = tile.start_y
        tiles.append(tile)
    return tiles


def _apply_refined_positions(
    tiles: list[TileRecord],
    update_positions_px: dict[str, tuple[float | None, float | None]],
) -> None:
    refined_entries: list[tuple[TileRecord, float, float]] = []
    xs: list[float] = []
    ys: list[float] = []
    for tile in tiles:
        refined_x, refined_y = update_positions_px.get(tile.tile_id, (None, None))
        if refined_x is None or refined_y is None:
            continue
        refined_x = float(refined_x)
        refined_y = float(refined_y)
        xs.append(refined_x)
        ys.append(refined_y)
        refined_entries.append((tile, refined_x, refined_y))
    if not refined_entries:
        return
    min_x = min(xs)
    min_y = min(ys)
    for tile, refined_x, refined_y in refined_entries:
        tile.transform.refined_x = refined_x - min_x
        tile.transform.refined_y = refined_y - min_y


def _canvas_extent(tiles: list[TileRecord], axis: str) -> int | None:
    starts: list[float] = []
    ends: list[float] = []
    for tile in tiles:
        start = tile.start_x if axis == "x" else tile.start_y
        size = tile.width if axis == "x" else tile.height
        if start is None or size is None:
            continue
        starts.append(float(start))
        ends.append(float(start) + float(size))
    if not starts or not ends:
        return None
    return int(round(max(ends) - min(starts)))


def _pixel_size_um(pixel_size_value: float | None, pixel_size_unit: str, fov_um: float | None, tile_width: int | None) -> float | None:
    unit_scale = _unit_to_um_scale(pixel_size_unit)
    if pixel_size_value is not None and unit_scale is not None:
        return float(pixel_size_value) * unit_scale
    if fov_um is not None and tile_width:
        return float(fov_um) / float(tile_width)
    return None


def _step_pixels(fov_um: float | None, overlap_um: float | None, tile_extent_px: int | None, pixel_size_um: float | None) -> float | None:
    if fov_um is not None and pixel_size_um:
        return (float(fov_um) - float(overlap_um or 0.0)) / float(pixel_size_um)
    if tile_extent_px is None:
        return None
    return float(tile_extent_px)


def _physical_um_to_px(value_um: float | None, pixel_size_um: float | None) -> float | None:
    if value_um is None:
        return None
    if pixel_size_um is None or abs(pixel_size_um) < 1e-12:
        return float(value_um)
    return float(value_um) / float(pixel_size_um)


def _unit_to_um_scale(unit: str) -> float | None:
    normalized = _normalize_unit(unit)
    if normalized == "um":
        return 1.0
    if normalized == "nm":
        return 0.001
    if normalized == "mm":
        return 1000.0
    return None


def _normalize_unit(unit: str) -> str:
    text = str(unit or "").strip().lower().replace("µ", "u")
    if text in {"um", "micrometer", "micrometers", "micron", "microns"}:
        return "um"
    if text in {"nm", "nanometer", "nanometers"}:
        return "nm"
    if text in {"mm", "millimeter", "millimeters"}:
        return "mm"
    return text


def _safe_float(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _clean_key(value: str) -> str:
    text = value.rsplit("}", 1)[-1]
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _pick_first_text(values: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = values.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _pick_first_int(values: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = values.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            continue
    return None


def _pick_first_float(values: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = values.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            continue
    return None
