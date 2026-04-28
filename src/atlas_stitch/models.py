from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AtlasMetadata:
    atlas_name: str = ""
    xml_path: str = ""
    source_directory: str = ""
    tile_root_override: str = ""
    source_software: str = ""
    tile_count: int = 0
    image_width: int | None = None
    image_height: int | None = None
    image_depth: int | None = None
    channel_count: int | None = None
    voxel_size_x: float | None = None
    voxel_size_y: float | None = None
    voxel_size_z: float | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "atlas_name": self.atlas_name,
            "xml_path": self.xml_path,
            "source_directory": self.source_directory,
            "tile_root_override": self.tile_root_override,
            "source_software": self.source_software,
            "tile_count": self.tile_count,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_depth": self.image_depth,
            "channel_count": self.channel_count,
            "voxel_size_x": self.voxel_size_x,
            "voxel_size_y": self.voxel_size_y,
            "voxel_size_z": self.voxel_size_z,
            "extra_metadata": dict(self.extra_metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasMetadata":
        return cls(
            atlas_name=str(data.get("atlas_name") or "").strip(),
            xml_path=str(data.get("xml_path") or "").strip(),
            source_directory=str(data.get("source_directory") or "").strip(),
            tile_root_override=str(data.get("tile_root_override") or "").strip(),
            source_software=str(data.get("source_software") or "").strip(),
            tile_count=int(data.get("tile_count") or 0),
            image_width=_optional_int(data.get("image_width")),
            image_height=_optional_int(data.get("image_height")),
            image_depth=_optional_int(data.get("image_depth")),
            channel_count=_optional_int(data.get("channel_count")),
            voxel_size_x=_optional_float(data.get("voxel_size_x")),
            voxel_size_y=_optional_float(data.get("voxel_size_y")),
            voxel_size_z=_optional_float(data.get("voxel_size_z")),
            extra_metadata=dict(data.get("extra_metadata") or {}),
        )


@dataclass
class AtlasExportInfo:
    path: str = ""
    mode: str = ""
    time: str = ""
    chunk_size: int | None = None
    build_pyramid: bool | None = None
    tile_count: int | None = None
    status: str = ""
    atlas_project_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mode": self.mode,
            "time": self.time,
            "chunk_size": self.chunk_size,
            "build_pyramid": self.build_pyramid,
            "tile_count": self.tile_count,
            "status": self.status,
            "atlas_project_path": self.atlas_project_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasExportInfo":
        return cls(
            path=str(data.get("path") or "").strip(),
            mode=str(data.get("mode") or "").strip(),
            time=str(data.get("time") or "").strip(),
            chunk_size=_optional_int(data.get("chunk_size")),
            build_pyramid=_optional_bool(data.get("build_pyramid")),
            tile_count=_optional_int(data.get("tile_count")),
            status=str(data.get("status") or "").strip(),
            atlas_project_path=str(data.get("atlas_project_path") or "").strip(),
        )

@dataclass
class TileTransform:
    nominal_x: float = 0.0
    nominal_y: float = 0.0
    nominal_z: float = 0.0
    refined_x: float | None = None
    refined_y: float | None = None
    refined_z: float | None = None
    manual_x: float | None = None
    manual_y: float | None = None
    manual_z: float | None = None
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    rotation_degrees: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "nominal_x": self.nominal_x,
            "nominal_y": self.nominal_y,
            "nominal_z": self.nominal_z,
            "refined_x": self.refined_x,
            "refined_y": self.refined_y,
            "refined_z": self.refined_z,
            "manual_x": self.manual_x,
            "manual_y": self.manual_y,
            "manual_z": self.manual_z,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "scale_z": self.scale_z,
            "rotation_degrees": self.rotation_degrees,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TileTransform":
        return cls(
            nominal_x=float(data.get("nominal_x") or 0.0),
            nominal_y=float(data.get("nominal_y") or 0.0),
            nominal_z=float(data.get("nominal_z") or 0.0),
            refined_x=_optional_float(data.get("refined_x")),
            refined_y=_optional_float(data.get("refined_y")),
            refined_z=_optional_float(data.get("refined_z")),
            manual_x=_optional_float(data.get("manual_x")),
            manual_y=_optional_float(data.get("manual_y")),
            manual_z=_optional_float(data.get("manual_z")),
            scale_x=float(data.get("scale_x") or 1.0),
            scale_y=float(data.get("scale_y") or 1.0),
            scale_z=float(data.get("scale_z") or 1.0),
            rotation_degrees=float(data.get("rotation_degrees") or 0.0),
        )


@dataclass
class TileRecord:
    tile_id: str
    file_name: str = ""
    source_path: str = ""
    resolved_path: str = ""
    repaired_path: str = ""
    repair_confidence_path: str = ""
    repair_attribution_path: str = ""
    row: int | None = None
    col: int | None = None
    start_x: float | None = None
    start_y: float | None = None
    position_x: float | None = None
    position_y: float | None = None
    width: int | None = None
    height: int | None = None
    exists: bool = False
    transform: TileTransform = field(default_factory=TileTransform)
    metadata: dict[str, Any] = field(default_factory=dict)
    repair_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "file_name": self.file_name,
            "source_path": self.source_path,
            "resolved_path": self.resolved_path,
            "repaired_path": self.repaired_path,
            "repair_confidence_path": self.repair_confidence_path,
            "repair_attribution_path": self.repair_attribution_path,
            "row": self.row,
            "col": self.col,
            "start_x": self.start_x,
            "start_y": self.start_y,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "width": self.width,
            "height": self.height,
            "exists": self.exists,
            "transform": self.transform.to_dict(),
            "metadata": dict(self.metadata),
            "repair_history": [dict(entry) for entry in self.repair_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TileRecord":
        return cls(
            tile_id=str(data.get("tile_id") or "").strip(),
            file_name=str(data.get("file_name") or "").strip(),
            source_path=str(data.get("source_path") or "").strip(),
            resolved_path=str(data.get("resolved_path") or "").strip(),
            repaired_path=str(data.get("repaired_path") or "").strip(),
            repair_confidence_path=str(data.get("repair_confidence_path") or "").strip(),
            repair_attribution_path=str(data.get("repair_attribution_path") or "").strip(),
            row=_optional_int(data.get("row")),
            col=_optional_int(data.get("col")),
            start_x=_optional_float(data.get("start_x")),
            start_y=_optional_float(data.get("start_y")),
            position_x=_optional_float(data.get("position_x")),
            position_y=_optional_float(data.get("position_y")),
            width=_optional_int(data.get("width")),
            height=_optional_int(data.get("height")),
            exists=bool(data.get("exists")),
            transform=TileTransform.from_dict(dict(data.get("transform") or {})),
            metadata=dict(data.get("metadata") or {}),
            repair_history=[dict(entry or {}) for entry in data.get("repair_history", [])],
        )


@dataclass
class AtlasProject:
    metadata: AtlasMetadata
    tiles: list[TileRecord] = field(default_factory=list)
    missing_tiles: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    last_export: AtlasExportInfo = field(default_factory=AtlasExportInfo)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "tiles": [tile.to_dict() for tile in self.tiles],
            "missing_tiles": list(self.missing_tiles),
            "warnings": list(self.warnings),
            "last_export": self.last_export.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasProject":
        return cls(
            metadata=AtlasMetadata.from_dict(dict(data.get("metadata") or {})),
            tiles=[TileRecord.from_dict(dict(item or {})) for item in data.get("tiles", [])],
            missing_tiles=[str(value).strip() for value in data.get("missing_tiles", []) if str(value).strip()],
            warnings=[str(value).strip() for value in data.get("warnings", []) if str(value).strip()],
            last_export=AtlasExportInfo.from_dict(dict(data.get("last_export") or {})),
        )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None
