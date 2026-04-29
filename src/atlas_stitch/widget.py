from __future__ import annotations

from pathlib import Path
from datetime import datetime
import math
import time
from typing import Any

import numpy as np
from napari import Viewer, current_viewer
from qtpy.QtCore import QObject, QThread, Qt, Signal, QSettings
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from skimage.transform import resize
from tifffile import imread

from .models import AtlasExportInfo, AtlasProject, TileRecord, TileTransform
from .project_state import load_atlas_project, save_atlas_project
from .refinement_overlap import (
    DEFAULT_ALIGNMENT_METHOD,
    ROBUST_ALIGNMENT_METHOD,
    ROBUST_OUTLIER_ALIGNMENT_METHOD,
    FEATURE_MATCHING_METHOD,
    ECC_TRANSLATION_METHOD,
    CENTER_OUT_CLOCKWISE_METHOD,
    CENTER_OUT_COUNTERCLOCKWISE_METHOD,
    build_neighbor_constraints,
)
from .refinement_solver import solve_refined_tile_positions
from .seam_repair import (
    BLEND_MODE_FEATHER,
    BLEND_MODE_HARD_REPLACE,
    DEFAULT_REPAIR_OVERLAP,
    REPAIR_MODE_FULL_OVERLAP,
    REPAIR_MODE_ROI_GUIDED,
    DONOR_DIRECTIONS,
    RepairDonorSpec,
    TileRepairRequest,
    TileRepairResult,
    load_preferred_tile_pixels,
    preferred_tile_path,
    reconstruct_tile_from_donors,
    save_repair_outputs,
)
from .xml_parser import parse_atlas_source

ATLAS_STITCH_SETTINGS_ORG = "napari-atlas-stitch"
ATLAS_STITCH_SETTINGS_APP = "atlas-stitch"
FUSION_OVERWRITE = "overwrite"
FUSION_LINEAR_BLEND = "linear_blend"
FUSION_AVERAGE = "average"
FUSION_MAX = "max_intensity"
FUSION_MIN = "min_intensity"

class PreviewWorker(QObject):
    progress = Signal(str, int, int)
    preview_ready = Signal(object, dict)
    error = Signal(str)
    finished = Signal()

    def __init__(self, tiles: list[TileRecord], downsample: int, placement_mode: str):
        super().__init__()
        self.tiles = tiles
        self.downsample = max(1, downsample)
        self.placement_mode = placement_mode

    def run(self) -> None:
        try:
            valid = [
                tile
                for tile in self.tiles
                if preferred_tile_path(tile)
                and Path(preferred_tile_path(tile)).expanduser().exists()
                and _preview_position(tile, self.placement_mode) is not None
            ]
            total = len(valid)
            if not valid:
                self.error.emit(f"No tiles with complete {self.placement_mode} placement were found for preview.")
                return

            self.progress.emit("Preview: preparing tile bounds", -1, -1)
            min_x = min(_preview_position(tile, self.placement_mode)[0] for tile in valid)
            min_y = min(_preview_position(tile, self.placement_mode)[1] for tile in valid)
            max_x = 0
            max_y = 0
            for tile in valid:
                width, height = self._infer_tile_shape(tile)
                tile_x, tile_y = _preview_position(tile, self.placement_mode) or (0.0, 0.0)
                max_x = max(max_x, tile_x + width)
                max_y = max(max_y, tile_y + height)

            final_width = math.ceil((max_x - min_x) / self.downsample)
            final_height = math.ceil((max_y - min_y) / self.downsample)
            preview_image = np.zeros((final_height, final_width), dtype=np.float32)

            for index, tile in enumerate(valid, start=1):
                self.progress.emit(f"Preview: reading tile {index} / {total}", index, total)
                path = Path(preferred_tile_path(tile))
                if not path.exists():
                    continue
                try:
                    data = load_preferred_tile_pixels(tile)
                except Exception:
                    continue
                tile_x, tile_y = _preview_position(tile, self.placement_mode) or (0.0, 0.0)
                y0, y1, x0, x1 = self._preview_bounds(
                    tile_x=tile_x,
                    tile_y=tile_y,
                    tile_width=data.shape[1],
                    tile_height=data.shape[0],
                    min_x=min_x,
                    min_y=min_y,
                    final_width=final_width,
                    final_height=final_height,
                )
                if y1 <= y0 or x1 <= x0:
                    continue
                data = self._downsample_to_shape(data, y1 - y0, x1 - x0)
                preview_image[y0:y1, x0:x1] = np.maximum(
                    preview_image[y0:y1, x0:x1],
                    data[: y1 - y0, : x1 - x0],
                )

            self.progress.emit("Preview: assembling coarse atlas", -1, -1)
            meta = {"translate": (min_y / self.downsample, min_x / self.downsample)}
            self.preview_ready.emit(preview_image, meta)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    @staticmethod
    def _downsample(data: Any, factor: int):
        if factor <= 1 or data.ndim < 2:
            return data
        slices = [slice(None, None, factor), slice(None, None, factor)]
        slices.extend([slice(None)] * (data.ndim - 2))
        return data[tuple(slices)]

    def _downsample_to_shape(self, data: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        if target_height <= 0 or target_width <= 0:
            return np.zeros((0, 0), dtype=np.float32)
        stride_view = self._downsample(data, self.downsample)
        if stride_view.shape[:2] == (target_height, target_width):
            return stride_view.astype(np.float32, copy=False)
        resized = resize(
            data,
            (target_height, target_width),
            order=1,
            preserve_range=True,
            anti_aliasing=False,
        )
        return resized.astype(np.float32, copy=False)

    def _preview_bounds(
        self,
        *,
        tile_x: float,
        tile_y: float,
        tile_width: int,
        tile_height: int,
        min_x: float,
        min_y: float,
        final_width: int,
        final_height: int,
    ) -> tuple[int, int, int, int]:
        x0 = int(math.floor((tile_x - min_x) / self.downsample))
        y0 = int(math.floor((tile_y - min_y) / self.downsample))
        x1 = int(math.ceil((tile_x - min_x + tile_width) / self.downsample))
        y1 = int(math.ceil((tile_y - min_y + tile_height) / self.downsample))
        x0 = max(0, min(x0, final_width))
        x1 = max(0, min(x1, final_width))
        y0 = max(0, min(y0, final_height))
        y1 = max(0, min(y1, final_height))
        return y0, y1, x0, x1

    def _infer_tile_shape(self, tile: TileRecord) -> tuple[int, int]:
        if tile.width is not None and tile.height is not None:
            return int(tile.width), int(tile.height)
        try:
            data = load_preferred_tile_pixels(tile)
        except Exception:
            return 1, 1
        height, width = data.shape[:2]
        return width, height



class ExportWorker(QObject):
    progress = Signal(str, int, int)
    completed = Signal(str, int, bool, str, str, str)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        project: AtlasProject,
        output_path: str,
        chunk_size: int,
        build_pyramid: bool,
        fusion_method: str,
        atlas_project_path: str = "",
        placement_mode: str = "nominal",
        tile_ids: list[str] | None = None,
    ):
        super().__init__()
        self.project = project
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.build_pyramid = build_pyramid
        self.fusion_method = fusion_method
        self.atlas_project_path = atlas_project_path
        self.placement_mode = placement_mode
        self.tile_ids = list(tile_ids or [])

    def run(self) -> None:
        try:
            from .ome_zarr_export import export_nominal_layout_to_omezarr

            exported = export_nominal_layout_to_omezarr(
                self.project,
                self.output_path,
                chunk_size=self.chunk_size,
                build_pyramid=self.build_pyramid,
                fusion_method=self.fusion_method,
                atlas_project_path=self.atlas_project_path or None,
                progress_callback=self._handle_export_progress,
                placement_mode=self.placement_mode,
                tile_ids=self.tile_ids,
            )
        except Exception as exc:
            self.error.emit(str(exc))
        else:
            self.completed.emit(
                str(exported),
                self.chunk_size,
                self.build_pyramid,
                self.atlas_project_path,
                self.fusion_method,
                self.placement_mode,
            )
        finally:
            self.finished.emit()

    def _handle_export_progress(self, stage: str, current: int | None, total: int | None) -> None:
        self.progress.emit(stage, -1 if current is None else int(current), -1 if total is None else int(total))

class AlignmentWorker(QObject):
    progress = Signal(str, int, int)
    completed = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, project: AtlasProject, method: str, min_confidence: float, max_correction_px: float | None):
        super().__init__()
        self.project = project
        self.method = method
        self.min_confidence = min_confidence
        self.max_correction_px = max_correction_px

    def run(self) -> None:
        started = time.monotonic()
        try:
            method_label = _alignment_method_label(self.method)
            self.progress.emit(f"Alignment ({method_label}): building neighbor constraints", -1, -1)
            overlap_fraction = _project_overlap_fraction(self.project)
            constraints = build_neighbor_constraints(
                self.project,
                method=self.method,
                overlap_fraction=overlap_fraction,
                min_confidence=self.min_confidence,
                max_correction_px=self.max_correction_px,
                progress_callback=self._handle_pair_progress,
            )
            self.progress.emit(f"Alignment ({method_label}): solving refined tile positions", -1, -1)
            solved_project = solve_refined_tile_positions(self.project, constraints)
            solved_project.metadata.extra_metadata["atlas_stitch_registration_elapsed_seconds"] = (
                time.monotonic() - started
            )
        except Exception as exc:
            self.error.emit(str(exc))
        else:
            self.completed.emit(solved_project)
        finally:
            self.finished.emit()

    def _handle_pair_progress(self, message: str, current: int | None, total: int | None) -> None:
        self.progress.emit(message, -1 if current is None else int(current), -1 if total is None else int(total))


class RepairWorker(QObject):
    progress = Signal(str, int, int)
    preview_ready = Signal(object)
    applied = Signal(object, dict)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        project: AtlasProject,
        request: TileRepairRequest,
        *,
        output_dir: str = "",
        apply_changes: bool = False,
        repair_mode: str = REPAIR_MODE_FULL_OVERLAP,
        blend_mode: str = BLEND_MODE_HARD_REPLACE,
        overlap_width: int = DEFAULT_REPAIR_OVERLAP,
    ):
        super().__init__()
        self.project = project
        self.request = request
        self.output_dir = output_dir
        self.apply_changes = apply_changes
        self.repair_mode = repair_mode
        self.blend_mode = blend_mode
        self.overlap_width = overlap_width

    def run(self) -> None:
        try:
            self.progress.emit("Repair: reconstructing target tile", -1, -1)
            result = reconstruct_tile_from_donors(self.project, self.request)
            if self.apply_changes:
                self.progress.emit("Repair: saving repaired outputs", -1, -1)
                target = next((tile for tile in self.project.tiles if tile.tile_id == self.request.target_tile_id), None)
                if target is None:
                    raise ValueError(f"Unknown target tile: {self.request.target_tile_id}")
                saved = save_repair_outputs(
                    target,
                    result,
                    self.output_dir,
                    repair_mode=self.repair_mode,
                    blend_mode=self.blend_mode,
                    overlap_width=self.overlap_width,
                )
                self.applied.emit(result, saved)
            else:
                self.preview_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class AtlasStitchWidget(QWidget):
    def __init__(self, viewer: Viewer | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.project: AtlasProject | None = None
        self._target_tile_id: str = ""
        self._lock_guard_active_tile_id: str = ""
        self._lock_guard_press_xy: tuple[float, float] | None = None
        self.viewer: Viewer | None = viewer or current_viewer()
        self._preview_layers: dict[str, Any] = {}
        self._preview_overlay_layers: dict[str, dict[str, Any]] = {}
        self._preview_tile_geometries: dict[str, dict[str, dict[str, Any]]] = {}
        self._preview_downsamples: dict[str, int] = {}
        self._preview_prefix = "Atlas Preview"
        self._opened_tile_layers: dict[str, Any] = {}
        self._opened_tile_order: list[str] = []
        self._opened_tile_bounds_layer: Any | None = None
        self._opened_tile_bounds_name = "Open Tile Bounds"
        self._preview_thread: QThread | None = None
        self._preview_worker: PreviewWorker | None = None
        self._preview_mode: str = "nominal"
        self._export_thread: QThread | None = None
        self._export_worker: ExportWorker | None = None
        self._alignment_thread: QThread | None = None
        self._alignment_worker: AlignmentWorker | None = None
        self._repair_thread: QThread | None = None
        self._repair_worker: RepairWorker | None = None
        self._repair_preview_result: TileRepairResult | None = None
        self._repair_preview_layers: dict[str, Any] = {}
        self._repair_overlap_layer: Any | None = None
        self._repair_roi_layer: Any | None = None
        self._repair_output_folder: str = ""
        self._project_path: str = ""
        self._viewer_mouse_menu_installed = False
        self._viewer_lock_guard_installed = False
        self.setWindowTitle("Atlas Stitch")
        self.setMinimumWidth(900)


        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)
        left_panel = QWidget()
        left_column = QVBoxLayout(left_panel)
        left_column.setSpacing(10)
        left_column.setContentsMargins(0, 0, 0, 0)
        right_panel = QWidget()
        right_column = QVBoxLayout(right_panel)
        right_column.setSpacing(10)
        right_column.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_splitter, 1)

        self.workflow_guide_label = QLabel(
            "Workflow: Load source → Preview nominal → Run auto-registration → Preview refined → Export."
        )
        self.workflow_guide_label.setWordWrap(True)
        left_column.addWidget(self.workflow_guide_label)

        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)
        xml_row = QHBoxLayout()
        xml_row.addWidget(QLabel("Atlas Source"))
        self.xml_path_edit = QLineEdit()
        self.xml_path_edit.setPlaceholderText("Select atlas XML or VE-MIF")
        self.xml_browse_button = QPushButton("Browse...")
        xml_row.addWidget(self.xml_path_edit, 1)
        xml_row.addWidget(self.xml_browse_button)
        input_layout.addLayout(xml_row)

        tile_root_row = QHBoxLayout()
        tile_root_row.addWidget(QLabel("Tile Root Override"))
        self.tile_root_edit = QLineEdit()
        self.tile_root_edit.setPlaceholderText("Optional directory to remap tile paths")
        self.tile_root_browse_button = QPushButton("Browse...")
        tile_root_row.addWidget(self.tile_root_edit, 1)
        tile_root_row.addWidget(self.tile_root_browse_button)
        input_layout.addLayout(tile_root_row)

        load_row = QHBoxLayout()
        self.load_button = QPushButton("Load Source")
        self.load_project_button = QPushButton("Load Project JSON")
        self.save_project_button = QPushButton("Save Project JSON")
        self.load_button.setToolTip("Parse atlas metadata and resolve tile image paths.")
        load_row.addWidget(self.load_button)
        load_row.addWidget(self.load_project_button)
        load_row.addWidget(self.save_project_button)
        load_row.addStretch(1)
        input_layout.addLayout(load_row)
        left_column.addWidget(input_group)

        inspect_group = QGroupBox("Inspect")
        inspect_layout = QFormLayout(inspect_group)
        inspect_layout.setLabelAlignment(Qt.AlignLeft)
        inspect_button_row = QHBoxLayout()
        self.preview_nominal_button = QPushButton("Preview Nominal Layout")
        self.clear_preview_button = QPushButton("Clear Preview Layers")
        self.preview_nominal_button.setToolTip(
            "Show the layout from stage/grid metadata before registration."
        )
        inspect_button_row.addWidget(self.preview_nominal_button)
        inspect_button_row.addWidget(self.clear_preview_button)
        inspect_button_row.addStretch(1)
        inspect_layout.addRow(inspect_button_row)

        table_locate_row = QHBoxLayout()
        self.grid_position_edit = QLineEdit()
        self.grid_position_edit.setPlaceholderText("r8 c11")
        self.grid_position_button = QPushButton("Locate Grid")
        table_locate_row.addWidget(self.grid_position_edit, 1)
        table_locate_row.addWidget(self.grid_position_button)
        inspect_layout.addRow("Locate Grid", table_locate_row)

        self.preview_downsample_edit = QLineEdit()
        self.preview_downsample_edit.setPlaceholderText("8")
        self.preview_downsample_edit.setText("8")
        inspect_layout.addRow("Preview downsample", self.preview_downsample_edit)
        left_column.addWidget(inspect_group)

        register_group = QGroupBox("Register")
        register_layout = QVBoxLayout(register_group)
        overlap_row = QHBoxLayout()
        overlap_row.addWidget(QLabel("Tile Overlap (%)"))
        self.overlap_percent_edit = QLineEdit()
        self.overlap_percent_edit.setPlaceholderText("10")
        self.overlap_percent_edit.setText("10")
        overlap_row.addWidget(self.overlap_percent_edit, 1)
        register_layout.addLayout(overlap_row)

        max_correction_row = QHBoxLayout()
        max_correction_row.addWidget(QLabel("Max Correction (px)"))
        self.max_correction_px_edit = QLineEdit()
        self.max_correction_px_edit.setPlaceholderText("50")
        self.max_correction_px_edit.setText("50")
        self.max_correction_px_edit.setToolTip("Reject an overlap correction if its dx/dy magnitude exceeds this value. Leave blank to disable this limit.")
        max_correction_row.addWidget(self.max_correction_px_edit, 1)
        register_layout.addLayout(max_correction_row)

        min_confidence_row = QHBoxLayout()
        min_confidence_row.addWidget(QLabel("Minimum Confidence"))
        self.min_confidence_edit = QLineEdit()
        self.min_confidence_edit.setPlaceholderText("0.20")
        self.min_confidence_edit.setText("0.20")
        self.min_confidence_edit.setToolTip("Reject an overlap match below this confidence and fall back to nominal spacing.")
        min_confidence_row.addWidget(self.min_confidence_edit, 1)
        register_layout.addLayout(min_confidence_row)

        alignment_row = QHBoxLayout()
        alignment_row.addWidget(QLabel("Alignment Method"))
        self.alignment_method_combo = QComboBox()
        self.alignment_method_combo.addItem("Light Translation", DEFAULT_ALIGNMENT_METHOD)
        self.alignment_method_combo.addItem("Robust Translation", ROBUST_ALIGNMENT_METHOD)
        self.alignment_method_combo.addItem("Robust Translation + Outlier Rejection", ROBUST_OUTLIER_ALIGNMENT_METHOD)
        self.alignment_method_combo.addItem("Feature Matching (SIFT/ORB)", FEATURE_MATCHING_METHOD)
        self.alignment_method_combo.addItem("ECC Translation", ECC_TRANSLATION_METHOD)
        self.alignment_method_combo.addItem("Center-Out Clockwise", CENTER_OUT_CLOCKWISE_METHOD)
        self.alignment_method_combo.addItem("Center-Out Counterclockwise", CENTER_OUT_COUNTERCLOCKWISE_METHOD)
        self.alignment_method_combo.setToolTip("Choose the registration method, then click the alignment button once.")
        alignment_row.addWidget(self.alignment_method_combo, 1)
        register_layout.addLayout(alignment_row)
        self.alignment_method_help = QLabel()
        self.alignment_method_help.setWordWrap(True)
        register_layout.addWidget(self.alignment_method_help)

        register_button_row = QHBoxLayout()
        self.estimate_alignment_button = QPushButton("Run Auto-Registration")
        self.preview_refined_button = QPushButton("Preview Refined Layout")
        self.estimate_alignment_button.setToolTip(
            "Register overlapping tile pairs and solve a global refined layout."
        )
        self.preview_refined_button.setToolTip("Show the optimized layout after auto-registration.")
        register_button_row.addWidget(self.estimate_alignment_button)
        register_button_row.addWidget(self.preview_refined_button)
        register_button_row.addStretch(1)
        register_layout.addLayout(register_button_row)
        left_column.addWidget(register_group)

        manual_group = QGroupBox("Manual Adjustment")
        manual_layout = QHBoxLayout(manual_group)
        self.preview_manual_button = QPushButton("Preview Manual Layout")
        self.save_manual_position_button = QPushButton("Save Manual Position")
        self.save_manual_position_button.setToolTip(
            "Save the current napari layer position for selected opened tile(s)."
        )
        manual_layout.addWidget(self.preview_manual_button)
        manual_layout.addWidget(self.save_manual_position_button)
        manual_layout.addStretch(1)
        left_column.addWidget(manual_group)

        export_group = QGroupBox("Export")
        export_layout = QFormLayout(export_group)
        export_layout.setLabelAlignment(Qt.AlignLeft)
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("Select export directory")
        self.output_folder_browse_button = QPushButton("Browse...")
        output_folder_row = QHBoxLayout()
        output_folder_row.addWidget(self.output_folder_edit, 1)
        output_folder_row.addWidget(self.output_folder_browse_button)
        export_layout.addRow("Output folder", output_folder_row)

        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("Base name (for example atlas)")
        export_layout.addRow("Output name", self.output_name_edit)

        self.chunk_size_edit = QLineEdit()
        self.chunk_size_edit.setPlaceholderText("256")
        self.chunk_size_edit.setText("256")
        export_layout.addRow("Chunk size", self.chunk_size_edit)

        self.build_pyramid_checkbox = QCheckBox("Build multiscale pyramid")
        self.build_pyramid_checkbox.setChecked(True)
        export_layout.addRow("Multiscale", self.build_pyramid_checkbox)

        fusion_row = QHBoxLayout()
        self.fusion_method_combo = QComboBox()
        self.fusion_method_combo.addItem("Overwrite", FUSION_OVERWRITE)
        self.fusion_method_combo.addItem("Linear Blend", FUSION_LINEAR_BLEND)
        self.fusion_method_combo.addItem("Average", FUSION_AVERAGE)
        self.fusion_method_combo.addItem("Max Intensity", FUSION_MAX)
        self.fusion_method_combo.addItem("Min Intensity", FUSION_MIN)
        fusion_row.addWidget(self.fusion_method_combo, 1)
        export_layout.addRow("Fusion method", fusion_row)

        self.export_placement_combo = QComboBox()
        self.export_placement_combo.addItem("Nominal", "nominal")
        self.export_placement_combo.addItem("Refined", "refined")
        self.export_placement_combo.addItem("Manual", "manual")
        export_layout.addRow("Export placement", self.export_placement_combo)

        self.export_worked_only_checkbox = QCheckBox("Export worked/manual tiles only")
        self.export_worked_only_checkbox.setChecked(False)
        export_layout.addRow("Export subset", self.export_worked_only_checkbox)

        export_button_row = QHBoxLayout()
        self.export_button = QPushButton("Export Stitched OME-Zarr")
        self.open_export_button = QPushButton("Open Last Export")
        self.export_button.setToolTip("Export the selected placement mode as an OME-Zarr mosaic.")
        export_button_row.addWidget(self.export_button)
        export_button_row.addWidget(self.open_export_button)
        export_button_row.addStretch(1)
        export_layout.addRow(export_button_row)
        self.export_button.setEnabled(False)
        self.open_export_button.setEnabled(False)
        self.estimate_alignment_button.setEnabled(False)
        self.preview_nominal_button.setEnabled(False)
        self.preview_refined_button.setEnabled(False)
        left_column.addWidget(export_group)

        progress_group = QGroupBox("Status / Activity")
        progress_layout = QVBoxLayout(progress_group)
        self.status_label = QLabel("Select an atlas XML or VE-MIF source, then load it to begin.")
        self.status_label.setWordWrap(True)
        self.progress_label = QLabel("Idle")
        self.progress_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self._reset_progress("Idle")
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        left_column.addWidget(progress_group)

        summary_group = QGroupBox("Atlas Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Atlas metadata summary will appear here.")
        self.summary_text.setMinimumHeight(220)
        summary_layout.addWidget(self.summary_text)
        left_column.addWidget(summary_group, 1)

        table_group = QGroupBox("Tiles")
        table_layout = QVBoxLayout(table_group)
        self.tile_table = QTableWidget(0, 11)
        self.tile_table.setHorizontalHeaderLabels(
            [
                "Grid Position",
                "File Name",
                "Exists",
                "Nominal X",
                "Nominal Y",
                "Refined X",
                "Refined Y",
                "Manual X",
                "Manual Y",
                "Resolved Path",
                "Tile ID",
            ]
        )
        self.tile_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.tile_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tile_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tile_table.verticalHeader().setVisible(False)
        self.tile_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.tile_table)
        right_splitter.addWidget(table_group)

        repair_group = QGroupBox("Advanced Seam Repair")
        repair_layout = QFormLayout(repair_group)
        repair_layout.setLabelAlignment(Qt.AlignLeft)
        repair_intro_label = QLabel("Optional: repair damaged tile borders after layout alignment.")
        repair_intro_label.setWordWrap(True)
        repair_layout.addRow(repair_intro_label)
        self.repair_target_combo = QComboBox()
        self.repair_target_combo.setEnabled(False)
        repair_layout.addRow("Target tile", self.repair_target_combo)

        self.repair_donor_text = QPlainTextEdit()
        self.repair_donor_text.setPlaceholderText(
            "One donor per line: tile_id,direction\nUse the grid-first table to find tiles.\nExample: tile_002,right"
        )
        self.repair_donor_text.setMinimumHeight(90)
        repair_layout.addRow("Donors", self.repair_donor_text)

        self.repair_overlap_width_edit = QLineEdit()
        self.repair_overlap_width_edit.setText(str(DEFAULT_REPAIR_OVERLAP))
        repair_layout.addRow("Overlap width", self.repair_overlap_width_edit)

        self.repair_mode_combo = QComboBox()
        self.repair_mode_combo.addItem("Full overlap", REPAIR_MODE_FULL_OVERLAP)
        self.repair_mode_combo.addItem("ROI-guided repair", REPAIR_MODE_ROI_GUIDED)
        repair_layout.addRow("Repair mode", self.repair_mode_combo)

        self.repair_blend_combo = QComboBox()
        self.repair_blend_combo.addItem("Hard replace", BLEND_MODE_HARD_REPLACE)
        self.repair_blend_combo.addItem("Feather blend", BLEND_MODE_FEATHER)
        repair_layout.addRow("Blend mode", self.repair_blend_combo)

        repair_output_row = QHBoxLayout()
        self.repair_output_folder_edit = QLineEdit()
        self.repair_output_folder_edit.setPlaceholderText("Folder for repaired outputs")
        self.repair_output_browse_button = QPushButton("Browse...")
        repair_output_row.addWidget(self.repair_output_folder_edit, 1)
        repair_output_row.addWidget(self.repair_output_browse_button)
        repair_layout.addRow("Repair output", repair_output_row)

        self.repair_help_label = QLabel(
            "Click 'Repair Tile from Donors' to enter repair mode, then draw one rectangular ROI "
            "in napari if ROI-guided repair is selected. Donors use the order listed above as priority."
        )
        self.repair_help_label.setWordWrap(True)
        repair_layout.addRow(self.repair_help_label)

        repair_button_row = QHBoxLayout()
        self.repair_start_button = QPushButton("Repair Tile from Donors")
        self.show_overlap_preview_button = QPushButton("Show Overlap Preview")
        self.preview_repair_button = QPushButton("Preview Repair")
        self.apply_repair_button = QPushButton("Apply Repair")
        self.cancel_repair_button = QPushButton("Cancel Repair")
        self.clear_repair_preview_button = QPushButton("Clear Repair Preview")
        for button in (
            self.repair_start_button,
            self.show_overlap_preview_button,
            self.preview_repair_button,
            self.apply_repair_button,
            self.cancel_repair_button,
            self.clear_repair_preview_button,
        ):
            repair_button_row.addWidget(button)
        repair_layout.addRow(repair_button_row)
        right_splitter.addWidget(repair_group)

        right_column.addWidget(right_splitter, 1)
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 2)



        self.xml_browse_button.clicked.connect(self._browse_xml)
        self.tile_root_browse_button.clicked.connect(self._browse_tile_root)
        self.output_folder_browse_button.clicked.connect(self._browse_output_folder)
        self.load_button.clicked.connect(self._load_project)
        self.save_project_button.clicked.connect(self._save_project)
        self.load_project_button.clicked.connect(self._load_project_json)
        self.alignment_method_combo.currentIndexChanged.connect(self._update_alignment_method_ui)
        self.estimate_alignment_button.clicked.connect(self._estimate_alignment)
        self.preview_nominal_button.clicked.connect(lambda: self._preview_layout("nominal"))
        self.preview_refined_button.clicked.connect(lambda: self._preview_layout("refined"))
        self.preview_manual_button.clicked.connect(lambda: self._preview_layout("manual"))
        self.save_manual_position_button.clicked.connect(self._save_current_manual_positions)
        self.clear_preview_button.clicked.connect(self._clear_preview)
        self.export_button.clicked.connect(self._export_omezarr)
        self.open_export_button.clicked.connect(self._open_export)
        self.grid_position_button.clicked.connect(self._locate_grid_position)
        self.grid_position_edit.returnPressed.connect(self._locate_grid_position)
        self.tile_table.itemSelectionChanged.connect(self._handle_tile_table_selection_changed)
        self.tile_table.cellDoubleClicked.connect(self._handle_tile_table_double_click)
        self.repair_output_browse_button.clicked.connect(self._browse_repair_output_folder)
        self.repair_start_button.clicked.connect(self._enter_repair_mode)
        self.show_overlap_preview_button.clicked.connect(self._show_repair_overlap_preview)
        self.preview_repair_button.clicked.connect(self._preview_repair)
        self.apply_repair_button.clicked.connect(self._apply_repair)
        self.cancel_repair_button.clicked.connect(self._cancel_repair)
        self.clear_repair_preview_button.clicked.connect(self._clear_repair_preview)

        self.xml_path_edit.editingFinished.connect(self._save_persistent_ui_state)
        self.tile_root_edit.editingFinished.connect(self._save_persistent_ui_state)
        self.output_folder_edit.editingFinished.connect(self._save_persistent_ui_state)
        self.repair_output_folder_edit.editingFinished.connect(self._save_persistent_ui_state)
        self.output_name_edit.editingFinished.connect(self._save_persistent_ui_state)
        self.chunk_size_edit.editingFinished.connect(self._save_persistent_ui_state)
        self.preview_downsample_edit.editingFinished.connect(self._save_persistent_ui_state)
        self.overlap_percent_edit.editingFinished.connect(self._save_persistent_ui_state)
        self.max_correction_px_edit.editingFinished.connect(self._save_persistent_ui_state)
        self.min_confidence_edit.editingFinished.connect(self._save_persistent_ui_state)

        self.build_pyramid_checkbox.toggled.connect(lambda _checked: self._save_persistent_ui_state())
        self.alignment_method_combo.currentIndexChanged.connect(lambda _index: self._save_persistent_ui_state())
        self.fusion_method_combo.currentIndexChanged.connect(lambda _index: self._save_persistent_ui_state())
        self.export_placement_combo.currentIndexChanged.connect(lambda _index: self._save_persistent_ui_state())


        self._load_persistent_ui_state()
        self._update_alignment_method_ui()
        self._update_refinement_controls()
        self._update_repair_controls()
        self._install_viewer_mouse_menu()
        self._install_viewer_lock_guard()        

    def _browse_xml(self) -> None:
        start_dir = self._default_existing_directory(
            self.xml_path_edit.text().strip(),
            self.tile_root_edit.text().strip(),
        )
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Atlas Source",
            start_dir,
            "Atlas Sources (*.xml *.ve-mif);;XML Files (*.xml);;VE-MIF Files (*.ve-mif);;All Files (*)",
        )
        if selected:
            self.xml_path_edit.setText(selected)
            self._save_persistent_ui_state()

    def _browse_tile_root(self) -> None:
        start_dir = self._default_existing_directory(
            self.tile_root_edit.text().strip(),
            self.xml_path_edit.text().strip(),
        )
        selected = QFileDialog.getExistingDirectory(self, "Select Tile Root Override", start_dir)
        if selected:
            self.tile_root_edit.setText(selected)
            self._save_persistent_ui_state()

    def _browse_output_folder(self) -> None:
        start_dir = self._default_existing_directory(
            self.output_folder_edit.text().strip(),
            self.xml_path_edit.text().strip(),
        )
        selected = QFileDialog.getExistingDirectory(self, "Select Output Folder", start_dir)
        if selected:
            self.output_folder_edit.setText(selected)
            self._save_persistent_ui_state()

    def _browse_repair_output_folder(self) -> None:
        start_dir = self._default_existing_directory(
            self.repair_output_folder_edit.text().strip(),
            self.output_folder_edit.text().strip(),
            self.xml_path_edit.text().strip(),
        )
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Repair Output Folder",
            start_dir,
        )
        if selected:
            self.repair_output_folder_edit.setText(selected)
            self._save_persistent_ui_state()


    def _settings(self) -> QSettings:
        return QSettings(ATLAS_STITCH_SETTINGS_ORG, ATLAS_STITCH_SETTINGS_APP)

    def _load_persistent_ui_state(self) -> None:
        settings = self._settings()

        self.xml_path_edit.setText(str(settings.value("paths/xml_path", "", type=str) or ""))
        self.tile_root_edit.setText(str(settings.value("paths/tile_root_override", "", type=str) or ""))
        self.output_folder_edit.setText(str(settings.value("paths/output_folder", "", type=str) or ""))
        self.repair_output_folder_edit.setText(str(settings.value("paths/repair_output_folder", "", type=str) or ""))
        self.output_name_edit.setText(str(settings.value("export/output_name", "", type=str) or ""))

        self.chunk_size_edit.setText(str(settings.value("export/chunk_size", "256", type=str) or "256"))
        self.preview_downsample_edit.setText(str(settings.value("preview/downsample", "8", type=str) or "8"))
        self.overlap_percent_edit.setText(str(settings.value("alignment/overlap_percent", "10", type=str) or "10"))
        self.max_correction_px_edit.setText(str(settings.value("alignment/max_correction_px", "50", type=str) or "50"))
        self.min_confidence_edit.setText(str(settings.value("alignment/min_confidence", "0.20", type=str) or "0.20"))

        build_pyramid = settings.value("export/build_pyramid", True, type=bool)
        self.build_pyramid_checkbox.setChecked(bool(build_pyramid))

        alignment_method = str(settings.value("alignment/method", "", type=str) or "")
        if alignment_method:
            index = self.alignment_method_combo.findData(alignment_method)
            if index >= 0:
                self.alignment_method_combo.setCurrentIndex(index)

        fusion_method = str(settings.value("export/fusion_method", "", type=str) or "")
        if fusion_method:
            index = self.fusion_method_combo.findData(fusion_method)
            if index >= 0:
                self.fusion_method_combo.setCurrentIndex(index)

        export_placement = str(settings.value("export/placement_mode", "", type=str) or "")
        if export_placement:
            index = self.export_placement_combo.findData(export_placement)
            if index >= 0:
                self.export_placement_combo.setCurrentIndex(index)

    def _save_persistent_ui_state(self) -> None:
        settings = self._settings()

        settings.setValue("paths/xml_path", self.xml_path_edit.text().strip())
        settings.setValue("paths/tile_root_override", self.tile_root_edit.text().strip())
        settings.setValue("paths/output_folder", self.output_folder_edit.text().strip())
        settings.setValue("paths/repair_output_folder", self.repair_output_folder_edit.text().strip())
        settings.setValue("export/output_name", self.output_name_edit.text().strip())

        settings.setValue("export/chunk_size", self.chunk_size_edit.text().strip() or "256")
        settings.setValue("preview/downsample", self.preview_downsample_edit.text().strip() or "8")
        settings.setValue("alignment/overlap_percent", self.overlap_percent_edit.text().strip() or "10")
        settings.setValue("alignment/max_correction_px", self.max_correction_px_edit.text().strip())
        settings.setValue("alignment/min_confidence", self.min_confidence_edit.text().strip() or "0.20")

        settings.setValue("export/build_pyramid", bool(self.build_pyramid_checkbox.isChecked()))
        settings.setValue("alignment/method", str(self.alignment_method_combo.currentData() or ""))
        settings.setValue("export/fusion_method", str(self.fusion_method_combo.currentData() or ""))
        settings.setValue("export/placement_mode", str(self.export_placement_combo.currentData() or ""))

        settings.sync()

    def _default_existing_directory(self, *candidates: str) -> str:
        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text:
                continue
            path = Path(text).expanduser()
            if path.is_file():
                parent = path.parent
                if parent.exists():
                    return str(parent)
            if path.is_dir():
                return str(path)
            parent = path.parent
            if parent.exists():
                return str(parent)
        return str(Path.home())

    def _load_project(self) -> None:
        xml_path = self.xml_path_edit.text().strip()
        tile_root_override = self.tile_root_edit.text().strip() or None
        if not xml_path:
            self._set_status("Select an atlas XML or VE-MIF source first.")
            return
        if not Path(xml_path).expanduser().exists():
            self._set_status("Atlas source file was not found.")
            return

        try:
            self.project = parse_atlas_source(xml_path, tile_root_override=tile_root_override)
        except Exception as exc:
            self.project = None
            self._project_path = ""
            self.summary_text.clear()
            self.tile_table.setRowCount(0)
            self._update_refinement_controls()
            self._set_status(f"Failed to load atlas source: {exc}")
            self._reset_progress("Idle")
            self._save_persistent_ui_state()
            return

        self._project_path = ""
        _refresh_project_tile_availability(self.project)
        self._sync_alignment_method_from_project()
        self._apply_recommended_preview_downsample()
        self._populate_summary()
        self._populate_tile_table()
        self._refresh_repair_tile_options()
        self._clear_opened_tile_layers()
        self._clear_repair_preview()
        self._clear_repair_overlap_layer()
        self._clear_preview_layers()
        self._update_open_export_enabled()
        self._update_refinement_controls()
        self._update_repair_controls()
        self._set_status(self._status_message())
        self._reset_progress("Ready")
        self._save_persistent_ui_state()


    def _save_project(self) -> None:
        if self.project is None:
            self._set_status("No project loaded to save.")
            return
        default_name = f"{self.project.metadata.atlas_name or 'atlas'}.json"
        selected, _ = QFileDialog.getSaveFileName(self, "Save Atlas Project", default_name, "JSON Files (*.json)")
        if not selected:
            self._set_status("Save cancelled.")
            return
        destination = Path(selected)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        try:
            save_atlas_project(self.project, str(destination))
        except Exception as exc:
            self._set_status(f"Failed to save project: {exc}")
            return
        self._project_path = str(destination)
        self._save_persistent_ui_state()

        if self.project.last_export.path:
            self.project.last_export.atlas_project_path = self._project_path
            self._populate_summary()
        self._set_status("Project JSON saved.")

    def _load_project_json(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Load Atlas Project", "", "JSON Files (*.json)")
        if not selected:
            self._set_status("Load cancelled.")
            return
        try:
            project = load_atlas_project(selected)
        except Exception as exc:
            self._set_status(f"Failed to load project: {exc}")
            return
        self.project = project
        self.xml_path_edit.setText(str(project.metadata.xml_path or ""))
        self.tile_root_edit.setText(str(project.metadata.tile_root_override or ""))

        self._project_path = str(Path(selected).expanduser())
        _refresh_project_tile_availability(self.project)
        self._sync_alignment_method_from_project()
        self._apply_recommended_preview_downsample()
        self._populate_summary()
        self._populate_tile_table()
        self._refresh_repair_tile_options()
        self._clear_opened_tile_layers()
        self._clear_repair_preview()
        self._clear_repair_overlap_layer()
        self._clear_preview_layers()
        self._update_open_export_enabled()
        self._update_refinement_controls()
        self._update_repair_controls()
        self._set_status(self._status_message())
        self._reset_progress("Ready")
        self._save_persistent_ui_state()

    def _estimate_alignment(self) -> None:
        project = self.project
        if project is None:
            self._set_status("Load a source or project JSON before running auto-registration.")
            return
        if self._alignment_thread and self._alignment_thread.isRunning():
            self._set_status("Auto-registration is already running.")
            return
        if not self._store_processing_settings():
            return
        min_confidence = self._parse_min_confidence()
        if min_confidence is None:
            return
        max_correction_px = self._parse_max_correction_px()
        if max_correction_px is False:
            return
        self.estimate_alignment_button.setEnabled(False)
        self.preview_refined_button.setEnabled(False)
        method = self._selected_alignment_method()
        method_label = _alignment_method_label(method)
        self.progress_label.setText(f"Alignment ({method_label}): queued")
        self._set_progress_busy()
        self._set_status(f"Auto-registration running with {method_label.lower()}.")
        self._start_alignment_worker(project, method, min_confidence=min_confidence, max_correction_px=max_correction_px)

    def _preview_layout(self, placement_mode: str) -> None:
        if self.viewer is None:
            self._set_status("Napari viewer required for preview.")
            return
        if self.project is None:
            self._set_status("Load an atlas project before previewing.")
            return
        if self._preview_thread and self._preview_thread.isRunning():
            self._set_status("Preview already running.")
            return
        downsample = self._parse_preview_downsample()
        if downsample is None:
            return
        self._preview_mode = placement_mode
        self._preview_downsamples[placement_mode] = downsample
        self.preview_nominal_button.setEnabled(False)
        self.preview_refined_button.setEnabled(False)
        self.preview_manual_button.setEnabled(False)
        self.save_manual_position_button.setEnabled(False)
        mode_label = _placement_mode_label(placement_mode)
        self.progress_label.setText(f"Preview {mode_label}: queued")
        self._set_progress_busy()
        self._set_status(f"Generating {mode_label.lower()} coarse preview.")
        self._start_preview_worker(self.project.tiles, downsample, placement_mode)

    def _clear_preview(self) -> None:
        self._clear_preview_layers()
        self._update_refinement_controls()
        self._set_status("Atlas preview cleared.")
        self._reset_progress("Ready")

      #mouse menu for opened tile layers in napari viewer  
    def _install_viewer_mouse_menu(self) -> None:
        if self.viewer is None:
            return
        if self._viewer_mouse_menu_installed:
            return
        self.viewer.mouse_drag_callbacks.append(self._atlas_right_click_menu_callback)
        self._viewer_mouse_menu_installed = True

    def _atlas_right_click_menu_callback(self, viewer, event):
        if getattr(event, "button", None) != 2:
            return

        self._update_opened_tile_bounds_overlay()

        world_position = getattr(event, "position", None)
        tile = self._tile_at_world_position(world_position)
        if tile is None:
            return

        try:
            event.handled = True
        except Exception:
            pass

        self._target_tile_id = tile.tile_id
        self._activate_tile_target(tile.tile_id, sync_table=False, center_camera=False)
        self._show_tile_context_menu(tile)
        self._finalize_tile_action(tile.tile_id)
        yield

    def _install_viewer_lock_guard(self) -> None:
        if self.viewer is None:
            return
        if self._viewer_lock_guard_installed:
            return
        self.viewer.mouse_drag_callbacks.append(self._atlas_locked_tile_guard_callback)
        self._viewer_lock_guard_installed = True


    def _atlas_locked_tile_guard_callback(self, viewer, event):
        if getattr(event, "button", None) != 1:
            return

        active_layer = None
        try:
            active_layer = viewer.layers.selection.active
        except Exception:
            active_layer = None

        tile_id = self._tile_id_for_layer(active_layer)
        if not tile_id:
            return

        tile = self._tile_by_id(tile_id)
        if tile is None:
            return

        if not bool(tile.metadata.get("manual_locked")):
            return

        layer = self._opened_tile_layers.get(tile_id)
        if layer is None:
            return

        self._lock_guard_active_tile_id = tile_id
        try:
            self._lock_guard_press_xy = self._layer_live_xy(layer)
        except Exception:
            self._lock_guard_press_xy = None

        # allow napari event cycle to run, then force the locked tile back
        yield

        self._restore_locked_layer_position(tile_id)
        self._lock_guard_active_tile_id = ""
        self._lock_guard_press_xy = None

        try:
            event.handled = True
        except Exception:
            pass


    def _tile_id_for_layer(self, layer) -> str:
        if layer is None:
            return ""
        for tile_id, opened_layer in self._opened_tile_layers.items():
            if opened_layer is layer:
                return tile_id
        return ""

    def _restore_locked_layer_position(self, tile_id: str) -> None:
        tile = self._tile_by_id(tile_id)
        layer = self._opened_tile_layers.get(tile_id)

        if tile is None or layer is None:
            return

        if not bool(tile.metadata.get("manual_locked")):
            return

        manual_xy = self._manual_position_for_tile(tile)
        if manual_xy is None:
            return

        target_x, target_y = manual_xy
        try:
            self._set_layer_live_xy(layer, target_x, target_y)
            
        except Exception:
            return
        self._update_opened_tile_bounds_overlay()
        self._set_status(f"Tile {tile_id} is locked; movement was reverted.")

    def _tile_at_world_position(self, world_position) -> TileRecord | None:
        if self.project is None or world_position is None:
            return None
        if len(world_position) < 2:
            return None

        y = float(world_position[0])
        x = float(world_position[1])

        for tile_id in reversed(self._opened_tile_order):
            tile = self._tile_by_id(tile_id)
            layer = self._opened_tile_layers.get(tile_id)
            if tile is None or layer is None:
                continue

            try:
                layer_x, layer_y = self._layer_live_xy(layer)
            except Exception:
                continue

            data = getattr(layer, "data", None)
            if data is None:
                continue

            shape = np.asarray(data).shape
            if len(shape) < 2:
                continue

            height = float(shape[0])
            width = float(shape[1])

            if layer_y <= y <= (layer_y + height) and layer_x <= x <= (layer_x + width):
                return tile

        return None

    def _opened_tile_bounds_shape_from_layer(self, tile_id: str) -> np.ndarray | None:
        layer = self._opened_tile_layers.get(tile_id)
        if layer is None:
            return None

        data = getattr(layer, "data", None)
        if data is None:
            return None

        shape = np.asarray(data).shape
        if len(shape) < 2:
            return None

        height = float(shape[0])
        width = float(shape[1])

        try:
            layer_x, layer_y = self._layer_live_xy(layer)
        except Exception:
            return None

        return np.asarray(
            [
                [layer_y, layer_x],
                [layer_y + height, layer_x],
                [layer_y + height, layer_x + width],
                [layer_y, layer_x + width],
            ],
            dtype=float,
        )


    def _tile_by_id(self, tile_id: str) -> TileRecord | None:
        if self.project is None:
            return None
        wanted = str(tile_id or "").strip()
        for tile in self.project.tiles:
            if tile.tile_id == wanted:
                return tile
        return None

    def _activate_opened_tile_layer(self, tile_id: str) -> None:
        if self.viewer is None:
            return
        layer = self._opened_tile_layers.get(tile_id)
        if layer is None:
            return
        try:
            self.viewer.layers.selection.active = layer
        except Exception:
            pass

    def _disarm_tile_interaction(self, tile_id: str = "") -> None:
        """
        Break any sticky post-menu / post-transform interaction state on the tile.

        Goal:
        - keep the tile selected if possible
        - force its mode out of transform/edit states
        - refresh overlays from the actual live position
        """
        if self.viewer is None:
            return

        wanted_tile_id = str(tile_id or self._target_tile_id or "").strip()
        layer = self._opened_tile_layers.get(wanted_tile_id) if wanted_tile_id else None

        if layer is None:
            try:
                layer = self.viewer.layers.selection.active
            except Exception:
                layer = None

        if layer is None:
            return

        try:
            # Keep it as the active layer, but out of any edit/transform mode
            self.viewer.layers.selection.active = layer
        except Exception:
            pass

        # Try common napari layer mode names safely.
        for neutral_mode in ("pan_zoom", "select"):
            try:
                if hasattr(layer, "mode"):
                    layer.mode = neutral_mode
                    break
            except Exception:
                continue

        # Refresh the visual bounds from the actual live layer position.
        try:
            self._update_opened_tile_bounds_overlay()
        except Exception:
            pass


    def _finalize_tile_action(self, tile_id: str = "") -> None:
        """
        Common tail step after save/lock/unlock/clear/menu actions.
        """
        self._disarm_tile_interaction(tile_id)


    def _select_tile_in_table(self, tile_id: str) -> None:
        if self.project is None:
            return
        for row_index, tile in enumerate(self.project.tiles):
            if tile.tile_id == tile_id:
                self.tile_table.clearSelection()
                self.tile_table.selectRow(row_index)
                self.tile_table.setCurrentCell(row_index, 0)
                self.tile_table.scrollToItem(self.tile_table.item(row_index, 0))
                return

    def _save_manual_position_for_tile(self, tile_id: str) -> bool:
        tile = self._tile_by_id(tile_id)
        if tile is None:
            self._set_status(f"Unknown tile: {tile_id}")
            return False

        layer = self._opened_tile_layers.get(tile_id)
        if layer is None:
            self._set_status(f"Tile {tile_id} is not currently open.")
            return False

        if bool(tile.metadata.get("manual_locked")) and (
            tile.transform.manual_x is not None and tile.transform.manual_y is not None
        ):
            self._set_status(f"Tile {tile_id} is locked. Unlock it before saving a new manual position.")
            return False

        try:
            manual_x, manual_y = self._layer_live_xy(layer)
        except Exception:
            self._set_status(f"Could not read current layer position for {tile_id}.")
            return False

        tile.transform.manual_x = float(manual_x)
        tile.transform.manual_y = float(manual_y)
        tile.metadata["manual_worked"] = True
        tile.metadata["manual_saved_time"] = datetime.now().astimezone().isoformat(timespec="seconds")

        if self.project is not None:
            self.project.metadata.extra_metadata["atlas_stitch_manual_tile_ids"] = self._worked_tile_ids()

        self._populate_summary()
        self._populate_tile_table()
        self._update_refinement_controls()
        self._set_status(f"Saved manual position for {tile_id}.")
        self._finalize_tile_action(tile_id)
        return True
    
    def _set_tile_manual_lock(self, tile_id: str, locked: bool) -> None:
        tile = self._tile_by_id(tile_id)
        if tile is None:
            self._set_status(f"Unknown tile: {tile_id}")
            return

        layer = self._opened_tile_layers.get(tile_id)

        if locked:
            if layer is None:
                self._set_status(f"Tile {tile_id} is not currently open.")
                return

            try:
                current_x, current_y = self._layer_live_xy(layer)
            except Exception:
                self._set_status(f"Could not read current layer position for {tile_id}.")
                return

            tile.transform.manual_x = float(current_x)
            tile.transform.manual_y = float(current_y)
            tile.metadata["manual_worked"] = True
            tile.metadata["manual_locked"] = True
            tile.metadata["manual_saved_time"] = datetime.now().astimezone().isoformat(timespec="seconds")

            try:
                self._set_layer_live_xy(layer, float(current_x), float(current_y))
            except Exception:
                pass

            if self.project is not None:
                self.project.metadata.extra_metadata["atlas_stitch_manual_tile_ids"] = self._worked_tile_ids()

            self._populate_tile_table()
            self._populate_summary()
            self._update_refinement_controls()
            self._set_status(f"Locked tile {tile_id} at current position.")
            self._finalize_tile_action(tile_id)
            return

        tile.metadata["manual_locked"] = False
        self._populate_tile_table()
        self._populate_summary()
        self._update_refinement_controls()
        self._set_status(f"Unlocked tile {tile_id}.")
        self._finalize_tile_action(tile_id)

    def _clear_manual_position_for_tile(self, tile_id: str) -> None:
        tile = self._tile_by_id(tile_id)
        if tile is None:
            self._set_status(f"Unknown tile: {tile_id}")
            return
        tile.transform.manual_x = None
        tile.transform.manual_y = None
        tile.transform.manual_z = None
        tile.metadata["manual_worked"] = False
        tile.metadata["manual_locked"] = False

        if self.project is not None:
            self.project.metadata.extra_metadata["atlas_stitch_manual_tile_ids"] = self._worked_tile_ids()

        self._populate_tile_table()
        self._populate_summary()
        self._update_refinement_controls()
        self._set_status(f"Cleared manual position for {tile_id}.")
        self._finalize_tile_action(tile_id)

    def _show_tile_context_menu(self, tile: TileRecord) -> None:
        menu = QMenu(self)

        title_action = menu.addAction(f"Tile: {tile.tile_id}")
        title_action.setEnabled(False)
        menu.addSeparator()

        is_locked = bool(tile.metadata.get("manual_locked"))
        has_manual = tile.transform.manual_x is not None and tile.transform.manual_y is not None

        lock_here_action = menu.addAction("Lock at Current Position")
        save_action = menu.addAction("Save Manual Position")
        unlock_action = menu.addAction("Unlock Tile")
        clear_action = menu.addAction("Clear Saved Position")
        menu.addSeparator()
        select_row_action = menu.addAction("Select Tile in Table")

        if is_locked:
            lock_here_action.setEnabled(False)
            save_action.setEnabled(False)
        else:
            unlock_action.setEnabled(False)

        if not has_manual:
            clear_action.setEnabled(False)

        chosen = menu.exec(QCursor.pos())

        if chosen is None:
            self._finalize_tile_action(tile.tile_id)
            return

        if chosen == lock_here_action:
            self._set_tile_manual_lock(tile.tile_id, True)
            self._finalize_tile_action(tile.tile_id)
            return

        if chosen == save_action:
            self._save_manual_position_for_tile(tile.tile_id)
            self._finalize_tile_action(tile.tile_id)
            return

        if chosen == unlock_action:
            self._set_tile_manual_lock(tile.tile_id, False)
            self._finalize_tile_action(tile.tile_id)
            return

        if chosen == clear_action:
            self._clear_manual_position_for_tile(tile.tile_id)
            self._finalize_tile_action(tile.tile_id)
            return

        if chosen == select_row_action:
            self._select_tile_row_safely(tile.tile_id)
            self._finalize_tile_action(tile.tile_id)
            return

        self._finalize_tile_action(tile.tile_id)
        
    def _select_tile_row_safely(self, tile_id: str) -> None:
        if self.project is None:
            return

        previous = self.tile_table.blockSignals(True)
        try:
            for row_index, tile in enumerate(self.project.tiles):
                if tile.tile_id == tile_id:
                    self.tile_table.clearSelection()
                    self.tile_table.selectRow(row_index)
                    self.tile_table.setCurrentCell(row_index, 0)
                    self.tile_table.scrollToItem(self.tile_table.item(row_index, 0))
                    break
        finally:
            self.tile_table.blockSignals(previous)


    def _export_omezarr(self) -> None:
        project = self.project
        if project is None:
            self._set_status("Load an atlas project before exporting.")
            return
        if self._export_thread and self._export_thread.isRunning():
            self._set_status("Export already running.")
            return

        output_folder = self.output_folder_edit.text().strip()
        if not output_folder:
            self._set_status("Select an output folder before exporting.")
            return

        chunk_size = self._parse_chunk_size()
        if chunk_size is None:
            return

        output_name = self.output_name_edit.text().strip() or (project.metadata.atlas_name or "atlas")
        destination = Path(output_folder).expanduser() / output_name

        if not self._store_processing_settings():
            return

        placement_mode = str(self.export_placement_combo.currentData() or "nominal")
        tile_ids: list[str] | None = None
        if self.export_worked_only_checkbox.isChecked():
            tile_ids = self._worked_tile_ids()
            if not tile_ids:
                self._set_status("No worked/manual tiles are available for export.")
                return

        self.export_button.setEnabled(False)
        self.progress_label.setText("Export: queued")
        self._set_progress_busy()
        self._set_status(f"Exporting stitched OME-Zarr using {placement_mode} placement.")
        self._save_persistent_ui_state()
        self._start_export_worker(
            project,
            str(destination),
            chunk_size=chunk_size,
            build_pyramid=self.build_pyramid_checkbox.isChecked(),
            fusion_method=self._selected_fusion_method(),
            placement_mode=placement_mode,
            tile_ids=tile_ids,
        )

    def _start_preview_worker(self, tiles: list[TileRecord], downsample: int, placement_mode: str) -> None:
        self._cleanup_preview_worker()
        thread = QThread()
        worker = PreviewWorker(tiles, downsample, placement_mode)
        worker.moveToThread(thread)
        worker.progress.connect(self._handle_preview_progress)
        worker.preview_ready.connect(self._handle_preview_ready)
        worker.error.connect(self._handle_preview_error)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._cleanup_preview_worker)
        thread.finished.connect(self._update_refinement_controls)
        thread.started.connect(worker.run)
        self._preview_worker = worker
        self._preview_thread = thread
        thread.start()

    def _start_alignment_worker(
        self,
        project: AtlasProject,
        method: str,
        *,
        min_confidence: float,
        max_correction_px: float | None,
    ) -> None:
        self._cleanup_alignment_worker()
        thread = QThread()
        worker = AlignmentWorker(project, method, min_confidence, max_correction_px)
        worker.moveToThread(thread)
        worker.progress.connect(self._handle_alignment_progress)
        worker.completed.connect(self._handle_alignment_complete)
        worker.error.connect(self._handle_alignment_error)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._cleanup_alignment_worker)
        thread.finished.connect(self._update_refinement_controls)
        thread.started.connect(worker.run)
        self._alignment_worker = worker
        self._alignment_thread = thread
        thread.start()

    def _start_export_worker(
        self,
        project: AtlasProject,
        output_path: str,
        *,
        chunk_size: int,
        build_pyramid: bool,
        fusion_method: str,
        placement_mode: str = "nominal",
        tile_ids: list[str] | None = None,
    ) -> None:
        self._cleanup_export_worker()
        thread = QThread()
        worker = ExportWorker(
            project,
            output_path,
            chunk_size,
            build_pyramid,
            fusion_method,
            atlas_project_path=self._project_path,
            placement_mode=placement_mode,
            tile_ids=tile_ids,
        )
        worker.moveToThread(thread)
        worker.progress.connect(self._handle_export_progress)
        worker.completed.connect(self._handle_export_complete)
        worker.error.connect(self._handle_export_error)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._cleanup_export_worker)
        thread.finished.connect(self._update_refinement_controls)
        thread.started.connect(worker.run)
        self._export_worker = worker
        self._export_thread = thread
        thread.start()

    def _handle_preview_progress(self, message: str, current: int, total: int) -> None:
        self.progress_label.setText(message)
        self._update_progress_bar(current, total)

    def _handle_preview_ready(self, payload: Any, metadata: dict[str, Any]) -> None:
        if self.viewer is None:
            return
        mode_label = _placement_mode_label(self._preview_mode)
        self._remove_preview_layer(self._preview_mode)
        layer_name = self._preview_layer_name(self._preview_mode)
        translate = metadata.get("translate", (0.0, 0.0))
        layer = self.viewer.add_image(payload, name=layer_name, translate=translate, blending="additive")
        self._preview_layers[self._preview_mode] = layer
        self._render_preview_overlays(self._preview_mode)
        self._sync_selected_tile_to_preview()
        self._update_refinement_controls()
        self.progress_label.setText("Preview complete")
        self._set_progress_complete()
        self._set_status(f"{mode_label} layout preview ready.")

    def _handle_preview_error(self, message: str) -> None:
        self._update_refinement_controls()
        self.progress_label.setText(f"Preview failed: {message}")
        self._reset_progress(f"Preview failed: {message}")
        self._set_status(f"Preview failed: {message}")

    def _handle_alignment_progress(self, message: str, current: int, total: int) -> None:
        self.progress_label.setText(message)
        self._update_progress_bar(current, total)

    def _handle_alignment_complete(self, solved_project: AtlasProject) -> None:
        self.project = solved_project
        self.estimate_alignment_button.setEnabled(True)
        self._update_refinement_controls()
        self._populate_summary()
        self._populate_tile_table()
        self._refresh_repair_tile_options()
        method_label = _alignment_method_summary_text(self.project.metadata.extra_metadata)
        elapsed_text = _elapsed_registration_text(self.project.metadata.extra_metadata)
        complete_text = f"Auto-registration complete ({method_label})"
        if elapsed_text:
            complete_text = f"{complete_text} in {elapsed_text}"
        self.progress_label.setText(complete_text)
        self._set_progress_complete()
        self._set_status(_alignment_completion_status(self.project))

    def _handle_alignment_error(self, message: str) -> None:
        self.estimate_alignment_button.setEnabled(True)
        self._update_refinement_controls()
        self.progress_label.setText(f"Auto-registration failed: {message}")
        self._reset_progress(f"Auto-registration failed: {message}")
        self._set_status("Auto-registration failed. Check tile paths, overlap percent, and alignment method.")

    def _handle_export_progress(self, stage: str, current: int, total: int) -> None:
        message = _format_export_progress(stage, current, total)
        self.progress_label.setText(message)
        self._update_progress_bar(current, total)

    def _handle_export_complete(
        self,
        exported_path: str,
        chunk_size: int,
        build_pyramid: bool,
        atlas_project_path: str,
        fusion_method: str,
        placement_mode: str,
    ) -> None:
        self.export_button.setEnabled(True)
        self.progress_label.setText(f"Export complete: {exported_path}")
        self._set_progress_complete()
        self._set_status("Stitched OME-Zarr export complete. Use Open Last Export to inspect it in napari.")

        if self.project is not None:
            exported_tile_ids = None
            if self.export_worked_only_checkbox.isChecked():
                exported_tile_ids = set(self._worked_tile_ids())

            tile_count = 0
            for tile in self.project.tiles:
                if not (tile.exists and tile.resolved_path):
                    continue
                if exported_tile_ids is not None and tile.tile_id not in exported_tile_ids:
                    continue

                if placement_mode == "manual":
                    if tile.transform.manual_x is None or tile.transform.manual_y is None:
                        continue
                elif placement_mode == "refined":
                    if tile.transform.refined_x is None or tile.transform.refined_y is None:
                        continue
                else:
                    nominal_x_ok = (tile.start_x is not None) or (tile.transform.nominal_x is not None)
                    nominal_y_ok = (tile.start_y is not None) or (tile.transform.nominal_y is not None)
                    if not (nominal_x_ok and nominal_y_ok):
                        continue

                tile_count += 1

            self.project.last_export = AtlasExportInfo(
                path=exported_path,
                mode=str(placement_mode or "nominal"),
                time=datetime.now().astimezone().isoformat(timespec="seconds"),
                chunk_size=chunk_size,
                build_pyramid=build_pyramid,
                tile_count=tile_count,
                status=f"completed ({fusion_method})",
                atlas_project_path=atlas_project_path,
            )
            self._populate_summary()
            self._update_open_export_enabled()


    def _handle_export_error(self, message: str) -> None:
        self.export_button.setEnabled(True)
        self.progress_label.setText(f"Export failed: {message}")
        self._reset_progress(f"Export failed: {message}")
        self._set_status("OME-Zarr export failed. Check output folder, tile paths, and selected placement mode.")
        if self.project is not None and self.project.last_export.path:
            self.project.last_export.status = "failed"
            self._populate_summary()

    def _parse_preview_downsample(self) -> int | None:
        text = self.preview_downsample_edit.text().strip()
        if not text:
            return self._recommended_preview_downsample()
        try:
            value = int(text)
        except ValueError:
            self._set_status("Preview downsample must be a positive integer.")
            return None
        if value < 1:
            self._set_status("Preview downsample must be at least 1.")
            return None
        recommended = self._recommended_preview_downsample()
        if value < recommended:
            self.preview_downsample_edit.setText(str(recommended))
            self._set_status(
                f"Preview downsample increased from {value} to {recommended} for this atlas size."
            )
            return recommended
        return value

    def _apply_recommended_preview_downsample(self) -> None:
        recommended = self._recommended_preview_downsample()
        current_text = self.preview_downsample_edit.text().strip()
        try:
            current_value = int(current_text) if current_text else 0
        except ValueError:
            current_value = 0
        if recommended > max(0, current_value):
            self.preview_downsample_edit.setText(str(recommended))

    def _recommended_preview_downsample(self) -> int:
        project = self.project
        if project is None:
            return 1
        width = project.metadata.image_width
        height = project.metadata.image_height
        if width is None or height is None or width <= 0 or height <= 0:
            width, height = _project_nominal_canvas(project)
        if width <= 0 or height <= 0:
            return 1
        limit_dim = max(
            math.ceil(width / 4096),
            math.ceil(height / 4096),
        )
        limit_bytes = math.ceil(math.sqrt((width * height * 4) / float(128 * 1024 * 1024)))
        recommended = max(1, limit_dim, limit_bytes)
        return _next_power_of_two(recommended)

    def _parse_chunk_size(self) -> int | None:
        text = self.chunk_size_edit.text().strip()
        if not text:
            return 256
        try:
            value = int(text)
        except ValueError:
            self._set_status("Chunk size must be a positive integer.")
            return None
        if value < 1:
            self._set_status("Chunk size must be at least 1.")
            return None
        return value

    def _selected_alignment_method(self) -> str:
        return str(self.alignment_method_combo.currentData() or DEFAULT_ALIGNMENT_METHOD)

    def _selected_fusion_method(self) -> str:
        return str(self.fusion_method_combo.currentData() or FUSION_OVERWRITE)

    def _parse_overlap_percent(self) -> float | None:
        text = self.overlap_percent_edit.text().strip()
        if not text:
            return 10.0
        try:
            value = float(text)
        except ValueError:
            self._set_status("Tile overlap percent must be a number.")
            return None
        if value <= 0 or value > 100:
            self._set_status("Tile overlap percent must be between 0 and 100.")
            return None
        return value

    def _parse_min_confidence(self) -> float | None:
        text = self.min_confidence_edit.text().strip()
        if not text:
            return 0.20
        try:
            value = float(text)
        except ValueError:
            self._set_status("Minimum confidence must be a number between 0 and 1.")
            return None
        if value < 0 or value > 1:
            self._set_status("Minimum confidence must be between 0 and 1.")
            return None
        return value

    def _parse_max_correction_px(self) -> float | None | bool:
        text = self.max_correction_px_edit.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            self._set_status("Max correction px must be a positive number, or blank to disable.")
            return False
        if value <= 0:
            self._set_status("Max correction px must be greater than 0, or blank to disable.")
            return False
        return value

    def _store_processing_settings(self) -> bool:
        if self.project is None:
            return True
        overlap_percent = self._parse_overlap_percent()
        if overlap_percent is None:
            return False
        self.project.metadata.extra_metadata["atlas_stitch_refinement_method"] = self._selected_alignment_method()
        self.project.metadata.extra_metadata["atlas_stitch_overlap_percent"] = overlap_percent
        self.project.metadata.extra_metadata["atlas_stitch_overlap_fraction"] = overlap_percent / 100.0
        min_confidence = self._parse_min_confidence()
        if min_confidence is None:
            return False
        max_correction_px = self._parse_max_correction_px()
        if max_correction_px is False:
            return False
        self.project.metadata.extra_metadata["atlas_stitch_min_confidence"] = min_confidence
        self.project.metadata.extra_metadata["atlas_stitch_max_correction_px"] = max_correction_px if max_correction_px is not None else ""
        self.project.metadata.extra_metadata["atlas_stitch_fusion_method"] = self._selected_fusion_method()
        return True

    def _update_alignment_method_ui(self) -> None:
        method = self._selected_alignment_method()
        method_label = _alignment_method_label(method)
        if method == ROBUST_ALIGNMENT_METHOD:
            self.estimate_alignment_button.setText("Run Auto-Registration")
            self.alignment_method_help.setText(
                "Robust translation tests multiple overlap widths before the global solve. "
                "Use this when light translation leaves visible seam errors."
            )
        elif method == ROBUST_OUTLIER_ALIGNMENT_METHOD:
            self.estimate_alignment_button.setText("Run Auto-Registration")
            self.alignment_method_help.setText(
                "Robust translation plus residual outlier rejection removes overlap pairs that disagree with the solved global layout. "
                "Use this when robust translation is close but a few bad pairs distort the mosaic."
            )
        elif method == FEATURE_MATCHING_METHOD:
            self.estimate_alignment_button.setText("Run Auto-Registration")
            self.alignment_method_help.setText(
                "Feature matching uses OpenCV SIFT when available, otherwise ORB, then estimates translation from matched overlap features. "
                "Use this when phase correlation struggles with sparse or uneven texture."
            )
        elif method == ECC_TRANSLATION_METHOD:
            self.estimate_alignment_button.setText("Run Auto-Registration")
            self.alignment_method_help.setText(
                "ECC translation uses OpenCV enhanced correlation alignment on overlap strips. "
                "Use this as an experimental alternative when feature matching is weak."
            )
        elif method == CENTER_OUT_CLOCKWISE_METHOD:
            self.estimate_alignment_button.setText("Run Auto-Registration")
            self.alignment_method_help.setText(
                "Center-out clockwise starts near the middle tile, attaches neighboring tiles outward in clockwise priority, "
                "and uses a spanning-tree constraint set. Use this when global registration is distorted by weak outer matches."
            )
        elif method == CENTER_OUT_COUNTERCLOCKWISE_METHOD:
            self.estimate_alignment_button.setText("Run Auto-Registration")
            self.alignment_method_help.setText(
                "Center-out counterclockwise starts near the middle tile, attaches neighboring tiles outward in counterclockwise priority, "
                "and uses a spanning-tree constraint set. Try this if clockwise chooses poor early neighbors."
            )
        else:
            self.estimate_alignment_button.setText("Run Auto-Registration")
            self.alignment_method_help.setText(
                "Light translation is the faster, conservative first-pass alignment method."
            )
        self.alignment_method_combo.setStatusTip(f"Selected alignment method: {method_label}")

    def _sync_alignment_method_from_project(self) -> None:
        if self.project is None:
            return
        extra = self.project.metadata.extra_metadata
        method = str(extra.get("atlas_stitch_refinement_method") or "").strip()
        if method:
            index = self.alignment_method_combo.findData(method)
            if index >= 0:
                self.alignment_method_combo.setCurrentIndex(index)
        overlap_percent = extra.get("atlas_stitch_overlap_percent")
        if overlap_percent in (None, ""):
            overlap_fraction = extra.get("atlas_stitch_overlap_fraction")
            if overlap_fraction not in (None, ""):
                try:
                    overlap_percent = float(overlap_fraction) * 100.0
                except (TypeError, ValueError):
                    overlap_percent = None
        if overlap_percent not in (None, ""):
            self.overlap_percent_edit.setText(_format_number(float(overlap_percent)))
        min_confidence = extra.get("atlas_stitch_min_confidence")
        if min_confidence not in (None, ""):
            self.min_confidence_edit.setText(_format_number(float(min_confidence)))
        max_correction_px = extra.get("atlas_stitch_max_correction_px")
        if max_correction_px not in (None, ""):
            self.max_correction_px_edit.setText(_format_number(float(max_correction_px)))
        fusion_method = str(extra.get("atlas_stitch_fusion_method") or "").strip()
        if fusion_method:
            fusion_index = self.fusion_method_combo.findData(fusion_method)
            if fusion_index >= 0:
                self.fusion_method_combo.setCurrentIndex(fusion_index)
        self._update_alignment_method_ui()

    def _populate_summary(self) -> None:
        if self.project is None:
            self.summary_text.clear()
            return
        self.summary_text.setPlainText(build_project_summary(self.project))

    def _populate_tile_table(self) -> None:
        project = self.project
        self.tile_table.setRowCount(0)
        if project is None:
            return

        self.tile_table.setRowCount(len(project.tiles))
        for row_index, tile in enumerate(project.tiles):
            nominal_x = tile.start_x if tile.start_x is not None else tile.transform.nominal_x
            nominal_y = tile.start_y if tile.start_y is not None else tile.transform.nominal_y
            refined_x = tile.transform.refined_x
            refined_y = tile.transform.refined_y
            manual_x = tile.transform.manual_x
            manual_y = tile.transform.manual_y
            file_name = Path(tile.file_name or tile.source_path or tile.resolved_path).name
            grid_position = _grid_position_label(tile)
            debug_tooltip = "\n".join(
                [
                    f"Grid Position: {grid_position}",
                    f"Tile ID: {tile.tile_id or '(none)'}",
                    f"Resolved Path: {tile.resolved_path or '(not available)'}",
                ]
            )
            values = [
                grid_position,
                file_name,
                "yes" if tile.exists else "no",
                _format_float(nominal_x),
                _format_float(nominal_y),
                _format_float(refined_x),
                _format_float(refined_y),
                _format_float(manual_x),
                _format_float(manual_y),
                tile.resolved_path,
                tile.tile_id,
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(debug_tooltip)
                if column_index == 2:
                    item.setTextAlignment(Qt.AlignCenter)
                self.tile_table.setItem(row_index, column_index, item)
        self.tile_table.resizeColumnsToContents()
        self.tile_table.setColumnHidden(9, True)
        self.tile_table.setColumnHidden(10, True)

    def _handle_tile_table_selection_changed(self) -> None:
        tiles = self._selected_tile_records()
        if not tiles:
            self._clear_preview_highlight("nominal")
            return

        focus_tile = tiles[-1]
        self._target_tile_id = focus_tile.tile_id
        self._open_selected_tiles(tiles, focus_tile=focus_tile)
        self._sync_selected_tile_to_preview()

    def _handle_tile_table_double_click(self, row: int, _column: int) -> None:
        tile = self._tile_record_for_row(row)
        if tile is None:
            return
        self._open_selected_tiles([tile], focus_tile=tile)
        self.on_tile_selected(tile.tile_id)
        self._focus_tile_view(tile)

    def _locate_grid_position(self) -> None:
        if self.project is None:
            self._set_status("Load an atlas project before locating a grid position.")
            return
        parsed = _parse_grid_position_text(self.grid_position_edit.text())
        if parsed is None:
            self._set_status("Enter grid position as r{row} c{col}, for example r8 c11.")
            return
        target_row, target_col = parsed
        for row_index, tile in enumerate(self.project.tiles):
            if tile.row == target_row and tile.col == target_col:
                self.tile_table.clearSelection()
                self.tile_table.selectRow(row_index)
                self.tile_table.setCurrentCell(row_index, 0)
                self.tile_table.scrollToItem(self.tile_table.item(row_index, 0))
                self._set_status(f"Located {_grid_position_label(tile)}.")
                return
        self._set_status(f"Grid position r{target_row} c{target_col} was not found.")

    def _sync_selected_tile_to_preview(self) -> None:
        tile_id = self._selected_tile_id()
        if not tile_id:
            self._clear_preview_highlight("nominal")
            return
        self.on_tile_selected(tile_id, center_camera=False)

    def on_tile_selected(self, tile_id: str, *, center_camera: bool = True) -> None:
        if self.viewer is None:
            return
        geometry = (
            self._preview_tile_geometries.get("manual", {}).get(tile_id)
            or self._preview_tile_geometries.get("nominal", {}).get(tile_id)
        )
        if geometry is None:
            return
        highlight_layer = self._preview_overlay_layers.get("nominal", {}).get("highlight")
        if highlight_layer is None:
            return
        highlight_layer.data = [geometry["shape"]]
        highlight_layer.properties = {"tile_id": np.asarray([tile_id], dtype=object)}
        highlight_layer.text = {"string": "{tile_id}", "color": "yellow", "size": 10, "anchor": "center"}
        highlight_layer.visible = True
        if center_camera:
            self._center_camera_on_point(geometry["center"])
        self._set_status(f"Highlighted tile {tile_id} in nominal preview.")

    def _selected_tile_id(self) -> str:
        tile = self._selected_tile_record()
        return tile.tile_id if tile is not None else ""

    def _selected_tile_record(self) -> TileRecord | None:
        row = self.tile_table.currentRow()
        return self._tile_record_for_row(row)

    def _selected_tile_records(self) -> list[TileRecord]:
        project = self.project
        selection_model = self.tile_table.selectionModel()
        if project is None or selection_model is None:
            return []
        rows = sorted(index.row() for index in selection_model.selectedRows())
        return [project.tiles[row] for row in rows if 0 <= row < len(project.tiles)]

    def _tile_record_for_row(self, row: int) -> TileRecord | None:
        project = self.project
        if project is None or row < 0 or row >= len(project.tiles):
            return None
        return project.tiles[row]
    def _worked_tile_ids(self) -> list[str]:
        if self.project is None:
            return []
        worked: list[str] = []
        for tile in self.project.tiles:
            if tile.transform.manual_x is not None and tile.transform.manual_y is not None:
                worked.append(tile.tile_id)
                continue
            if bool(tile.metadata.get("manual_worked")):
                worked.append(tile.tile_id)
        return worked


    def _layer_live_xy(self, layer) -> tuple[float, float]:
        """
        Return the current visible atlas-space (x, y) of the layer.

        Base placement comes from layer.translate.
        Interactive/manual movement comes from layer.affine.translate.
        The visible position is their sum.
        """
        base_translate = tuple(getattr(layer, "translate", (0.0, 0.0)))
        if len(base_translate) < 2:
            raise RuntimeError("Layer does not expose a usable translate vector.")

        base_y = float(base_translate[-2])
        base_x = float(base_translate[-1])

        affine_y = 0.0
        affine_x = 0.0
        try:
            affine_translate = tuple(layer.affine.translate)
            if len(affine_translate) >= 2:
                affine_y = float(affine_translate[-2])
                affine_x = float(affine_translate[-1])
        except Exception:
            pass

        return base_x + affine_x, base_y + affine_y

    def _set_layer_live_xy(self, layer, x: float, y: float) -> None:
        """
        Set the current visible atlas-space (x, y) of the layer.

        Keep layer.translate as the base anchor.
        Store the live/manual offset only in layer.affine.translate.
        """
        base_translate = tuple(getattr(layer, "translate", (0.0, 0.0)))
        if len(base_translate) < 2:
            raise RuntimeError("Layer does not expose a usable translate vector.")

        base_y = float(base_translate[-2])
        base_x = float(base_translate[-1])

        delta_x = float(x) - base_x
        delta_y = float(y) - base_y

        try:
            affine = layer.affine
            affine_translate = list(tuple(affine.translate))
            if len(affine_translate) < 2:
                raise RuntimeError("Layer affine transform does not expose a usable translate vector.")
            affine_translate[-2] = delta_y
            affine_translate[-1] = delta_x
            affine.translate = tuple(affine_translate)
        except Exception as exc:
            raise RuntimeError(f"Could not set live layer position: {exc}") from exc

    def _reset_layer_affine_translation(self, layer) -> None:
        try:
            affine = layer.affine
            affine_translate = list(tuple(affine.translate))
            if len(affine_translate) >= 2:
                affine_translate[-2] = 0.0
                affine_translate[-1] = 0.0
                affine.translate = tuple(affine_translate)
        except Exception:
            pass


    def _manual_position_for_tile(self, tile: TileRecord) -> tuple[float, float] | None:
        if tile.transform.manual_x is None or tile.transform.manual_y is None:
            return None
        return float(tile.transform.manual_x), float(tile.transform.manual_y)

    def _activate_tile_target(
        self,
        tile_id: str,
        *,
        sync_table: bool = False,
        center_camera: bool = False,
    ) -> None:
        tile = self._tile_by_id(tile_id)
        if tile is None:
            return

        self._target_tile_id = tile_id
        self._activate_opened_tile_layer(tile_id)

        if sync_table:
            self._select_tile_row_safely(tile_id)

        self.on_tile_selected(tile_id, center_camera=center_camera)

    
    def _layer_current_xy(self, layer) -> tuple[float, float]:
        """
        Backward-compatible wrapper.
        Return current atlas-space X/Y for an opened tile layer.
        Returns (x, y).
        """
        return self._layer_live_xy(layer)

    def _save_current_manual_positions(self) -> None:
        if self.project is None or self.viewer is None:
            
            self._set_status("Load a project and open at least one tile before saving manual positions.")
            return
        
        selected_tiles = self._selected_tile_records()
        if not selected_tiles:
            selected_tiles = [
                tile for tile in self.project.tiles
                if tile.tile_id in self._opened_tile_layers
            ]

        if not selected_tiles:
            self._set_status("Select one or more opened tiles before saving manual positions.")
            return

        saved_ids: list[str] = []
        missing_ids: list[str] = []

        for tile in selected_tiles:
            if bool(tile.metadata.get("manual_locked")) and (
                tile.transform.manual_x is not None and tile.transform.manual_y is not None
            ):
                missing_ids.append(f"{tile.tile_id} (locked)")
                continue

            layer = self._opened_tile_layers.get(tile.tile_id)
            if layer is None:
                missing_ids.append(tile.tile_id)
                continue

            try:
                manual_x, manual_y = self._layer_live_xy(layer)
            except Exception:
                self._set_status(f"Could not read current layer position for {tile.tile_id}.")
                return

            tile.transform.manual_x = float(manual_x)
            tile.transform.manual_y = float(manual_y)
            tile.metadata["manual_worked"] = True
            tile.metadata["manual_saved_time"] = datetime.now().astimezone().isoformat(timespec="seconds")
            saved_ids.append(tile.tile_id)

        if self.project is not None:
            self.project.metadata.extra_metadata["atlas_stitch_manual_tile_ids"] = self._worked_tile_ids()

        self._populate_summary()
        self._populate_tile_table()
        self._update_refinement_controls()
        self._update_opened_tile_bounds_overlay()
        

        if saved_ids:
            self._set_status(f"Saved manual position for: {', '.join(saved_ids)}")
        elif missing_ids:
            self._set_status(f"No opened layers found for selected tile(s): {', '.join(missing_ids)}")
        else:
            self._set_status("No manual positions were saved.")

    def _open_selected_tiles(self, tiles: list[TileRecord], focus_tile: TileRecord | None = None) -> None:
        if self.viewer is None:
            return
        opened_ids: list[str] = []
        zero_ids: list[str] = []
        skipped_ids: list[str] = []
        focus = focus_tile or (tiles[-1] if tiles else None)

        for tile in tiles:
            if tile.tile_id in self._opened_tile_layers:
                continue
            data, problem = self._load_openable_tile_data(tile)
            if problem == "zero":
                zero_ids.append(tile.tile_id)
                continue
            if data is None:
                skipped_ids.append(tile.tile_id)
                continue
            translate = _tile_translate(tile, "manual")
            layer = self.viewer.add_image(
                data,
                name=f"Atlas Tile {tile.tile_id}",
                translate=translate,
                blending="additive",
            )
            self._reset_layer_affine_translation(layer)
            try:
                finite = np.asarray(data)[np.isfinite(data)]
                if finite.size:
                    lower = float(np.percentile(finite, 1.0))
                    upper = float(np.percentile(finite, 99.0))
                    if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
                        layer.contrast_limits = (lower, upper)
            except Exception:
                pass
            self._opened_tile_layers[tile.tile_id] = layer
            self._opened_tile_order.append(tile.tile_id)
            opened_ids.append(tile.tile_id)

        self._update_opened_tile_bounds_overlay()
        self._set_status(_format_open_tiles_status(opened_ids, zero_ids, skipped_ids, focus.tile_id if focus else ""))
        self._update_refinement_controls()

    def _load_openable_tile_data(self, tile: TileRecord) -> tuple[np.ndarray | None, str | None]:
        resolved_path = str(tile.resolved_path or "").strip()
        if not resolved_path:
            return None, "missing_path"
        path = Path(resolved_path).expanduser()
        if not path.exists():
            return None, "missing_file"
        try:
            data = imread(path)
        except Exception:
            return None, "read_error"
        if data.ndim > 2:
            data = data[0]
        if np.all(data == 0):
            return None, "zero"
        return data, None

    def _focus_tile_view(self, tile: TileRecord) -> None:
        if self.viewer is None:
            return
        opened_layer = self._opened_tile_layers.get(tile.tile_id)
        if opened_layer is not None:
            self._focus_raw_tile_layer(opened_layer)
            return
        geometry = self._preview_tile_geometries.get("nominal", {}).get(tile.tile_id)
        if geometry is not None:
            self._center_camera_on_point(geometry["center"])
            self._zoom_to_tile(tile)
            return
        
        translate = _tile_translate(tile, "manual")

        tile_width, tile_height = _tile_shape(tile)
        center_y = float(translate[0]) + (float(tile_height) / 2.0)
        center_x = float(translate[1]) + (float(tile_width) / 2.0)
        self._center_camera_on_point((center_y, center_x))
        self._zoom_to_tile(tile)

    def _focus_layer(self, layer: Any) -> None:
        if self.viewer is None or layer is None:
            return
        try:
            layer.visible = True
        except Exception:
            pass
        try:
            self.viewer.layers.selection.active = layer
        except Exception:
            pass
        try:
            previous_visibility = {existing_layer: bool(existing_layer.visible) for existing_layer in self.viewer.layers}
            for existing_layer in self.viewer.layers:
                existing_layer.visible = existing_layer is layer
            self.viewer.reset_view()
            for existing_layer, was_visible in previous_visibility.items():
                existing_layer.visible = was_visible
            layer.visible = True
            return
        except Exception:
            pass
        try:
            data_shape = np.asarray(layer.data).shape[:2]
            translate = getattr(layer, "translate", (0.0, 0.0))
            scale = getattr(layer, "scale", (1.0, 1.0))
            center_y = float(translate[-2]) + (float(data_shape[0]) * float(scale[-2]) / 2.0)
            center_x = float(translate[-1]) + (float(data_shape[1]) * float(scale[-1]) / 2.0)
            size_y = max(1.0, float(data_shape[0]) * float(scale[-2]))
            size_x = max(1.0, float(data_shape[1]) * float(scale[-1]))
            self._center_camera_on_point((center_y, center_x))
            self._zoom_to_extent(size_y=size_y, size_x=size_x)
            return
        except Exception:
            pass
        try:
            extent = np.asarray(layer.extent.world, dtype=float)
            displayed = list(self.viewer.dims.displayed)
            if extent.shape[0] >= 2 and len(displayed) >= 2:
                y_index = displayed[-2]
                x_index = displayed[-1]
                center_y = float((extent[0, y_index] + extent[1, y_index]) / 2.0)
                center_x = float((extent[0, x_index] + extent[1, x_index]) / 2.0)
                size_y = max(1.0, float(extent[1, y_index] - extent[0, y_index]))
                size_x = max(1.0, float(extent[1, x_index] - extent[0, x_index]))
                self._center_camera_on_point((center_y, center_x))
                self._zoom_to_extent(size_y=size_y, size_x=size_x)
                return
        except Exception:
            pass
        if hasattr(layer, "data"):
            self._zoom_to_tile_from_shape(np.asarray(layer.data).shape[:2])

    def _focus_raw_tile_layer(self, layer: Any) -> None:
        if self.viewer is None or layer is None:
            return
        self._focus_layer(layer)

    def _clear_opened_tile_layers(self) -> None:
        if self.viewer is None:
            self._opened_tile_layers.clear()
            self._opened_tile_order.clear()
            self._opened_tile_bounds_layer = None
            return
        for tile_id in list(self._opened_tile_order):
            layer = self._opened_tile_layers.pop(tile_id, None)
            if layer is None:
                continue
            try:
                self.viewer.layers.remove(layer)
            except Exception:
                pass
        self._opened_tile_order.clear()
        if self._opened_tile_bounds_layer is not None:
            try:
                self.viewer.layers.remove(self._opened_tile_bounds_layer)
            except Exception:
                pass
            self._opened_tile_bounds_layer = None

    def _update_opened_tile_bounds_overlay(self) -> None:
        if self.viewer is None:
            self._opened_tile_bounds_layer = None
            return

        if self.project is None or not self._opened_tile_order:
            if self._opened_tile_bounds_layer is not None:
                try:
                    self._opened_tile_bounds_layer.data = []
                    self._opened_tile_bounds_layer.properties = {"tile_id": np.asarray([], dtype=object)}
                    self._opened_tile_bounds_layer.visible = False
                except Exception:
                    pass
            return

        shapes_data: list[np.ndarray] = []
        tile_ids: list[str] = []

        for tile_id in self._opened_tile_order:
            shape = self._opened_tile_bounds_shape_from_layer(tile_id)
            if shape is None:
                continue
            shapes_data.append(shape)
            tile_ids.append(tile_id)

        if self._opened_tile_bounds_layer is None:
            layer = self.viewer.add_shapes(
                shapes_data,
                shape_type="polygon",
                name=self._opened_tile_bounds_name,
                edge_color="yellow",
                edge_width=2.5,
                face_color="transparent",
                opacity=1.0,
                blending="translucent_no_depth",
                properties={"tile_id": np.asarray(tile_ids, dtype=object)},
                text={"string": "{tile_id}", "color": "yellow", "size": 10, "anchor": "center"},
            )
            try:
                layer.editable = False
            except Exception:
                pass
            try:
                layer.mode = "pan_zoom"
            except Exception:
                pass
            self._opened_tile_bounds_layer = layer
            return

        try:
            self._opened_tile_bounds_layer.data = shapes_data
            self._opened_tile_bounds_layer.properties = {"tile_id": np.asarray(tile_ids, dtype=object)}
            self._opened_tile_bounds_layer.text = {
                "string": "{tile_id}",
                "color": "yellow",
                "size": 10,
                "anchor": "center",
            }
            self._opened_tile_bounds_layer.visible = bool(shapes_data)
        except Exception:
            # fallback: recreate only if in-place update fails
            try:
                self.viewer.layers.remove(self._opened_tile_bounds_layer)
            except Exception:
                pass
            self._opened_tile_bounds_layer = None
            self._update_opened_tile_bounds_overlay()

    def _refresh_repair_tile_options(self) -> None:
        current_target = str(self.repair_target_combo.currentData() or "")
        self.repair_target_combo.clear()
        if self.project is None:
            return
        for tile in self.project.tiles:
            label = f"{_grid_position_label(tile)}"
            file_name = Path(tile.file_name or tile.source_path or tile.resolved_path).name
            if file_name:
                label = f"{label} | {file_name}"
            self.repair_target_combo.addItem(label, tile.tile_id)
        if current_target:
            index = self.repair_target_combo.findData(current_target)
            if index >= 0:
                self.repair_target_combo.setCurrentIndex(index)

    def _update_repair_controls(self) -> None:
        has_project = self.project is not None
        repair_running = bool(self._repair_thread and self._repair_thread.isRunning())
        self.repair_target_combo.setEnabled(has_project and not repair_running)
        self.repair_donor_text.setEnabled(has_project and not repair_running)
        self.repair_overlap_width_edit.setEnabled(has_project and not repair_running)
        self.repair_mode_combo.setEnabled(has_project and not repair_running)
        self.repair_blend_combo.setEnabled(has_project and not repair_running)
        self.repair_output_folder_edit.setEnabled(has_project and not repair_running)
        self.repair_output_browse_button.setEnabled(has_project and not repair_running)
        self.repair_start_button.setEnabled(has_project and not repair_running)
        self.show_overlap_preview_button.setEnabled(has_project and not repair_running)
        self.preview_repair_button.setEnabled(has_project and not repair_running)
        self.apply_repair_button.setEnabled(has_project and not repair_running)
        self.cancel_repair_button.setEnabled(has_project)
        self.clear_repair_preview_button.setEnabled(has_project)

    def _enter_repair_mode(self) -> None:
        if self.project is None or self.viewer is None:
            self._set_status("Load an atlas project before entering seam repair.")
            return
        self._refresh_repair_tile_options()
        if not self.repair_output_folder_edit.text().strip():
            default_root = (
                Path(self.project.metadata.xml_path).expanduser().parent
                if self.project.metadata.xml_path
                else Path.cwd()
            )
            self.repair_output_folder_edit.setText(str(default_root / "atlas_seam_repairs"))
        self._ensure_repair_roi_layer()
        self._set_status("Repair mode active. Configure target and donors, then draw one rectangular ROI if needed.")

    def _ensure_repair_roi_layer(self) -> None:
        if self.viewer is None:
            return
        if self._repair_roi_layer is not None and self._repair_roi_layer in self.viewer.layers:
            return
        self._repair_roi_layer = self.viewer.add_shapes(
            [],
            shape_type="rectangle",
            name="Atlas Repair ROI",
            edge_color="yellow",
            edge_width=2.0,
            face_color="transparent",
            opacity=1.0,
            blending="translucent_no_depth",
        )
        self._repair_roi_layer.mode = "add_rectangle"

    def _show_repair_overlap_preview(self) -> None:
        request = self._build_repair_request()
        if request is None:
            return
        target_tile = self._repair_target_tile(request.target_tile_id)
        if target_tile is None:
            self._set_status("Select a valid repair target tile.")
            return
        donor_tiles = [tile for tile in (_tile for _tile in (self._repair_target_tile(spec.tile_id) for spec in request.donors)) if tile is not None]
        self._open_selected_tiles([target_tile] + donor_tiles, focus_tile=target_tile)
        self._ensure_repair_roi_layer()
        self._clear_repair_overlap_layer()
        overlap_shapes = [_tile_bounds_shape(target_tile, "nominal")]
        overlap_names = [target_tile.tile_id]
        for donor_spec in request.donors:
            donor_tile = self._repair_target_tile(donor_spec.tile_id)
            if donor_tile is None:
                continue
            overlap_shapes.append(_tile_bounds_shape(donor_tile, "nominal"))
            overlap_names.append(f"{donor_tile.tile_id} ({donor_spec.direction})")
        if self.viewer is not None and overlap_shapes:
            self._repair_overlap_layer = self.viewer.add_shapes(
                overlap_shapes,
                shape_type="polygon",
                name="Atlas Repair Overlap",
                edge_color="magenta",
                edge_width=2.0,
                face_color="transparent",
                opacity=1.0,
                blending="translucent_no_depth",
                properties={"tile_id": np.asarray(overlap_names, dtype=object)},
                text={"string": "{tile_id}", "color": "yellow", "size": 10, "anchor": "center"},
            )
        self._set_status("Overlap preview shown. Adjust ROI if using ROI-guided repair, then click Preview Repair.")

    def _preview_repair(self) -> None:
        request = self._build_repair_request()
        if request is None or self.project is None:
            return
        self._repair_preview_result = None
        self._clear_repair_preview_layers()
        self._start_repair_worker(request, apply_changes=False)

    def _apply_repair(self) -> None:
        request = self._build_repair_request()
        if request is None or self.project is None:
            return
        output_dir = self.repair_output_folder_edit.text().strip()
        if not output_dir:
            self._set_status("Select a repair output folder before applying seam repair.")
            return
        self._start_repair_worker(request, apply_changes=True)

    def _cancel_repair(self) -> None:
        self._clear_repair_preview()
        self._clear_repair_overlap_layer()
        self._set_status("Repair preview cleared.")

    def _clear_repair_preview(self) -> None:
        self._repair_preview_result = None
        self._clear_repair_preview_layers()

    def _build_repair_request(self) -> TileRepairRequest | None:
        if self.project is None:
            self._set_status("Load an atlas project before repairing tiles.")
            return None
        target_tile_id = str(self.repair_target_combo.currentData() or "").strip()
        if not target_tile_id:
            self._set_status("Select a repair target tile.")
            return None
        try:
            donors = self._parse_repair_donors()
            overlap_width = self._parse_repair_overlap_width()
        except ValueError as exc:
            self._set_status(str(exc))
            return None
        repair_mode = str(self.repair_mode_combo.currentData() or REPAIR_MODE_FULL_OVERLAP)
        roi_bounds = self._repair_roi_bounds_for_target(target_tile_id)
        if repair_mode == REPAIR_MODE_ROI_GUIDED and roi_bounds is None:
            self._set_status("ROI-guided repair requires one rectangular ROI in the Atlas Repair ROI layer.")
            return None
        return TileRepairRequest(
            target_tile_id=target_tile_id,
            donors=donors,
            overlap_width=overlap_width,
            repair_mode=repair_mode,
            blend_mode=str(self.repair_blend_combo.currentData() or BLEND_MODE_HARD_REPLACE),
            roi_bounds=roi_bounds,
        )

    def _parse_repair_donors(self) -> list[RepairDonorSpec]:
        text = self.repair_donor_text.toPlainText().strip()
        if not text:
            raise ValueError("Enter one donor per line as tile_id,direction.")
        donors: list[RepairDonorSpec] = []
        for priority, raw_line in enumerate(text.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",") if part.strip()]
            if len(parts) < 2:
                raise ValueError(f"Invalid donor line: {line}. Expected tile_id,direction")
            tile_id, direction = parts[0], parts[1].lower()
            if direction not in DONOR_DIRECTIONS:
                raise ValueError(f"Unsupported donor direction: {direction}")
            donors.append(RepairDonorSpec(tile_id=tile_id, direction=direction, priority=priority))
        if not donors:
            raise ValueError("Enter at least one donor tile.")
        return donors

    def _parse_repair_overlap_width(self) -> int:
        text = self.repair_overlap_width_edit.text().strip()
        if not text:
            return DEFAULT_REPAIR_OVERLAP
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError("Repair overlap width must be a positive integer.") from exc
        if value < 1:
            raise ValueError("Repair overlap width must be at least 1.")
        return value

    def _repair_roi_bounds_for_target(self, target_tile_id: str) -> tuple[float, float, float, float] | None:
        if self._repair_roi_layer is None or len(self._repair_roi_layer.data) == 0:
            return None
        target_tile = self._repair_target_tile(target_tile_id)
        if target_tile is None:
            return None
        shape = np.asarray(self._repair_roi_layer.data[-1], dtype=float)
        y0 = float(np.min(shape[:, 0]))
        x0 = float(np.min(shape[:, 1]))
        y1 = float(np.max(shape[:, 0]))
        x1 = float(np.max(shape[:, 1]))
        translate_y, translate_x = _tile_translate(target_tile, "nominal")
        return y0 - translate_y, x0 - translate_x, y1 - translate_y, x1 - translate_x

    def _repair_target_tile(self, tile_id: str) -> TileRecord | None:
        if self.project is None:
            return None
        for tile in self.project.tiles:
            if tile.tile_id == tile_id:
                return tile
        return None

    def _start_repair_worker(self, request: TileRepairRequest, *, apply_changes: bool) -> None:
        if self.project is None:
            return
        self._cleanup_repair_worker()
        thread = QThread()
        worker = RepairWorker(
            self.project,
            request,
            output_dir=self.repair_output_folder_edit.text().strip(),
            apply_changes=apply_changes,
            repair_mode=request.repair_mode,
            blend_mode=request.blend_mode,
            overlap_width=request.overlap_width,
        )
        worker.moveToThread(thread)
        worker.progress.connect(self._handle_repair_progress)
        worker.preview_ready.connect(self._handle_repair_preview_ready)
        worker.applied.connect(self._handle_repair_applied)
        worker.error.connect(self._handle_repair_error)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._cleanup_repair_worker)
        thread.finished.connect(self._update_repair_controls)
        thread.started.connect(worker.run)
        self._repair_worker = worker
        self._repair_thread = thread
        self._update_repair_controls()
        self.progress_label.setText("Repair: queued")
        self._set_progress_busy()
        thread.start()

    def _handle_repair_progress(self, message: str, current: int, total: int) -> None:
        self.progress_label.setText(message)
        self._update_progress_bar(current, total)

    def _handle_repair_preview_ready(self, result: TileRepairResult) -> None:
        self._repair_preview_result = result
        self._show_repair_result_layers(result)
        self.progress_label.setText("Repair preview ready")
        self._set_progress_complete()
        self._set_status(f"Repair preview ready for {result.target_tile_id}.")

    def _handle_repair_applied(self, result: TileRepairResult, saved: dict[str, Any]) -> None:
        self._repair_preview_result = result
        target_tile = self._repair_target_tile(result.target_tile_id)
        if target_tile is not None:
            target_tile.repaired_path = str(saved.get("repaired_path") or "")
            target_tile.repair_confidence_path = str(saved.get("confidence_path") or "")
            target_tile.repair_attribution_path = str(saved.get("attribution_path") or "")
            target_tile.repair_history.append(dict(saved.get("history_entry") or {}))
            target_tile.metadata["atlas_stitch_repair_status"] = "applied"
        self._show_repair_result_layers(result)
        self._populate_summary()
        self.progress_label.setText("Repair applied")
        self._set_progress_complete()
        self._set_status(f"Applied seam repair for {result.target_tile_id}.")
        if self.project is not None and "nominal" in self._preview_layers and not (self._preview_thread and self._preview_thread.isRunning()):
            self._preview_layout("nominal")

    def _handle_repair_error(self, message: str) -> None:
        self.progress_label.setText(f"Repair failed: {message}")
        self._reset_progress(f"Repair failed: {message}")
        self._set_status(f"Repair failed: {message}")

    def _show_repair_result_layers(self, result: TileRepairResult) -> None:
        if self.viewer is None:
            return
        self._clear_repair_preview_layers()
        target_tile = self._repair_target_tile(result.target_tile_id)
        if target_tile is None:
            return
        translate = _tile_translate(target_tile, "nominal")
        self._repair_preview_layers["repaired"] = self.viewer.add_image(
            result.repaired_tile,
            name=f"Atlas Repair Preview ({result.target_tile_id})",
            translate=translate,
            blending="additive",
        )
        self._repair_preview_layers["confidence"] = self.viewer.add_image(
            result.confidence_map.astype(np.float32, copy=False),
            name=f"Atlas Repair Confidence ({result.target_tile_id})",
            translate=translate,
            blending="translucent",
            colormap="viridis",
            opacity=0.7,
        )
        self._repair_preview_layers["attribution"] = self.viewer.add_labels(
            result.attribution_map.astype(np.uint16, copy=False),
            name=f"Atlas Repair Attribution ({result.target_tile_id})",
            translate=translate,
            opacity=0.4,
        )
        self._focus_tile_view(target_tile)

    def _clear_repair_preview_layers(self) -> None:
        if self.viewer is None:
            self._repair_preview_layers.clear()
            return
        for layer in list(self._repair_preview_layers.values()):
            try:
                self.viewer.layers.remove(layer)
            except Exception:
                pass
        self._repair_preview_layers.clear()

    def _clear_repair_overlap_layer(self) -> None:
        if self.viewer is None:
            self._repair_overlap_layer = None
            return
        layer = self._repair_overlap_layer
        self._repair_overlap_layer = None
        if layer is None:
            return
        try:
            self.viewer.layers.remove(layer)
        except Exception:
            pass

    def _status_message(self) -> str:
        project = self.project
        if project is None:
            return "No atlas project loaded."
        if project.missing_tiles:
            return f"Loaded {len(project.tiles)} tile(s). Missing files detected for {len(project.missing_tiles)} tile(s)."
        return f"Loaded {len(project.tiles)} tile(s). All resolved tile files were found."

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _reset_progress(self, message: str) -> None:
        self.progress_label.setText(message)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(message)

    def _set_progress_busy(self) -> None:
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("")

    def _set_progress_complete(self) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_bar.setFormat("Complete")

    def _update_progress_bar(self, current: int, total: int) -> None:
        if current < 0 or total <= 0:
            self._set_progress_busy()
            return
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(max(0, min(current, total)))
        self.progress_bar.setFormat("%v / %m")

    def _clear_preview_layers(self) -> None:
        if self.viewer is None:
            self._preview_layers.clear()
            self._preview_overlay_layers.clear()
            self._preview_tile_geometries.clear()
            self._preview_downsamples.clear()
            return
        for layer in list(self._preview_layers.values()):
            try:
                self.viewer.layers.remove(layer)
            except Exception:
                pass
        for overlay_layers in list(self._preview_overlay_layers.values()):
            for layer in overlay_layers.values():
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
        self._preview_layers.clear()
        self._preview_overlay_layers.clear()
        self._preview_tile_geometries.clear()
        self._preview_downsamples.clear()

    def _remove_preview_layer(self, placement_mode: str) -> None:
        if self.viewer is None:
            self._preview_layers.pop(placement_mode, None)
            self._preview_overlay_layers.pop(placement_mode, None)
            self._preview_tile_geometries.pop(placement_mode, None)
            return
        layer = self._preview_layers.pop(placement_mode, None)
        if layer is None:
            overlay_layers = self._preview_overlay_layers.pop(placement_mode, {})
            for overlay_layer in overlay_layers.values():
                try:
                    self.viewer.layers.remove(overlay_layer)
                except Exception:
                    pass
            self._preview_tile_geometries.pop(placement_mode, None)
            self._preview_downsamples.pop(placement_mode, None)
            return
        try:
            self.viewer.layers.remove(layer)
        except Exception:
            pass
        overlay_layers = self._preview_overlay_layers.pop(placement_mode, {})
        for overlay_layer in overlay_layers.values():
            try:
                self.viewer.layers.remove(overlay_layer)
            except Exception:
                pass
        self._preview_tile_geometries.pop(placement_mode, None)
        self._preview_downsamples.pop(placement_mode, None)

    def _preview_layer_name(self, placement_mode: str) -> str:
        mode_label = _placement_mode_label(placement_mode)
        return f"{self._preview_prefix} ({mode_label})"

    def _render_preview_overlays(self, placement_mode: str) -> None:
        if self.viewer is None or self.project is None:
            return
        image_layer = self._preview_layers.get(placement_mode)
        downsample = self._preview_downsamples.get(placement_mode)
        if image_layer is None or downsample is None:
            return

        overlay_layers = self._preview_overlay_layers.pop(placement_mode, {})
        for layer in overlay_layers.values():
            try:
                self.viewer.layers.remove(layer)
            except Exception:
                pass

        shapes_data: list[np.ndarray] = []
        tile_ids: list[str] = []
        tile_geometries: dict[str, dict[str, Any]] = {}
        for tile in self.project.tiles:
            preview_position = _preview_position(tile, placement_mode)
            if preview_position is None:
                continue
            tile_width, tile_height = _tile_shape(tile)
            tile_x, tile_y = preview_position
            scaled_x0 = float(tile_x) / downsample
            scaled_y0 = float(tile_y) / downsample
            scaled_x1 = float(tile_x + tile_width) / downsample
            scaled_y1 = float(tile_y + tile_height) / downsample
            rectangle = np.asarray(
                [
                    [scaled_y0, scaled_x0],
                    [scaled_y1, scaled_x0],
                    [scaled_y1, scaled_x1],
                    [scaled_y0, scaled_x1],
                ],
                dtype=float,
            )
            translate_y, translate_x = image_layer.translate[-2], image_layer.translate[-1]
            center = (
                float((scaled_y0 + scaled_y1) / 2.0 + translate_y),
                float((scaled_x0 + scaled_x1) / 2.0 + translate_x),
            )
            shapes_data.append(rectangle)
            tile_ids.append(tile.tile_id)
            tile_geometries[tile.tile_id] = {"shape": rectangle, "center": center}

        footprints_layer = self.viewer.add_shapes(
            shapes_data,
            shape_type="polygon",
            name=f"{self._preview_layer_name(placement_mode)} Footprints",
            edge_color="cyan",
            edge_width=1.5,
            face_color="transparent",
            opacity=1.0,
            blending="translucent_no_depth",
            translate=image_layer.translate,
            properties={"tile_id": np.asarray(tile_ids, dtype=object)},
            text={"string": "{tile_id}", "color": "yellow", "size": 8, "anchor": "center"},
        )
        highlight_layer = self.viewer.add_shapes(
            [],
            shape_type="polygon",
            name=f"{self._preview_layer_name(placement_mode)} Highlight",
            edge_color="yellow",
            edge_width=4,
            face_color="transparent",
            opacity=1.0,
            blending="translucent_no_depth",
            translate=image_layer.translate,
            properties={"tile_id": np.asarray([], dtype=object)},
            text={"string": "{tile_id}", "color": "yellow", "size": 10, "anchor": "center"},
        )
        highlight_layer.visible = False
        self._preview_overlay_layers[placement_mode] = {"footprints": footprints_layer, "highlight": highlight_layer}
        self._preview_tile_geometries[placement_mode] = tile_geometries

    def _clear_preview_highlight(self, placement_mode: str) -> None:
        highlight_layer = self._preview_overlay_layers.get(placement_mode, {}).get("highlight")
        if highlight_layer is None:
            return
        highlight_layer.data = []
        highlight_layer.properties = {"tile_id": np.asarray([], dtype=object)}
        highlight_layer.visible = False

    def _center_camera_on_point(self, point: tuple[float, float]) -> None:
        if self.viewer is None:
            return
        center = list(self.viewer.camera.center)
        displayed = list(self.viewer.dims.displayed)
        if len(center) < 2 or len(displayed) < 2:
            self.viewer.camera.center = point
            return
        center[displayed[-2]] = float(point[0])
        center[displayed[-1]] = float(point[1])
        self.viewer.camera.center = tuple(center)

    def _zoom_to_tile(self, tile: TileRecord) -> None:
        self._zoom_to_tile_from_shape(_tile_shape(tile))

    def _zoom_to_tile_from_shape(self, shape: tuple[int, int]) -> None:
        if self.viewer is None:
            return
        tile_width, tile_height = shape
        self._zoom_to_extent(size_y=float(tile_height), size_x=float(tile_width))

    def _zoom_to_extent(self, *, size_y: float, size_x: float) -> None:
        if self.viewer is None:
            return
        viewport_height = 900.0
        viewport_width = 900.0
        try:
            canvas_size = self.viewer.window.qt_viewer.canvas.size
            viewport_width = max(float(canvas_size[0]), 1.0)
            viewport_height = max(float(canvas_size[1]), 1.0)
        except Exception:
            pass
        target_zoom = 0.85 * min(viewport_height / max(size_y, 1.0), viewport_width / max(size_x, 1.0))
        self.viewer.camera.zoom = max(target_zoom, 0.05)

    def _open_export(self) -> None:
        if self.viewer is None or self.project is None:
            self._set_status("Napari viewer required to open exported OME-Zarr.")
            return
        export_path = str(self.project.last_export.path or "").strip()
        if not export_path:
            self._set_status("No exported OME-Zarr path is available.")
            return
        path = Path(export_path).expanduser()
        if not path.exists():
            self._set_status("Last exported OME-Zarr path was not found.")
            return
        try:
            loaded = self.viewer.open([str(path)], stack=False)
        except Exception as exc:
            self._set_status(f"Failed to open exported OME-Zarr: {exc}")
            return
        self._set_status(
            f"Opened exported OME-Zarr: {path}" if loaded else f"Open requested for exported OME-Zarr: {path}"
        )

    def _update_open_export_enabled(self) -> None:
        if self.project is None:
            self.open_export_button.setEnabled(False)
            return
        export_path = str(self.project.last_export.path or "").strip()
        self.open_export_button.setEnabled(bool(export_path and Path(export_path).expanduser().exists()))

    def _is_preview_running(self) -> bool:
        return bool(self._preview_thread and self._preview_thread.isRunning())

    def _is_alignment_running(self) -> bool:
        return bool(self._alignment_thread and self._alignment_thread.isRunning())

    def _is_export_running(self) -> bool:
        return bool(self._export_thread and self._export_thread.isRunning())

    def _has_refined_positions(self) -> bool:
        return bool(
            self.project is not None
            and any(tile.transform.refined_x is not None and tile.transform.refined_y is not None for tile in self.project.tiles)
        )

    def _has_manual_positions(self) -> bool:
        return bool(
            self.project is not None
            and any(tile.transform.manual_x is not None and tile.transform.manual_y is not None for tile in self.project.tiles)
        )

    def _has_exportable_tiles(self) -> bool:
        return bool(
            self.project is not None
            and any(tile.exists and str(tile.resolved_path or preferred_tile_path(tile) or "").strip() for tile in self.project.tiles)
        )

    def _has_usable_neighboring_tiles(self) -> bool:
        if self.project is None:
            return False
        usable_by_grid: dict[tuple[int, int], TileRecord] = {}
        for tile in self.project.tiles:
            if tile.row is None or tile.col is None:
                continue
            if not (tile.exists and str(tile.resolved_path or preferred_tile_path(tile) or "").strip()):
                continue
            usable_by_grid[(int(tile.row), int(tile.col))] = tile
        for row, col in usable_by_grid:
            if (row, col + 1) in usable_by_grid or (row + 1, col) in usable_by_grid:
                return True
        return False

    def _update_refinement_controls(self) -> None:
        project = self.project
        has_project = project is not None
        preview_running = self._is_preview_running()
        alignment_running = self._is_alignment_running()
        export_running = self._is_export_running()
        has_refined = self._has_refined_positions()
        has_manual = self._has_manual_positions()
        has_exportable_tiles = self._has_exportable_tiles()
        has_usable_neighbors = self._has_usable_neighboring_tiles()

        self.load_button.setEnabled(not alignment_running and not export_running)
        self.load_project_button.setEnabled(not alignment_running and not export_running)
        self.save_project_button.setEnabled(has_project and not alignment_running and not export_running)

        self.preview_nominal_button.setEnabled(has_project and not preview_running)
        self.clear_preview_button.setEnabled(has_project and not preview_running)
        self.grid_position_button.setEnabled(has_project)
        self.grid_position_edit.setEnabled(has_project)

        self.estimate_alignment_button.setEnabled(
            has_project and has_usable_neighbors and not alignment_running and not preview_running and not export_running
        )
        self.preview_refined_button.setEnabled(
            has_refined and not preview_running and not alignment_running
        )
        self.preview_manual_button.setEnabled(
            has_manual and not preview_running
        )
        self.save_manual_position_button.setEnabled(
            has_project and bool(self._opened_tile_layers) and not preview_running and not alignment_running
        )
        self.export_button.setEnabled(has_project and has_exportable_tiles and not alignment_running and not export_running)
        self._update_open_export_enabled()
        self._update_repair_controls()

    def _cleanup_preview_worker(self) -> None:
        if self._preview_worker is not None:
            self._preview_worker.deleteLater()
            self._preview_worker = None
        if self._preview_thread is not None:
            self._preview_thread.wait()
            self._preview_thread.deleteLater()
            self._preview_thread = None

    def _cleanup_export_worker(self) -> None:
        if self._export_worker is not None:
            self._export_worker.deleteLater()
            self._export_worker = None
        if self._export_thread is not None:
            self._export_thread.wait()
            self._export_thread.deleteLater()
            self._export_thread = None

    def _cleanup_alignment_worker(self) -> None:
        if self._alignment_worker is not None:
            self._alignment_worker.deleteLater()
            self._alignment_worker = None
        if self._alignment_thread is not None:
            self._alignment_thread.wait()
            self._alignment_thread.deleteLater()
            self._alignment_thread = None

    def _cleanup_repair_worker(self) -> None:
        if self._repair_worker is not None:
            self._repair_worker.deleteLater()
            self._repair_worker = None
        if self._repair_thread is not None:
            self._repair_thread.wait()
            self._repair_thread.deleteLater()
            self._repair_thread = None


def atlas_stitch_widget(napari_viewer=None) -> QWidget:
    return AtlasStitchWidget(viewer=napari_viewer)


def open_atlas_stitch_widget(viewer: Viewer):
    existing_widget = getattr(viewer.window, "dock_widgets", {}).get("Atlas Stitch")
    if existing_widget is not None:
        try:
            existing_widget.show()
            existing_widget.raise_()
        except Exception:
            pass
        return existing_widget
    widget = AtlasStitchWidget(viewer=viewer)
    dock_widget = viewer.window.add_dock_widget(widget, name="Atlas Stitch", area="right")
    try:
        dock_widget.show()
        dock_widget.raise_()
    except Exception:
        pass
    return dock_widget


def build_project_summary(project: AtlasProject) -> str:
    metadata = project.metadata
    extra = metadata.extra_metadata
    ignored_count = _metadata_int(extra, "ignored_non_tile_elements")
    duplicate_count = _metadata_int(extra, "duplicate_tile_elements")
    repaired_count = len([tile for tile in project.tiles if str(tile.repaired_path or "").strip()])
    manual_count = len(
        [tile for tile in project.tiles if tile.transform.manual_x is not None and tile.transform.manual_y is not None]
    )
    sections = [
        (
            "Source",
            [
                f"Atlas name: {metadata.atlas_name or '(unnamed)'}",
                f"Source path: {metadata.xml_path or '(not available)'}",
                f"Tile root override: {metadata.tile_root_override or '(none)'}",
                f"Source directory: {metadata.source_directory or '(not available)'}",
                f"Source software: {_source_software_text(project)}",
                f"Pixel size: {_pixel_size_text(project)}",
                f"Pixel size unit: {_pixel_size_unit(project)}",
                f"Bit depth / bits per sample: {_bit_depth_text(project)}",
                f"Samples per pixel / channels: {_samples_channels_text(project)}",
            ],
        ),
        (
            "Tile validation",
            [
                f"Tiles parsed: {len(project.tiles)}",
                f"Missing tiles: {len(project.missing_tiles)}",
                f"Tiles with repaired content: {repaired_count}",
                f"Tile entries used: {len(project.tiles)}",
                f"Ignored non-tile XML elements: {ignored_count if ignored_count is not None else '(not reported)'}",
                f"Duplicate tile XML elements ignored: {duplicate_count if duplicate_count is not None else '0'}",
            ],
        ),
        (
            "Layout / placement",
            [
                f"Nominal canvas: {_nominal_canvas_text(metadata)}",
                f"Depth: {_optional_dimension_text(metadata.image_depth)}",
                f"Tile placement mode: {_placement_mode_text(project)}",
            ],
        ),
        (
            "Registration",
            [
                f"Refinement method: {_alignment_method_summary_text(extra)}",
                f"Tile overlap: {_metadata_value_text(extra.get('atlas_stitch_overlap_percent'))} %",
                f"Minimum confidence: {_metadata_value_text(extra.get('atlas_stitch_min_confidence'))}",
                f"Max correction px: {_metadata_value_text(extra.get('atlas_stitch_max_correction_px'))}",
                f"Fusion method: {_metadata_text(extra, 'atlas_stitch_fusion_method') or '(not available)'}",
                f"Refinement status: {_metadata_text(extra, 'atlas_stitch_refinement_status') or '(not available)'}",
                f"Neighbor pair count: {_metadata_value_text(extra.get('atlas_stitch_neighbor_pairs_total'))}",
                f"Accepted pair count: {_metadata_value_text(extra.get('atlas_stitch_neighbor_pairs_accepted'))}",
                f"Fallback count: {_fallback_count_text(extra.get('atlas_stitch_neighbor_fallback_reasons'))}",
                f"Mean residual px: {_metadata_value_text(extra.get('atlas_stitch_mean_residual_px'))}",
                f"Max residual px: {_metadata_value_text(extra.get('atlas_stitch_max_residual_px'))}",
                f"High-residual pair count: {_metadata_value_text(extra.get('atlas_stitch_high_residual_pair_count'))}",
                f"Low-confidence pair count: {_low_confidence_pair_count_text(extra)}",
                f"Auto-registration runtime: {_elapsed_registration_text(extra) or '(not available)'}",
                f"Constraint count: {_metadata_value_text(extra.get('atlas_stitch_constraint_count'))}",
                f"Constrained tile count: {_metadata_value_text(extra.get('atlas_stitch_constrained_tile_count'))}",
                f"Isolated tile count: {_metadata_value_text(extra.get('atlas_stitch_isolated_tile_count'))}",
                f"Anchor component count: {_metadata_value_text(extra.get('atlas_stitch_anchor_component_count'))}",
                f"Refined tile count: {_metadata_value_text(extra.get('atlas_stitch_refined_tile_count'))}",
                f"Neighbor fallback reasons: {_fallback_reason_text(extra.get('atlas_stitch_neighbor_fallback_reasons'))}",
                f"Neighbor skipped reasons: {_fallback_reason_text(extra.get('atlas_stitch_neighbor_skip_reasons'))}",
            ],
        ),
        (
            "Manual adjustments",
            [
                f"Tiles with saved manual positions: {manual_count}",
                f"Worked/manual tile ids: {_manual_tile_ids_text(project)}",
            ],
        ),
        (
            "Last export",
            [
                f"Last exported OME-Zarr path: {_export_text(project.last_export.path)}",
                f"Export mode: {_export_text(project.last_export.mode)}",
                f"Export time: {_export_text(project.last_export.time)}",
                f"Chunk size: {_export_value_text(project.last_export.chunk_size)}",
                f"Pyramid enabled: {_export_bool_text(project.last_export.build_pyramid)}",
                f"Exported tile count: {_export_value_text(project.last_export.tile_count)}",
                f"Export status: {_export_text(project.last_export.status)}",
                f"Linked atlas project path: {_export_text(project.last_export.atlas_project_path)}",
            ],
        ),
        (
            "Warnings",
            [
                f"Missing tile files: {len(project.missing_tiles)}",
            ]
            + [f"Note: {warning}" for warning in project.warnings],
        ),
    ]
    lines: list[str] = []
    for title, entries in sections:
        lines.append(title)
        lines.extend(f"  {entry}" for entry in entries if entry)
        lines.append("")
    return "\n".join(lines).strip()


def _source_software_text(project: AtlasProject) -> str:
    metadata = project.metadata
    return metadata.source_software or _metadata_text(
        metadata.extra_metadata,
        "software",
        "application",
        "generator",
        "vendor",
    )


def _placement_mode_text(project: AtlasProject) -> str:
    extra = project.metadata.extra_metadata
    placement_mode = _metadata_text(extra, "atlas_stitch_placement_mode")
    if placement_mode:
        return placement_mode
    return "nominal"


def _alignment_method_summary_text(metadata: dict[str, Any]) -> str:
    method = _metadata_text(metadata, "atlas_stitch_refinement_method")
    if not method:
        return "(not available)"
    return _alignment_method_label(method)


def _nominal_canvas_text(metadata) -> str:
    width = metadata.image_width
    height = metadata.image_height
    if width is None or height is None:
        return "(not available)"
    if metadata.image_depth is not None:
        return f"{width} x {height} x {metadata.image_depth} px"
    return f"{width} x {height} px"


def _optional_dimension_text(value: int | None) -> str:
    return "(not available)" if value is None else str(value)


def _project_nominal_canvas(project: AtlasProject) -> tuple[int, int]:
    max_x = 0.0
    max_y = 0.0
    for tile in project.tiles:
        tile_x = tile.start_x if tile.start_x is not None else tile.transform.nominal_x
        tile_y = tile.start_y if tile.start_y is not None else tile.transform.nominal_y
        tile_width, tile_height = _tile_shape(tile)
        max_x = max(max_x, float(tile_x) + float(tile_width))
        max_y = max(max_y, float(tile_y) + float(tile_height))
    return int(math.ceil(max_x)), int(math.ceil(max_y))


def _pixel_size_text(project: AtlasProject) -> str:
    metadata = project.metadata
    unit = _pixel_size_unit(project)
    fallback_value = _metadata_float(
        metadata.extra_metadata,
        "pixel_size_value",
        "pixelsizevalue",
        "pixel_size",
        "pixelsize",
        "value",
    )
    voxel_x = metadata.voxel_size_x if metadata.voxel_size_x is not None else fallback_value
    voxel_y = metadata.voxel_size_y if metadata.voxel_size_y is not None else fallback_value
    voxel_z = metadata.voxel_size_z
    parts: list[str] = []
    if voxel_x is not None and voxel_y is not None:
        if abs(voxel_x - voxel_y) < 1e-12:
            parts.append(f"{_format_number(voxel_x)} {unit}/pixel")
        else:
            parts.append(f"X: {_format_number(voxel_x)} {unit}/pixel")
            parts.append(f"Y: {_format_number(voxel_y)} {unit}/pixel")
    else:
        if voxel_x is not None:
            parts.append(f"X: {_format_number(voxel_x)} {unit}/pixel")
        if voxel_y is not None:
            parts.append(f"Y: {_format_number(voxel_y)} {unit}/pixel")
    if voxel_z is not None:
        parts.append(f"Z: {_format_number(voxel_z)} {unit}")
    return ", ".join(parts) if parts else "(not available)"


def _pixel_size_unit(project: AtlasProject) -> str:
    value = _metadata_text(
        project.metadata.extra_metadata,
        "pixel_size_unit",
        "pixel_unit",
        "voxel_size_unit",
        "voxel_unit",
        "physicalsizeunitx",
        "physicalsizeunity",
        "unit",
        "units",
    )
    return _normalize_unit(value) if value else "um"


def _bit_depth_text(project: AtlasProject) -> str:
    value = _metadata_text(
        project.metadata.extra_metadata,
        "bit_depth",
        "bitdepth",
        "bits_per_sample",
        "bitspersample",
        "bit_per_sample",
        "bitpersample",
        "significant_bits",
    )
    if not value:
        return "(not available)"
    try:
        return f"{int(float(value))}-bit"
    except (TypeError, ValueError):
        return str(value)


def _samples_channels_text(project: AtlasProject) -> str:
    samples = _metadata_text(
        project.metadata.extra_metadata,
        "samples_per_pixel",
        "samplesperpixel",
        "sample_per_pixel",
        "sampleperpixel",
        "samples",
    )
    channels = project.metadata.channel_count
    if samples:
        return str(samples)
    if channels is not None:
        return str(channels)
    return "(not available)"


def _metadata_text(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _metadata_value_text(value: Any) -> str:
    return "(not available)" if value is None or str(value).strip() == "" else str(value)


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_float(metadata: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _normalize_unit(value: str) -> str:
    normalized = str(value).strip()
    lowered = normalized.lower()
    if lowered in {"um", "µm", "μm", "micrometer", "micrometers", "micron", "microns", "micrometre", "micrometres"}:
        return "µm"
    if lowered in {"nm", "nanometer", "nanometers", "nanometre", "nanometres"}:
        return "nm"
    if lowered in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return "mm"
    return normalized


def _export_text(value: str) -> str:
    return str(value).strip() or "(not available)"


def _export_value_text(value: Any) -> str:
    return "(not available)" if value is None or str(value).strip() == "" else str(value)


def _export_bool_text(value: bool | None) -> str:
    if value is None:
        return "(not available)"
    return "yes" if value else "no"


def _format_export_progress(stage: str, current: int, total: int) -> str:
    if stage == "Preparing tiles" and current >= 0 and total > 0:
        return f"Export: preparing tiles {current} / {total}"
    if stage == "Reading tiles" and current >= 0 and total > 0:
        return f"Export: reading tile {current} / {total}"
    if stage == "Assembling atlas":
        return "Export: assembling atlas"
    if stage == "Writing OME-Zarr":
        return "Export: writing OME-Zarr"
    if stage == "Building pyramid":
        return "Export: building pyramid"
    if stage == "Finalizing metadata":
        return "Export: finalizing metadata"
    if stage == "Export complete":
        return "Export complete"
    return f"Export: {stage}"


def _elapsed_registration_text(metadata: dict[str, Any]) -> str:
    value = metadata.get("atlas_stitch_registration_elapsed_seconds")
    if value in (None, ""):
        return ""
    try:
        seconds = max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return ""
    return _format_elapsed_seconds(seconds)


def _format_elapsed_seconds(seconds: int) -> str:
    minutes, remaining_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hr")
    if minutes:
        parts.append(f"{minutes} min")
    if remaining_seconds or not parts:
        parts.append(f"{remaining_seconds} sec")
    return " ".join(parts)


def _alignment_completion_status(project: AtlasProject) -> str:
    extra = project.metadata.extra_metadata
    elapsed_text = _elapsed_registration_text(extra)
    lead = "Auto-registration complete"
    if elapsed_text:
        lead = f"{lead} in {elapsed_text}"
    return (
        f"{lead}. "
        f"Pairs: total {_metadata_value_text(extra.get('atlas_stitch_neighbor_pairs_total'))}, "
        f"accepted {_metadata_value_text(extra.get('atlas_stitch_neighbor_pairs_accepted'))}. "
        f"Fallbacks: {_fallback_reason_text(extra.get('atlas_stitch_neighbor_fallback_reasons'))}. "
        f"Skipped: {_fallback_reason_text(extra.get('atlas_stitch_neighbor_skip_reasons'))}. "
        f"Constrained tiles: {_metadata_value_text(extra.get('atlas_stitch_constrained_tile_count'))}; "
        f"isolated tiles: {_metadata_value_text(extra.get('atlas_stitch_isolated_tile_count'))}. "
        "Review the refined layout and alignment summary before exporting."
    )


def _format_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _fallback_reason_text(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "(not available)"
    return ", ".join(f"{key}={count}" for key, count in sorted(value.items()))


def _fallback_count_text(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "(not available)"
    try:
        return str(sum(int(count) for count in value.values()))
    except (TypeError, ValueError):
        return "(not available)"


def _low_confidence_pair_count_text(metadata: dict[str, Any]) -> str:
    explicit = metadata.get("atlas_stitch_low_confidence_pair_count")
    if explicit not in (None, ""):
        return str(explicit)
    skip_reasons = metadata.get("atlas_stitch_neighbor_skip_reasons")
    fallback_reasons = metadata.get("atlas_stitch_neighbor_fallback_reasons")
    count = 0
    found = False
    for reasons in (skip_reasons, fallback_reasons):
        if not isinstance(reasons, dict):
            continue
        for key, value in reasons.items():
            if "confidence" not in str(key):
                continue
            found = True
            try:
                count += int(value)
            except (TypeError, ValueError):
                pass
    return str(count) if found else "(not available)"


def _manual_tile_ids_text(project: AtlasProject) -> str:
    tile_ids = [
        tile.tile_id
        for tile in project.tiles
        if tile.transform.manual_x is not None and tile.transform.manual_y is not None
    ]
    return ", ".join(tile_ids) if tile_ids else "(none)"


def _alignment_method_label(method: str) -> str:
    value = str(method or "").strip().lower()
    if value == ROBUST_ALIGNMENT_METHOD:
        return "Robust Translation"
    if value == ROBUST_OUTLIER_ALIGNMENT_METHOD:
        return "Robust Translation + Outlier Rejection"
    if value == FEATURE_MATCHING_METHOD:
        return "Feature Matching"
    if value == ECC_TRANSLATION_METHOD:
        return "ECC Translation"
    if value == CENTER_OUT_CLOCKWISE_METHOD:
        return "Center-Out Clockwise"
    if value == CENTER_OUT_COUNTERCLOCKWISE_METHOD:
        return "Center-Out Counterclockwise"
    return "Light Translation"


def _placement_mode_label(placement_mode: str) -> str:
    value = str(placement_mode or "").strip().lower()
    if value == "refined":
        return "Refined"
    if value == "manual":
        return "Manual"
    return "Nominal"


def _next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    power = 1
    while power < value:
        power *= 2
    return power


def _refresh_project_tile_availability(project: AtlasProject | None) -> None:
    if project is None:
        return
    missing: list[str] = []
    for tile in project.tiles:
        path = preferred_tile_path(tile)
        tile.exists = bool(path and Path(path).expanduser().exists())
        if tile.start_x is None:
            tile.start_x = float(tile.transform.nominal_x)
        if tile.start_y is None:
            tile.start_y = float(tile.transform.nominal_y)
        if not tile.exists:
            missing.append(tile.file_name or tile.tile_id)
    project.missing_tiles = missing


def _project_overlap_fraction(project: AtlasProject | None) -> float:
    if project is None:
        return 0.1
    extra = project.metadata.extra_metadata
    value = extra.get("atlas_stitch_overlap_fraction")
    if value in (None, ""):
        percent = extra.get("atlas_stitch_overlap_percent")
        if percent not in (None, ""):
            try:
                return max(0.01, min(1.0, float(percent) / 100.0))
            except (TypeError, ValueError):
                return 0.1
        return 0.1
    try:
        return max(0.01, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.1


def _preview_position(tile: TileRecord, placement_mode: str) -> tuple[float, float] | None:
    mode = str(placement_mode or "nominal").strip().lower()

    if mode == "manual":
        if tile.transform.manual_x is not None and tile.transform.manual_y is not None:
            return float(tile.transform.manual_x), float(tile.transform.manual_y)
        if tile.start_x is not None and tile.start_y is not None:
            return float(tile.start_x), float(tile.start_y)
        return float(tile.transform.nominal_x), float(tile.transform.nominal_y)

    if mode == "refined":
        if tile.transform.refined_x is not None and tile.transform.refined_y is not None:
            return float(tile.transform.refined_x), float(tile.transform.refined_y)
        if tile.start_x is not None and tile.start_y is not None:
            return float(tile.start_x), float(tile.start_y)
        return float(tile.transform.nominal_x), float(tile.transform.nominal_y)

    if tile.start_x is not None and tile.start_y is not None:
        return float(tile.start_x), float(tile.start_y)
    return float(tile.transform.nominal_x), float(tile.transform.nominal_y)


def _tile_translate(tile: TileRecord, placement_mode: str) -> tuple[float, float]:
    position = _preview_position(tile, placement_mode)
    if position is None:
        return 0.0, 0.0
    tile_x, tile_y = position
    return float(tile_y), float(tile_x)


def _tile_bounds_shape(tile: TileRecord, placement_mode: str) -> np.ndarray:
    tile_width, tile_height = _tile_shape(tile)
    translate_y, translate_x = _tile_translate(tile, placement_mode)
    return np.asarray(
        [
            [translate_y, translate_x],
            [translate_y + float(tile_height), translate_x],
            [translate_y + float(tile_height), translate_x + float(tile_width)],
            [translate_y, translate_x + float(tile_width)],
        ],
        dtype=float,
    )


def _grid_position_label(tile: TileRecord) -> str:
    row_text = "?" if tile.row is None else str(tile.row)
    col_text = "?" if tile.col is None else str(tile.col)
    return f"r{row_text} c{col_text}"


def _parse_grid_position_text(text: str) -> tuple[int, int] | None:
    cleaned = str(text or "").strip().lower().replace(",", " ")
    if not cleaned:
        return None
    parts = [part for part in cleaned.split() if part]
    if len(parts) == 1 and "c" in parts[0]:
        row_part, col_part = parts[0].split("c", 1)
        parts = [row_part, f"c{col_part}"]
    if len(parts) != 2:
        return None
    row_part, col_part = parts
    if not row_part.startswith("r") or not col_part.startswith("c"):
        return None
    try:
        return int(row_part[1:]), int(col_part[1:])
    except ValueError:
        return None


def _tile_shape(tile: TileRecord) -> tuple[int, int]:
    if tile.width is not None and tile.height is not None:
        return int(tile.width), int(tile.height)
    if preferred_tile_path(tile):
        try:
            data = load_preferred_tile_pixels(tile)
        except Exception:
            return 1, 1
        height, width = data.shape[:2]
        return int(width), int(height)
    return 1, 1


def _format_open_tiles_status(opened_ids: list[str], zero_ids: list[str], skipped_ids: list[str], focus_tile_id: str) -> str:
    parts: list[str] = []
    if opened_ids:
        parts.append(f"Opened {len(opened_ids)} tile(s)")
    if zero_ids:
        parts.append(f"Skipped zero-value tile(s): {_summarize_tile_ids(zero_ids)}")
    if skipped_ids:
        parts.append(f"Could not open tile(s): {_summarize_tile_ids(skipped_ids)}")
    if not parts and focus_tile_id:
        return f"Tile {focus_tile_id} is already open."
    if focus_tile_id and parts:
        return f"Focused {focus_tile_id}. " + " ".join(parts)
    return " ".join(parts) if parts else "No tiles opened."


def _summarize_tile_ids(tile_ids: list[str], limit: int = 5) -> str:
    unique_ids = [tile_id for tile_id in tile_ids if tile_id]
    if len(unique_ids) <= limit:
        return ", ".join(unique_ids)
    visible = ", ".join(unique_ids[:limit])
    return f"{visible}, +{len(unique_ids) - limit} more"
