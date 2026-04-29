# Changelog

All notable changes to `napari-atlas-stitch` are documented here.

## 1.1.0 - Guided Workflow Refactor

This release reorganizes the Atlas Stitch widget into a clearer workflow for loading atlas sources, inspecting tile layout, running overlap-based registration, applying manual adjustments, and exporting stitched OME-Zarr mosaics.

### Added

- Guided workflow sections in the main widget: Input, Inspect, Register, Manual Adjustment, Export, and Status / Activity.
- A workflow guide label near the top of the widget: Load source → Preview nominal → Run auto-registration → Preview refined → Export.
- Clearer button labels and tooltips for the main user actions.
- A dedicated Inspect section for preview downsample settings and grid lookup.
- A dedicated Export section for output folder, output name, chunk size, pyramid export, fusion method, placement mode, subset export, export, and open-last-export actions.
- Centralized button enablement through `_update_refinement_controls` for project-loaded, preview-running, alignment-running, and export-running states.
- Safer gating for refined and manual previews so those buttons are enabled only when matching positions exist.
- Auto-registration gating so registration is enabled only when usable neighboring tiles are available.
- Export gating so export is enabled only when exportable tile files are available.

### Changed

- Replaced the previous crowded Actions group with a step-oriented workflow layout.
- Renamed the seam repair panel to Advanced Seam Repair and marked it as optional after layout alignment.
- Moved preview downsample and grid lookup controls from the general action area into Inspect.
- Moved export controls and open-last-export actions into Export.
- Kept registration translation-based; this release does not introduce rotation, affine, deformable registration, or bad-pair override UI.
- Preserved existing backend calls, including `parse_atlas_source`, `build_neighbor_constraints`, `solve_refined_tile_positions`, `export_nominal_layout_to_omezarr`, `save_atlas_project`, and `load_atlas_project`.
- Preserved public package exports.
- Preserved export function naming for backward compatibility.

### Seam Repair

- Seam repair algorithms, workers, ROI behavior, donor logic, and save logic were intentionally left unchanged.
- No changes were made to `RepairDonorSpec`, `TileRepairRequest`, `reconstruct_tile_from_donors`, `save_repair_outputs`, or the repair worker logic.

### Validation

Manual napari validation workflow:

1. Start napari.
2. Open `Plugins > Atlas Stitch > Atlas Stitch`.
3. Select an atlas XML or VE-MIF source.
4. Optionally select a tile-root override.
5. Click Load Source.
6. Confirm the tile table and summary populate.
7. Click Preview Nominal Layout.
8. Set overlap percent and alignment method.
9. Click Run Auto-Registration.
10. Click Preview Refined Layout.
11. Choose Export placement = Refined.
12. Click Export Stitched OME-Zarr.
13. Click Open Last Export.

Additional validation reported for this release:

- Manifest still points to `atlas_stitch.widget:atlas_stitch_widget`.
- Import smoke check passed in the napari development environment.
- Offscreen Qt widget smoke check passed.
- `py_compile` passed for `widget.py`.

### Known Limitations

- No bad-pair override UI is included yet.
- Residual and high-residual diagnostics are displayed only when metadata exists; this release does not add new diagnostic computation.
- UI-builder helper extraction was deferred to keep the patch focused and lower risk.
- Registration remains translation-based only.

## 1.0.1 - Packaging And Documentation

### Added

- MIT license file for repository and package distributions.
- Documentation folder placeholder for future user and developer guides.

### Changed

- Bumped package version to `1.0.1`.

## 1.0.0 - Initial Standalone Release

This is a work-in-progress standalone release. The main Atlas Stitch workflow is available, but many functions still need broader testing and validation on real ZEISS Volutome and JEOL EM datasets.

### Added

- Standalone napari plugin package for Atlas Stitch, split out from `napari-chat-assistant`.
- napari manifest entry for opening the widget from `Plugins > Atlas Stitch > Atlas Stitch`.
- ZEISS Volutome atlas XML parsing.
- JEOL EM VE-MIF atlas parsing.
- Atlas project model for metadata, tile records, nominal transforms, refined transforms, manual transforms, repair history, and export metadata.
- Tile path resolution with optional tile-root override for moved acquisition folders.
- Tile availability checks and missing-tile reporting.
- Project save/load support using JSON atlas project files.
- Atlas summary panel with source metadata, tile counts, missing-tile counts, manual-position counts, repair counts, and export history.
- Tile table for inspecting file names, resolved paths, tile IDs, nominal positions, refined positions, and manual positions.
- Coarse napari preview generation for nominal, refined, and manual layouts.
- Overlap-based alignment refinement using neighboring tile constraints.
- Robust refinement diagnostics for accepted neighbor pairs, skipped pairs, fallback reasons, constrained tiles, isolated tiles, and anchor components.
- Manual tile-position workflow for saving adjusted positions from napari layers.
- Viewer context menu support for tile-centered workflows.
- Guard behavior for manually locked tiles.
- Donor-based seam repair workflow with full-overlap and ROI-guided modes.
- Hard-replace and feather-blend seam repair modes.
- Repair preview, overlap preview, apply, cancel, and clear controls.
- Repaired tile output support with confidence and attribution outputs.
- OME-Zarr export for stitched atlas mosaics.
- Optional multiscale pyramid export.
- Export placement modes for nominal, refined, and manual layouts.
- Export subset mode for worked/manual tiles.
- Fusion modes for overwrite, linear blend, average, max intensity, and min intensity.
- OME-Zarr metadata for atlas source path, tile root, tile count, placement mode, selected tile IDs, physical pixel size, sample metadata, fusion method, and export version.
- Focused test suite for atlas parsing metadata, project persistence, refinement solving, overlap refinement, OME-Zarr export, and manual stitching workflows.
- Smoke-test script for running refinement on a real atlas source from the command line.

### Changed

- Package name is now `napari-atlas-stitch`.
- Python import package is now `atlas_stitch`.
- Plugin settings namespace is now `napari-atlas-stitch`.
- OME-Zarr export metadata namespace is now `napari_atlas_stitch`.
- OME-Zarr export imports are lazy-loaded so the widget can import before export dependencies are exercised.

### Compatibility

- Requires Python 3.9 or newer.
- Uses `ome-zarr>=0.11` with `zarr>=3` for OME-Zarr export.
- `napari-chat-assistant` can still launch Atlas Stitch from its Advanced menu and will prefer this standalone plugin when it is installed in the same environment.
