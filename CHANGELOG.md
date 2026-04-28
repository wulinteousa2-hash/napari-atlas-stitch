# Changelog

All notable changes to `napari-atlas-stitch` are documented here.

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
