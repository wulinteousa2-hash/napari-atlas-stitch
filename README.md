# napari-atlas-stitch

Standalone napari widget for stitching atlas tile exports from ZEISS Volutome and JEOL EM workflows.

Version: `1.1.0`

`napari-atlas-stitch` parses atlas metadata, checks tile availability, previews nominal/refined/manual tile layouts in napari, estimates refined tile positions from overlapping neighboring tiles, supports optional donor-based seam repair, and exports stitched atlas mosaics as OME-Zarr.

## Development Status

This plugin is under active development. Version `1.1.0` introduces a cleaner guided workflow UI, but alignment refinement, seam repair, and export settings should still be validated on your own acquisition workflow before production use.

## Supported Sources

- ZEISS Volutome atlas XML sources
- JEOL EM VE-MIF atlas sources
- 2D tile image files referenced by those atlas sources
- Saved `.json` atlas project files produced by this widget

## Install

From a local/git clode checkout:

```bash
git clone https://github.com/wulinteousa2-hash/napari-atlas-stitch
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
```

## Open In Napari

Start napari and open:

```text
Plugins > Atlas Stitch > Atlas Stitch
```

## Guided Workflow

The main widget is organized into workflow sections:

1. **Input** - Select an atlas XML or VE-MIF source, optionally set a tile-root override, then load or save an atlas project JSON.
2. **Inspect** - Review the tile table, locate grid positions, and preview the nominal layout from stage/grid metadata.
3. **Register** - Set tile overlap, choose light or robust translation, then run auto-registration to solve a refined global layout.
4. **Manual Adjustment** - Open selected tiles in napari, move them if needed, and save manual positions.
5. **Export** - Choose output settings, placement mode, fusion method, and export the stitched OME-Zarr mosaic.
6. **Status / Activity** - Monitor progress, status messages, and long-running operations.

Advanced seam repair is available as an optional panel for repairing damaged tile borders after layout alignment.

## Typical Use

1. Select an atlas XML or VE-MIF source.
2. Optionally choose a tile-root override if image paths moved after acquisition.
3. Click **Load Source**.
4. Confirm that the tile table and atlas summary populate.
5. Click **Preview Nominal Layout**.
6. Set overlap percent and alignment method.
7. Click **Run Auto-Registration**.
8. Click **Preview Refined Layout**.
9. Choose the export placement, usually `Refined` after registration.
10. Click **Export Stitched OME-Zarr**.
11. Click **Open Last Export** to inspect the result in napari.

## Outputs

- Stitched OME-Zarr mosaics
- Optional multiscale pyramids
- Saved atlas project JSON files
- Repaired tile outputs and repair metadata when seam repair is used

## Current Limitations

- Registration is translation-based only.
- No bad-pair override UI is included yet.
- Residual/high-residual diagnostics are displayed only when metadata exists.
- Seam repair is optional and should be validated carefully before replacing original tile data in downstream analysis.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes and feature details.



OME-Zarr export uses `ome-zarr>=0.11` with `zarr>=3`. If tests fail while importing `ome_zarr`, reinstall the package in a clean or updated environment so pip can resolve those versions together.
