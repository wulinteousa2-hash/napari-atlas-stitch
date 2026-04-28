# napari-atlas-stitch

Standalone napari widget for stitching atlas tile exports from ZEISS Volutome and JEOL EM workflows.

The plugin can parse atlas source metadata, inspect tile availability, preview nominal/refined/manual layouts in napari, refine tile positions from overlap constraints, repair seams from donor tiles, and export stitched atlas mosaics as OME-Zarr.

## Install for Development

```bash
python -m pip install -e ".[test]"
```

Then start napari and open the widget from:

```text
Plugins > Atlas Stitch > Atlas Stitch
```

## Supported Inputs

- ZEISS Volutome atlas XML sources
- JEOL EM VE-MIF atlas sources
- 2D tile image files referenced by those atlas sources
- Saved `.json` atlas project files produced by this widget

## Main Workflow

1. Select an atlas XML or VE-MIF source.
2. Optionally choose a tile-root override if image paths moved after acquisition.
3. Load the atlas project and inspect missing or resolved tiles.
4. Preview nominal placement, estimate refined alignment, or save manual tile positions.
5. Export the selected placement as OME-Zarr.

## Tests

```bash
python -m pytest
```

OME-Zarr export uses `ome-zarr>=0.11` with `zarr>=3`. If tests fail while importing `ome_zarr`, reinstall the package in a clean or updated environment so pip can resolve those versions together.

## Notes

This repository was split out from `napari-chat-assistant`. The chat assistant can still launch Atlas Stitch from its Advanced menu, and will prefer this standalone plugin when it is installed in the same environment.
