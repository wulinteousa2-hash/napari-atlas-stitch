# napari-atlas-stitch

Standalone napari widget for stitching atlas tile exports from ZEISS Volutome and JEOL EM workflows.

Version: `1.0.0`

`napari-atlas-stitch` parses atlas metadata, checks tile availability, previews nominal/refined/manual tile layouts in napari, estimates refined tile positions from overlaps, supports donor-based seam repair, and exports stitched atlas mosaics as OME-Zarr.

## Development Status

This plugin is under active development. The core workflow has been split into a standalone napari package, but many functions still need broader testing on real ZEISS Volutome and JEOL EM datasets. Treat alignment refinement, seam repair, and export settings as experimental until validated on your acquisition workflow.

## Supported Sources

- ZEISS Volutome atlas XML sources
- JEOL EM VE-MIF atlas sources
- 2D tile image files referenced by those atlas sources
- Saved `.json` atlas project files produced by this widget

## Install

From a local checkout:

```bash
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

## Main Workflow

1. Select an atlas XML or VE-MIF source.
2. Optionally choose a tile-root override if image paths moved after acquisition.
3. Load the atlas project and inspect missing or resolved tiles.
4. Preview the nominal placement.
5. Estimate refined alignment from overlaps when the atlas has usable neighboring tiles.
6. Save manual tile positions or repair seams when needed.
7. Export the selected placement as OME-Zarr.

## Outputs

- Stitched OME-Zarr mosaics
- Optional multiscale pyramids
- Saved atlas project JSON files
- Repaired tile outputs and repair metadata when seam repair is used

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes and feature details.

## Relationship To napari-chat-assistant

This repository was split out from `napari-chat-assistant` so Atlas Stitch can be installed, released, and pushed independently. The chat assistant can still launch Atlas Stitch from its Advanced menu, and will prefer this standalone plugin when it is installed in the same environment.

## Tests

```bash
python -m pytest
```

OME-Zarr export uses `ome-zarr>=0.11` with `zarr>=3`. If tests fail while importing `ome_zarr`, reinstall the package in a clean or updated environment so pip can resolve those versions together.
