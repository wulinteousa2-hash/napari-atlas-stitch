# napari-atlas-stitch

Standalone napari widget for stitching atlas tile exports from ZEISS Volutome and JEOL EM workflows.

Version: `1.2.0`

`napari-atlas-stitch` parses atlas metadata, checks tile availability, previews nominal, refined, and manual tile layouts in napari, estimates refined tile positions from overlapping neighboring tiles, supports optional donor-based seam repair, and exports stitched atlas mosaics as OME-Zarr.

## Development Status

This plugin is under active development. Alignment refinement, seam repair, and export settings should be treated as experimental until validated on your own ZEISS Volutome or JEOL EM acquisition workflow.

## Alignment Refinement

Version `1.2.0` adds additional overlap-registration options and safety controls for EM tile mosaics.

Atlas Stitch estimates translations between neighboring overlapping tiles, then solves a global layout from the accepted pairwise constraints. This helps prevent one local adjustment from disrupting the whole mosaic.

Available registration modes:

- **Light translation registration** — fast phase-correlation-based translation estimation.
- **Robust translation registration** — tests multiple overlap-strip widths and selects the most reliable translation estimate.
- **Robust translation with residual-based outlier rejection** — removes pairwise constraints that disagree with the solved global layout.
- **Feature-based registration** — uses local image features from overlapping regions when intensity correlation is unreliable.
- **ECC intensity-based registration** — uses enhanced correlation coefficient optimization for translation alignment.
- **Center-out seeded registration** — starts near the center of the tile grid and expands outward using clockwise or counterclockwise neighbor priority. This is useful when weak outer-tile matches distort the global layout.

Additional safeguards:

- **Maximum correction** — rejects translation corrections larger than the configured pixel limit.
- **Minimum confidence** — rejects weak pairwise registrations before global optimization.

Recommended EM starting settings:

```text
Registration mode: Robust translation with residual-based outlier rejection
Tile overlap: 10–20%
Maximum correction: 50 px
Minimum confidence: 0.20
```

Version: `1.1.0`

napari-atlas-stitch parses atlas metadata, checks tile availability, previews nominal/refined/manual tile layouts in napari, estimates refined tile positions from overlapping neighboring tiles, supports optional donor-based seam repair, and exports stitched atlas mosaics as OME-Zarr.


Version: `1.0.1`

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
git clone https://github.com/wulinteousa2-hash/napari-atlas-stitch
cd napari-atlas-stitch
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


## Tests

```bash
python -m pytest
```

OME-Zarr export uses `ome-zarr>=0.11` with `zarr>=3`. If tests fail while importing `ome_zarr`, reinstall the package in a clean or updated environment so pip can resolve those versions together.
