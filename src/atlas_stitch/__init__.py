from __future__ import annotations

__version__ = "0.1.0"

_EXPORTS = {
    "AtlasExportInfo": ("atlas_stitch.models", "AtlasExportInfo"),
    "AtlasMetadata": ("atlas_stitch.models", "AtlasMetadata"),
    "AtlasProject": ("atlas_stitch.models", "AtlasProject"),
    "AtlasStitchWidget": ("atlas_stitch.widget", "AtlasStitchWidget"),
    "NeighborConstraint": ("atlas_stitch.refinement_solver", "NeighborConstraint"),
    "RepairDonorSpec": ("atlas_stitch.seam_repair", "RepairDonorSpec"),
    "TileRecord": ("atlas_stitch.models", "TileRecord"),
    "TileRepairRequest": ("atlas_stitch.seam_repair", "TileRepairRequest"),
    "TileRepairResult": ("atlas_stitch.seam_repair", "TileRepairResult"),
    "TileTransform": ("atlas_stitch.models", "TileTransform"),
    "build_neighbor_constraints": ("atlas_stitch.refinement_overlap", "build_neighbor_constraints"),
    "estimate_translation_phasecorr": ("atlas_stitch.refinement_overlap", "estimate_translation_phasecorr"),
    "export_nominal_layout_to_omezarr": ("atlas_stitch.ome_zarr_export", "export_nominal_layout_to_omezarr"),
    "extract_overlap_strip": ("atlas_stitch.refinement_overlap", "extract_overlap_strip"),
    "load_atlas_project": ("atlas_stitch.project_state", "load_atlas_project"),
    "parse_atlas_source": ("atlas_stitch.xml_parser", "parse_atlas_source"),
    "parse_atlas_vemif": ("atlas_stitch.xml_parser", "parse_atlas_vemif"),
    "parse_atlas_xml": ("atlas_stitch.xml_parser", "parse_atlas_xml"),
    "reconstruct_tile_from_donors": ("atlas_stitch.seam_repair", "reconstruct_tile_from_donors"),
    "save_atlas_project": ("atlas_stitch.project_state", "save_atlas_project"),
    "solve_refined_tile_positions": ("atlas_stitch.refinement_solver", "solve_refined_tile_positions"),
    "summarize_neighbor_constraints": ("atlas_stitch.refinement_diagnostics", "summarize_neighbor_constraints"),
    "summarize_refined_positions": ("atlas_stitch.refinement_diagnostics", "summarize_refined_positions"),
}

__all__ = sorted([*list(_EXPORTS), "__version__"])


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'atlas_stitch' has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
