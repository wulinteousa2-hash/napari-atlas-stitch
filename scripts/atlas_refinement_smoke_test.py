from __future__ import annotations

import argparse
from pathlib import Path

from atlas_stitch import (
    build_neighbor_constraints,
    parse_atlas_xml,
    solve_refined_tile_positions,
    summarize_neighbor_constraints,
    summarize_refined_positions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the atlas_stitch Phase 3 refinement backend on a real atlas XML.")
    parser.add_argument("xml_path", help="Path to atlas XML, for example '/path/to/Region 1.xml'")
    parser.add_argument("--tile-root", default="", help="Optional tile root override directory")
    args = parser.parse_args()

    xml_path = Path(args.xml_path).expanduser()
    project = parse_atlas_xml(str(xml_path), tile_root_override=args.tile_root or None)
    constraints = build_neighbor_constraints(project)
    solved = solve_refined_tile_positions(project, constraints)

    constraint_summary = summarize_neighbor_constraints(project, constraints)
    refined_summary = summarize_refined_positions(solved)

    print("Atlas Refinement Smoke Test")
    print(f"XML: {xml_path}")
    print(f"Atlas: {project.metadata.atlas_name or '(unnamed)'}")
    print("")
    print("Constraint diagnostics")
    for key in (
        "tile_count",
        "neighbor_pairs_total",
        "neighbor_pairs_accepted",
        "constraint_count",
        "constrained_tile_count",
        "isolated_tile_count",
        "anchor_component_count",
    ):
        print(f"  {key}: {constraint_summary[key]}")
    skip_reasons = dict(constraint_summary.get("skip_reasons") or {})
    if skip_reasons:
        print("  skip_reasons:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"    {reason}: {count}")
    fallback_reasons = dict(constraint_summary.get("fallback_reasons") or {})
    if fallback_reasons:
        print("  fallback_reasons:")
        for reason, count in sorted(fallback_reasons.items()):
            print(f"    {reason}: {count}")
    print("")
    print("Refined position diagnostics")
    for key in (
        "refined_tile_count",
        "max_abs_shift_x",
        "max_abs_shift_y",
        "mean_shift_x",
        "mean_shift_y",
        "std_shift_x",
        "std_shift_y",
        "zero_like_refined_tile_count",
    ):
        print(f"  {key}: {refined_summary[key]}")
    print("")
    print("First 10 nominal vs refined positions")
    for tile in solved.tiles[:10]:
        print(
            f"  {tile.tile_id}: nominal=({tile.transform.nominal_x}, {tile.transform.nominal_y}) "
            f"refined=({tile.transform.refined_x}, {tile.transform.refined_y})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
