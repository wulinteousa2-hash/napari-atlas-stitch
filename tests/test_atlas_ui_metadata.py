from __future__ import annotations

import textwrap

from atlas_stitch.models import AtlasExportInfo, AtlasMetadata, AtlasProject
from atlas_stitch.project_state import load_atlas_project, save_atlas_project
from atlas_stitch.widget import build_project_summary
from atlas_stitch.xml_parser import parse_atlas_xml


def test_parse_atlas_xml_reports_ignored_non_tile_elements_without_malformed_language(tmp_path):
    xml_path = tmp_path / "atlas.xml"
    xml_path.write_text(
        textwrap.dedent(
            """\
            <Atlas>
              <Metadata>
                <Name>Demo Atlas</Name>
                <PixelSize>
                  <Unit>µm</Unit>
                  <Value>0.00600000005215406</Value>
                </PixelSize>
                <BitPerSample>8</BitPerSample>
                <SamplePerPixel>1</SamplePerPixel>
              </Metadata>
              <Tile>
                <Row>0</Row>
                <Column>0</Column>
                <FilePath>tile_000.tif</FilePath>
                <StartX>0</StartX>
                <StartY>0</StartY>
              </Tile>
              <MetadataNote importance="high">qc passed</MetadataNote>
            </Atlas>
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "tile_000.tif").write_bytes(b"placeholder")

    project = parse_atlas_xml(str(xml_path))

    assert len(project.tiles) == 1
    assert project.metadata.voxel_size_x == 0.00600000005215406
    assert project.metadata.voxel_size_y == 0.00600000005215406
    assert project.metadata.extra_metadata["pixel_size_unit"] == "µm"
    assert any("Ignored" in warning and "non-tile XML element" in warning for warning in project.warnings)
    assert all("malformed" not in warning.lower() for warning in project.warnings)
    assert project.metadata.extra_metadata["ignored_non_tile_elements"] >= 1


def test_build_project_summary_surfaces_em_metadata_and_parsing_notes():
    project = AtlasProject(
        metadata=AtlasMetadata(
            atlas_name="Demo Atlas",
            xml_path="/data/demo.xml",
            source_directory="/data",
            tile_root_override="/tiles",
            source_software="AtlasSoft",
            image_width=65167,
            image_height=45200,
            channel_count=1,
            voxel_size_x=0.006,
            voxel_size_y=0.006,
            extra_metadata={
                "pixel_size_unit": "µm",
                "bit_per_sample": "8",
                "sample_per_pixel": "1",
                "ignored_non_tile_elements": 5898,
                "duplicate_tile_elements": 2,
            },
        ),
        warnings=["Ignored 5898 non-tile XML element(s) during parsing."],
        last_export=AtlasExportInfo(
            path="/tmp/atlas.ome.zarr",
            mode="nominal",
            time="2026-04-07T14:30:00-05:00",
            chunk_size=256,
            build_pyramid=True,
            tile_count=736,
            status="completed",
            atlas_project_path="/tmp/project.atlas.json",
        ),
    )

    summary = build_project_summary(project)

    assert "Atlas" in summary
    assert "Atlas name: Demo Atlas" in summary
    assert "Nominal canvas: 65167 x 45200 px" in summary
    assert "Pixel size: 0.006 µm/pixel" in summary
    assert "Pixel size unit: µm" in summary
    assert "Bit depth / bits per sample: 8-bit" in summary
    assert "Samples per pixel / channels: 1" in summary
    assert "Last exported OME-Zarr path: /tmp/atlas.ome.zarr" in summary
    assert "Export mode: nominal" in summary
    assert "Chunk size: 256" in summary
    assert "Pyramid enabled: yes" in summary
    assert "Linked atlas project path: /tmp/project.atlas.json" in summary
    assert "Ignored non-tile XML elements: 5898" in summary


def test_atlas_project_round_trips_last_export_metadata(tmp_path):
    project = AtlasProject(
        metadata=AtlasMetadata(atlas_name="Demo Atlas"),
        last_export=AtlasExportInfo(
            path="/tmp/atlas.ome.zarr",
            mode="nominal",
            time="2026-04-07T14:30:00-05:00",
            chunk_size=128,
            build_pyramid=False,
            tile_count=42,
            status="completed",
            atlas_project_path="/tmp/demo.atlas.json",
        ),
    )
    destination = tmp_path / "demo.atlas.json"

    save_atlas_project(project, str(destination))
    restored = load_atlas_project(destination)

    assert restored.last_export.path == "/tmp/atlas.ome.zarr"
    assert restored.last_export.mode == "nominal"
    assert restored.last_export.chunk_size == 128
    assert restored.last_export.build_pyramid is False
    assert restored.last_export.atlas_project_path == "/tmp/demo.atlas.json"
