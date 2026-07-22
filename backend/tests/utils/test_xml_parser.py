"""Tests for XMLParser over small GROBID-like TEI fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.utils.XMLParser import XMLParser


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_xml_parser_reads_metadata_and_text_from_grobid_fixture() -> None:
    """Checks title, abstract, body text, sections, and authors from TEI XML."""
    parser = XMLParser(FIXTURES_DIR / "scipdf_xml" / "sample.xml")

    assert parser.get_title() == "Fixture Model Paper"
    assert parser.get_abstract() == "Fixture abstract."
    assert parser.get_authors() == ["Jane Doe", "John Smith"]
    assert "We evaluate FixtureModel with MRR and Hits@10." in parser.get_full_text()
    assert parser.get_sections(["Experiments"]) == {
        "Experiments": "Experiments We evaluate FixtureModel with MRR and Hits@10."
    }


def test_xml_parser_returns_all_sections_when_no_target_filter_is_given() -> None:
    """Checks that XMLParser can expose all body sections, not only target ones."""
    parser = XMLParser(FIXTURES_DIR / "scipdf_xml" / "sample.xml")

    sections = parser.get_sections()

    assert set(sections) == {
        "Experiments",
        "Discussion",
    }
    assert "This section is not selected" in sections["Discussion"]


def test_xml_parser_falls_back_to_plain_title_and_extracts_arxiv_id() -> None:
    """Checks title fallback and arXiv ID parsing from idno values with versions."""
    parser = XMLParser(FIXTURES_DIR / "scipdf_xml" / "no_target_sections.xml")

    assert parser.get_title() == "No Target Section Paper"
    assert parser.get_arxiv_id() == "2401.01234"


def test_xml_parser_returns_empty_values_when_optional_blocks_are_missing(
    workspace_tmp_path: Path,
) -> None:
    """Checks missing optional TEI blocks return empty defaults instead of errors."""
    xml_path = workspace_tmp_path / "minimal.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><body /></text>
</TEI>
""",
        encoding="utf-8",
    )

    parser = XMLParser(xml_path)

    assert parser.get_title() is None
    assert parser.get_abstract() == ""
    assert parser.get_full_text() == ""
    assert parser.get_sections(["Experiments"]) == {}
    assert parser.get_arxiv_id() is None
    assert parser.get_authors() == []


def test_xml_parser_raises_when_xml_file_does_not_exist(
    workspace_tmp_path: Path,
) -> None:
    """Checks that missing XML fixtures fail early with a clear file error."""
    with pytest.raises(FileNotFoundError):
        XMLParser(workspace_tmp_path / "missing.xml")
