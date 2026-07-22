"""Tests for PDFHandler orchestration.

Planned scope:
- PDF reuse versus download;
- SciPDF/GROBID XML generation failures;
- LightOCR JSON generation and skip paths;
- XML/OCR value extraction;
- progress callbacks;
- ModelCard save behavior.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from backend.pdf_handler import PDFHandler


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def build_handler_without_init() -> PDFHandler:
    handler = PDFHandler.__new__(PDFHandler)
    handler._log = lambda message: None
    return handler


def test_extract_id_from_url_reads_arxiv_id_from_pdf_url() -> None:
    """Checks that PDFHandler can derive the paper ID from an arXiv PDF URL."""
    handler = build_handler_without_init()

    assert (
        handler._extract_id_from_url("https://arxiv.org/pdf/1802.09691.pdf")
        == "1802.09691"
    )


def test_extract_id_from_url_rejects_urls_without_arxiv_id() -> None:
    """Checks that non-arXiv-style URLs fail before pipeline paths are built."""
    handler = build_handler_without_init()

    with pytest.raises(ValueError):
        handler._extract_id_from_url("https://example.org/paper.pdf")


def test_emit_stage_forwards_progress_payload() -> None:
    """Checks that pipeline stage callbacks receive step metadata and details."""
    stages: list[dict] = []

    PDFHandler._emit_stage(
        stages.append,
        "extracting_xml",
        "Extracting XML with GROBID",
        3,
        detail="sample.xml",
        item_current=1,
        item_total=2,
    )

    assert stages == [
        {
            "key": "extracting_xml",
            "label": "Extracting XML with GROBID",
            "step": 3,
            "total": 7,
            "detail": "sample.xml",
            "item_current": 1,
            "item_total": 2,
        }
    ]


def test_process_with_scipdf_returns_generated_xml_path(
    workspace_tmp_path: Path,
) -> None:
    """Checks that SciPDF success is accepted only when the XML file exists."""
    handler = build_handler_without_init()
    xml_path = workspace_tmp_path / "sample.xml"

    class FakeSciPdfParser:
        def process(self, pdf_path: Path, xml_save_path: Path) -> Path:
            xml_save_path.write_text("<TEI/>", encoding="utf-8")
            return xml_save_path

    handler._scipdf_parser = FakeSciPdfParser()

    assert handler._process_with_scipdf(
        workspace_tmp_path / "sample.pdf",
        xml_path,
    ) == xml_path


def test_process_with_scipdf_raises_when_xml_is_missing(
    workspace_tmp_path: Path,
) -> None:
    """Checks that a successful-looking SciPDF call without XML is an error."""
    handler = build_handler_without_init()

    class FakeSciPdfParser:
        def process(self, pdf_path: Path, xml_save_path: Path) -> Path:
            return xml_save_path

    handler._scipdf_parser = FakeSciPdfParser()

    with pytest.raises(RuntimeError, match="SciPDF did not generate XML"):
        handler._process_with_scipdf(
            workspace_tmp_path / "sample.pdf",
            workspace_tmp_path / "missing.xml",
        )


def test_process_with_lightocr_reuses_existing_json(
    workspace_tmp_path: Path,
) -> None:
    """Checks that existing OCR JSON avoids loading the heavy LightOCR parser."""
    handler = build_handler_without_init()
    json_path = workspace_tmp_path / "sample.json"
    json_path.write_text("{}", encoding="utf-8")

    assert handler._process_with_lightocr(
        workspace_tmp_path / "sample.pdf",
        json_path,
    ) == json_path


def test_extract_values_reads_xml_and_lightocr_json() -> None:
    """Checks that XML and OCR artifacts are merged into extracted pipeline data."""
    handler = build_handler_without_init()
    xml_path = FIXTURES_DIR / "scipdf_xml" / "sample.xml"
    json_path = FIXTURES_DIR / "lightocr_json" / "sample.json"

    extracted_data = handler._extract_values(
        xml_path,
        json_path,
        "1802.09691",
    )

    assert extracted_data["arxiv_id"] == "1802.09691"
    assert extracted_data["title"] == "Fixture Model Paper"
    assert extracted_data["abstract"] == "Fixture abstract."
    assert extracted_data["authors"] == ["Jane Doe", "John Smith"]
    assert "MRR" in extracted_data["full_text"]
    assert "Experiments" in extracted_data["sections"]
    assert extracted_data["tables"] == json.loads(
        json_path.read_text(encoding="utf-8")
    )


def test_extract_values_omits_tables_when_lightocr_json_is_missing(
    workspace_tmp_path: Path,
) -> None:
    """Checks XML extraction still works when OCR JSON has not been generated."""
    handler = build_handler_without_init()
    xml_path = workspace_tmp_path / "sample.xml"
    shutil.copyfile(
        FIXTURES_DIR / "scipdf_xml" / "sample.xml",
        xml_path,
    )

    extracted_data = handler._extract_values(
        xml_path,
        workspace_tmp_path / "missing.json",
        "1802.09691",
    )

    assert extracted_data["title"] == "Fixture Model Paper"
    assert extracted_data["sections"] == (
        "Experiments We evaluate FixtureModel with MRR and Hits@10."
    )
    assert extracted_data["tables"] == []


def test_extract_values_keeps_full_text_when_target_sections_are_missing() -> None:
    """Checks no-target XML still provides full text for later extractor fallback."""
    handler = build_handler_without_init()
    xml_path = FIXTURES_DIR / "scipdf_xml" / "no_target_sections.xml"
    json_path = FIXTURES_DIR / "lightocr_json" / "empty.json"

    extracted_data = handler._extract_values(
        xml_path,
        json_path,
        "2401.01234",
    )

    assert extracted_data["title"] == "No Target Section Paper"
    assert extracted_data["authors"] == ["Ada Lovelace"]
    assert extracted_data["sections"] == ""
    assert "The fixture model is described here." in extracted_data["full_text"]
    assert extracted_data["tables"] == {
        "documents": []
    }


def test_handle_pdf_reuses_existing_pdf_and_saves_modelcard(
    workspace_tmp_path: Path,
) -> None:
    """Checks the happy PDFHandler flow with SciPDF, LightOCR, and card mocked."""
    handler = build_handler_without_init()
    handler._pdf_dir = workspace_tmp_path / "raw" / "pdfs"
    handler._xml_dir = workspace_tmp_path / "interim" / "scipdf_xml"
    handler._json_dir = workspace_tmp_path / "interim" / "lightocr_json"
    handler._modelcards_dir = workspace_tmp_path / "processed" / "modelcards"
    handler._pdf_dir.mkdir(parents=True)
    shutil.copyfile(
        FIXTURES_DIR / "pdfs" / "sample.pdf",
        handler._pdf_dir / "1802.09691.pdf",
    )

    class FakeLlama:
        def extract(self, text: str, question: str) -> list[str]:
            return ["FixtureModel"]

    class FakeSciPdfParser:
        def process(self, pdf_path: Path, xml_save_path: Path) -> Path:
            xml_save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                FIXTURES_DIR / "scipdf_xml" / "sample.xml",
                xml_save_path,
            )
            return xml_save_path

    calls = {
        "download": 0,
        "lightocr": 0,
        "modelcard": 0,
    }
    stages: list[dict] = []
    expected_card = {
        "@context": "https://w3id.org/fair4ml",
        "@id": "https://w3id.org/p2mc/model/1802.09691",
        "name": "FixtureModel",
    }

    def fake_download(pdf_url: str, save_path: Path) -> Path:
        calls["download"] += 1
        return save_path

    def fake_process_with_lightocr(
        pdf_path: Path,
        json_save_path: Path,
        on_progress=None,
    ) -> Path:
        calls["lightocr"] += 1
        json_save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            FIXTURES_DIR / "lightocr_json" / "sample.json",
            json_save_path,
        )
        return json_save_path

    def fake_generate_modelcard(extracted_data: dict) -> dict:
        calls["modelcard"] += 1
        assert extracted_data["title"] == "Fixture Model Paper"
        assert extracted_data["tables"]["documents"]
        return expected_card

    handler._llama = FakeLlama()
    handler._scipdf_parser = FakeSciPdfParser()
    handler._download_pdf = fake_download
    handler._process_with_lightocr = fake_process_with_lightocr
    handler._generate_modelcard = fake_generate_modelcard

    card = handler.handle_pdf(
        "https://arxiv.org/pdf/1802.09691.pdf",
        on_stage=stages.append,
    )

    assert card == expected_card
    assert calls == {
        "download": 0,
        "lightocr": 1,
        "modelcard": 1,
    }
    assert [
        stage["key"]
        for stage in stages
    ] == [
        "initializing",
        "using_existing_pdf",
        "extracting_xml",
        "extracting_tables",
        "extracting_values",
        "generating_modelcard",
        "saving_modelcard",
    ]
    assert json.loads(
        (
            handler._modelcards_dir
            / "1802.09691_modelcard.json"
        ).read_text(encoding="utf-8")
    ) == expected_card


def test_handle_pdf_downloads_missing_pdf_before_processing(
    workspace_tmp_path: Path,
) -> None:
    """Checks that PDFHandler downloads the PDF when it is not already stored."""
    handler = build_handler_without_init()
    handler._pdf_dir = workspace_tmp_path / "raw" / "pdfs"
    handler._xml_dir = workspace_tmp_path / "interim" / "scipdf_xml"
    handler._json_dir = workspace_tmp_path / "interim" / "lightocr_json"
    handler._modelcards_dir = workspace_tmp_path / "processed" / "modelcards"

    class FakeLlama:
        def extract(self, text: str, question: str) -> list[str]:
            return ["FixtureModel"]

    class FakeSciPdfParser:
        def process(self, pdf_path: Path, xml_save_path: Path) -> Path:
            assert pdf_path.is_file()
            xml_save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                FIXTURES_DIR / "scipdf_xml" / "sample.xml",
                xml_save_path,
            )
            return xml_save_path

    calls = {
        "download": 0,
    }

    def fake_download(pdf_url: str, save_path: Path) -> Path:
        calls["download"] += 1
        save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            FIXTURES_DIR / "pdfs" / "sample.pdf",
            save_path,
        )
        return save_path

    def fake_process_with_lightocr(
        pdf_path: Path,
        json_save_path: Path,
        on_progress=None,
    ) -> Path:
        json_save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            FIXTURES_DIR / "lightocr_json" / "sample.json",
            json_save_path,
        )
        return json_save_path

    handler._llama = FakeLlama()
    handler._scipdf_parser = FakeSciPdfParser()
    handler._download_pdf = fake_download
    handler._process_with_lightocr = fake_process_with_lightocr
    handler._generate_modelcard = lambda extracted_data: {"name": "FixtureModel"}

    card = handler.handle_pdf("https://arxiv.org/pdf/1802.09691.pdf")

    assert card == {
        "name": "FixtureModel",
    }
    assert calls["download"] == 1
    assert (handler._pdf_dir / "1802.09691.pdf").is_file()


def test_handle_pdf_without_detected_model_skips_lightocr_but_returns_error(
    workspace_tmp_path: Path,
) -> None:
    """Checks the current no-model branch: LightOCR is skipped but JSON setup fails."""
    handler = build_handler_without_init()
    handler._pdf_dir = workspace_tmp_path / "raw" / "pdfs"
    handler._xml_dir = workspace_tmp_path / "interim" / "scipdf_xml"
    handler._json_dir = workspace_tmp_path / "interim" / "lightocr_json"
    handler._modelcards_dir = workspace_tmp_path / "processed" / "modelcards"
    handler._pdf_dir.mkdir(parents=True)
    shutil.copyfile(
        FIXTURES_DIR / "pdfs" / "sample.pdf",
        handler._pdf_dir / "1802.09691.pdf",
    )

    class FakeLlama:
        def extract(self, text: str, question: str) -> list[str]:
            return []

    class FakeSciPdfParser:
        def process(self, pdf_path: Path, xml_save_path: Path) -> Path:
            xml_save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                FIXTURES_DIR / "scipdf_xml" / "sample.xml",
                xml_save_path,
            )
            return xml_save_path

    def fail_if_lightocr_runs(
        pdf_path: Path,
        json_save_path: Path,
        on_progress=None,
    ) -> Path:
        raise AssertionError("LightOCR should be skipped when no model is found")

    handler._llama = FakeLlama()
    handler._scipdf_parser = FakeSciPdfParser()
    handler._process_with_lightocr = fail_if_lightocr_runs
    handler._generate_modelcard = lambda extracted_data: {"name": "Unknown"}

    card = handler.handle_pdf("https://arxiv.org/pdf/1802.09691.pdf")

    assert card["failed_url"] == "https://arxiv.org/pdf/1802.09691.pdf"
    assert "lightocr_json" in card["error"]
