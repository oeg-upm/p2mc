"""Tests for worker job processing and status files.

Planned scope:
- RabbitMQ message validation;
- status transitions for queued, processing, completed, and failed jobs;
- cached ModelCard reuse;
- artifact path serialization;
- PDFHandler error propagation.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from backend import worker


def test_validate_message_accepts_valid_job_message() -> None:
    """Checks that valid RabbitMQ payloads are normalized for processing."""
    job_id = str(uuid4())
    message = {
        "job_id": job_id,
        "arxiv_id": "1802.09691",
        "url": "https://arxiv.org/pdf/1802.09691.pdf",
    }

    assert worker.validate_message(message) == (
        job_id,
        "1802.09691",
        "https://arxiv.org/pdf/1802.09691.pdf",
    )


def test_validate_message_rejects_invalid_job_id() -> None:
    """Checks that invalid job UUIDs are rejected before status files are used."""
    with pytest.raises(ValueError):
        worker.validate_message(
            {
                "job_id": "not-a-uuid",
                "arxiv_id": "1802.09691",
                "url": "https://arxiv.org/pdf/1802.09691.pdf",
            }
        )


def test_validate_message_rejects_invalid_arxiv_id() -> None:
    """Checks that malformed arXiv IDs cannot reach PDFHandler paths."""
    with pytest.raises(ValueError, match="Invalid base arXiv ID"):
        worker.validate_message(
            {
                "job_id": str(uuid4()),
                "arxiv_id": "not-an-arxiv-id",
                "url": "https://arxiv.org/pdf/1802.09691.pdf",
            }
        )


def test_validate_message_rejects_non_http_urls() -> None:
    """Checks that worker messages cannot point to unsupported URL schemes."""
    with pytest.raises(ValueError, match="Invalid URL"):
        worker.validate_message(
            {
                "job_id": str(uuid4()),
                "arxiv_id": "1802.09691",
                "url": "file:///tmp/paper.pdf",
            }
        )


def test_write_status_atomic_and_read_status_round_trip(
    workspace_tmp_path: Path,
) -> None:
    """Checks that status JSON is written atomically and read back as a dict."""
    status_path = workspace_tmp_path / "jobs" / str(uuid4()) / "status.json"
    status_data = {
        "job_id": "sample",
        "status": "queued",
    }

    worker.write_status_atomic(status_path, status_data)

    assert worker.read_status(status_path) == status_data


def test_get_artifact_paths_uses_data_root(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmp_path: Path,
) -> None:
    """Checks that worker artifact paths are built under the configured data root."""
    monkeypatch.setattr(worker, "DATA_ROOT", workspace_tmp_path)

    paths = worker.get_artifact_paths("1802.09691")

    assert (
        paths["pdf"]
        == workspace_tmp_path / "raw" / "pdfs" / "1802.09691.pdf"
    )
    assert (
        paths["modelcard"]
        == workspace_tmp_path
        / "processed"
        / "modelcards"
        / "1802.09691_modelcard.json"
    )


def test_serialize_existing_artifact_paths_keeps_only_existing_files(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmp_path: Path,
) -> None:
    """Checks that API artifacts expose only files present on disk."""
    monkeypatch.setattr(worker, "DATA_ROOT", workspace_tmp_path)
    xml_path = workspace_tmp_path / "interim" / "scipdf_xml" / "sample.xml"
    missing_json_path = (
        workspace_tmp_path / "interim" / "lightocr_json" / "sample.json"
    )
    xml_path.parent.mkdir(parents=True)
    xml_path.write_text("<TEI/>", encoding="utf-8")

    assert worker.serialize_existing_artifact_paths(
        {
            "xml": xml_path,
            "lightocr_json": missing_json_path,
        }
    ) == {
        "xml": "interim/scipdf_xml/sample.xml",
    }


def test_read_existing_modelcard_returns_card_when_json_is_valid(
    workspace_tmp_path: Path,
) -> None:
    """Checks that cached ModelCards can be reused by worker jobs."""
    modelcard_path = workspace_tmp_path / "modelcard.json"
    card = {
        "@context": "https://w3id.org/fair4ml",
        "name": "FixtureModel",
    }
    modelcard_path.write_text(json.dumps(card), encoding="utf-8")

    assert worker.read_existing_modelcard({"modelcard": modelcard_path}) == card


def test_read_existing_modelcard_returns_none_for_invalid_json(
    workspace_tmp_path: Path,
) -> None:
    """Checks that corrupt cached ModelCards are ignored safely."""
    modelcard_path = workspace_tmp_path / "modelcard.json"
    modelcard_path.write_text("{not-json", encoding="utf-8")

    assert worker.read_existing_modelcard({"modelcard": modelcard_path}) is None


def test_verify_artifacts_raises_when_expected_file_is_missing(
    workspace_tmp_path: Path,
) -> None:
    """Checks that completed jobs fail if required artifacts were not produced."""
    existing_path = workspace_tmp_path / "sample.xml"
    missing_path = workspace_tmp_path / "missing.json"
    existing_path.write_text("<TEI/>", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing.json"):
        worker.verify_artifacts(
            {
                "xml": existing_path,
                "lightocr_json": missing_path,
            }
        )


def test_process_job_marks_status_failed_when_pdf_handler_returns_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmp_path: Path,
) -> None:
    """Checks that PDFHandler errors are persisted as failed job status."""
    monkeypatch.setattr(worker, "DATA_ROOT", workspace_tmp_path)
    job_id = str(uuid4())
    message = {
        "job_id": job_id,
        "arxiv_id": "1802.09691",
        "url": "https://arxiv.org/pdf/1802.09691.pdf",
    }

    class FailingPDFHandler:
        def handle_pdf(self, url: str, on_stage=None) -> dict:
            if on_stage is not None:
                on_stage(
                    worker.build_pipeline_stage(
                        "extracting_xml",
                        "Extracting XML with GROBID",
                        3,
                    )
                )

            return {
                "error": "boom",
            }

    worker.process_job(message, FailingPDFHandler())

    status = worker.read_status(worker.get_status_path(job_id))
    assert status is not None
    assert status["status"] == "failed"
    assert status["error"] == {
        "type": "RuntimeError",
        "message": "boom",
    }
    assert status["pipeline_stage"] == {
        "key": "failed",
        "label": "Failed",
        "step": 3,
        "total": 7,
        "detail": "Extracting XML with GROBID",
    }
