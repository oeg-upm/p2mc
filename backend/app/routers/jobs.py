import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError

from backend import DATA_DIR
from backend.app.schemas import (
    AskedJob,
    AskedJobResponse,
    JobsListResponse,
    JobSummaryResponse,
    StatusJobResponse,
    parse_arxiv_url,
)
from backend.rabbitmq import RabbitMQPublishError, publish_job


router = APIRouter()
ALLOWED_ARTIFACTS = {
    "pdf": "application/pdf",
    "xml": "application/xml",
    "lightocr_json": "application/json",
    "modelcard": "application/ld+json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_status_path(job_id: str) -> Path:
    return DATA_DIR / "jobs" / job_id / "status.json"


def get_jobs_dir() -> Path:
    return DATA_DIR / "jobs"


def read_status_file(status_path: Path) -> StatusJobResponse:
    with status_path.open("r", encoding="utf-8") as file:
        return StatusJobResponse.model_validate(json.load(file))


def get_existing_job_status(job_id: str) -> StatusJobResponse:
    status_path = get_status_path(str(job_id))

    if not status_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return read_status_file(status_path)


def build_job_summary(status: StatusJobResponse) -> JobSummaryResponse:
    return JobSummaryResponse(
        job_id=status.job_id,
        arxiv_id=status.arxiv_id,
        url=status.url,
        status=status.status,
        created_at=status.created_at,
        started_at=status.started_at,
        updated_at=status.updated_at,
        completed_at=status.completed_at,
        error=status.error,
    )


def resolve_artifact_path(
    status: StatusJobResponse,
    artifact_name: str,
) -> Path:
    if artifact_name not in ALLOWED_ARTIFACTS:
        raise HTTPException(
            status_code=404,
            detail="Artifact not found",
        )

    artifacts = status.artifacts or {}
    relative_path = artifacts.get(artifact_name)

    if relative_path is None:
        raise HTTPException(
            status_code=404,
            detail="Artifact not available",
        )

    data_root = DATA_DIR.resolve()
    artifact_path = (data_root / relative_path).resolve()

    try:
        artifact_path.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail="Artifact not found",
        ) from exc

    if not artifact_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Artifact file not found",
        )

    return artifact_path


def write_status_atomic(
    status_path: Path,
    status_data: dict,
) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = status_path.with_name(
        f".{status_path.name}.{os.getpid()}.tmp"
    )

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                status_data,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, status_path)

    finally:
        temporary_path.unlink(missing_ok=True)


def build_queued_status(
    job_id: str,
    arxiv_id: str,
    pdf_url: str,
) -> dict:
    created_at = utc_now()

    return {
        "job_id": job_id,
        "arxiv_id": arxiv_id,
        "url": pdf_url,
        "status": "queued",
        "created_at": created_at,
        "started_at": None,
        "updated_at": created_at,
        "completed_at": None,
        "error": None,
        "artifacts": None,
        "card": None,
    }


@router.post("/launch-job", response_model=AskedJobResponse, status_code=202)
def launch_job(asked_job: AskedJob):
    arxiv_id, pdf_url = parse_arxiv_url(str(asked_job.url))
    job_id = str(uuid4())
    status_path = get_status_path(job_id)
    queued_status = build_queued_status(job_id, arxiv_id, pdf_url)

    write_status_atomic(status_path, queued_status)

    try:
        publish_job(pdf_url, job_id, arxiv_id)

    except RabbitMQPublishError as exc:
        failed_at = utc_now()
        write_status_atomic(
            status_path,
            {
                **queued_status,
                "status": "failed",
                "updated_at": failed_at,
                "completed_at": failed_at,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
        )
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return AskedJobResponse(
        job_id=job_id,
        arxiv_id=arxiv_id,
        url=pdf_url,
        status=queued_status["status"],
    )


@router.get("/jobs", response_model=JobsListResponse)
def list_jobs() -> JobsListResponse:
    jobs_dir = get_jobs_dir()

    if not jobs_dir.is_dir():
        return JobsListResponse(jobs=[])

    summaries = []

    for status_path in jobs_dir.glob("*/status.json"):
        try:
            status = read_status_file(status_path)
        except (
            OSError,
            json.JSONDecodeError,
            ValidationError,
        ):
            continue

        summaries.append(build_job_summary(status))

    summaries.sort(
        key=lambda summary: summary.updated_at,
        reverse=True,
    )

    return JobsListResponse(jobs=summaries)


@router.get("/job-status/{job_id}", response_model=StatusJobResponse)
def get_job_status(job_id: str) -> StatusJobResponse:
    return get_existing_job_status(job_id)

@router.get("/{job_id}/artifacts/{artifact_name}")
def download_artifact(
    job_id: str,
    artifact_name: str,
) -> FileResponse:
    status = get_existing_job_status(job_id)
    artifact_path = resolve_artifact_path(status, artifact_name)

    return FileResponse(
        artifact_path,
        media_type=ALLOWED_ARTIFACTS[artifact_name],
        filename=artifact_path.name,
    )
