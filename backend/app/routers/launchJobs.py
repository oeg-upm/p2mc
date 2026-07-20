import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from backend import DATA_DIR
from backend.app.schemas import (
    AskedJob,
    AskedJobResponse,
    StatusJobResponse,
    parse_arxiv_url,
)
from backend.rabbitmq import RabbitMQPublishError, publish_job


router = APIRouter()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_status_path(job_id: str) -> Path:
    return DATA_DIR / "jobs" / job_id / "status.json"


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


@router.post("/", response_model=AskedJobResponse, status_code=202)
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


@router.get("/job-status/{job_id}", response_model=StatusJobResponse)
def get_job_status(job_id: str) -> StatusJobResponse:
    status_path = get_status_path(str(job_id))

    if not status_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    with status_path.open("r", encoding="utf-8") as file:
        return StatusJobResponse.model_validate(json.load(file))
