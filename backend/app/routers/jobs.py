import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import ValidationError

from backend import DATA_DIR
from backend.app.schemas import (
    AskedJob,
    AskedJobResponse,
    ArtifactContentResponse,
    JobsListResponse,
    JobSummaryResponse,
    StatusJobResponse,
    UploadedPDFResponse,
    parse_arxiv_url,
)
from backend.rabbitmq import RabbitMQPublishError, publish_job


router = APIRouter()
PIPELINE_TOTAL_STEPS = 7

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024

VIEWABLE_ARTIFACTS = {
    "xml": "application/xml",
    "lightocr_json": "application/json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_pipeline_stage(
    key: str,
    label: str,
    step: int | None = None,
    *,
    detail: str | None = None,
) -> dict:
    stage = {
        "key": key,
        "label": label,
    }

    if step is not None:
        stage["step"] = step
        stage["total"] = PIPELINE_TOTAL_STEPS

    if detail is not None:
        stage["detail"] = detail

    return stage


def get_status_path(job_id: str) -> Path:
    return DATA_DIR / "jobs" / job_id / "status.json"


def get_jobs_dir() -> Path:
    return DATA_DIR / "jobs"


def get_artifact_paths(arxiv_id: str) -> dict[str, Path]:
    return {
        "pdf": DATA_DIR / "raw" / "pdfs" / f"{arxiv_id}.pdf",
        "xml": (
            DATA_DIR
            / "interim"
            / "scipdf_xml"
            / f"{arxiv_id}.xml"
        ),
        "lightocr_json": (
            DATA_DIR
            / "interim"
            / "lightocr_json"
            / f"{arxiv_id}.json"
        ),
        "modelcard": (
            DATA_DIR
            / "processed"
            / "modelcards"
            / f"{arxiv_id}_modelcard.json"
        ),
    }


def serialize_existing_artifact_paths(
    artifact_paths: dict[str, Path],
) -> dict[str, str]:
    data_root = DATA_DIR.resolve()
    return {
        name: path.relative_to(data_root).as_posix()
        for name, path in artifact_paths.items()
        if path.is_file()
    }


def read_existing_modelcard(
    artifact_paths: dict[str, Path],
) -> dict | None:
    modelcard_path = artifact_paths["modelcard"]

    if not modelcard_path.is_file():
        return None

    try:
        with modelcard_path.open("r", encoding="utf-8") as file:
            card = json.load(file)
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(card, dict):
        return None

    return card


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
        source_type=status.source_type,
        document_id=status.document_id,
        arxiv_id=status.arxiv_id,
        url=status.url,
        original_filename=status.original_filename,
        stored_filename=status.stored_filename,
        pdf_path=status.pdf_path,
        size_bytes=status.size_bytes,
        status=status.status,
        created_at=status.created_at,
        started_at=status.started_at,
        updated_at=status.updated_at,
        completed_at=status.completed_at,
        error=status.error,
        pipeline_stage=status.pipeline_stage,
    )


def resolve_artifact_path(
    status: StatusJobResponse,
    artifact_name: str,
) -> Path:
    if artifact_name not in VIEWABLE_ARTIFACTS:
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


def create_initial_status_json(
    job_id: str,
    source_type: str,
    status: str,
    *,
    document_id: str | None = None,
    arxiv_id: str | None = None,
    url: str | None = None,
    original_filename: str | None = None,
    stored_filename: str | None = None,
    pdf_path: str | None = None,
) -> dict:
    created_at = utc_now()

    stage_label = {
        "queued": "Queued",
        "uploaded": "PDF uploaded",
    }.get(
        status,
        status.replace("_", " ").title(),
    )

    status_data = {
        "job_id": job_id,
        "source_type": source_type,
        "document_id": document_id,
        "arxiv_id": arxiv_id,
        "url": url,
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "pdf_path": pdf_path,
        "status": status,
        "created_at": created_at,
        "started_at": None,
        "updated_at": created_at,
        "completed_at": None,
        "error": None,
        "artifacts": None,
        "card": None,
        "pipeline_stage": build_pipeline_stage(
            status,
            stage_label,
        ),
    }

    write_status_atomic(
        get_status_path(job_id),
        status_data,
    )

    return status_data


def build_cached_completed_status(
    job_id: str,
    arxiv_id: str,
    pdf_url: str,
    card: dict,
    artifact_paths: dict[str, Path],
) -> dict:
    completed_at = utc_now()

    return {
        "job_id": job_id,
        "arxiv_id": arxiv_id,
        "url": pdf_url,
        "status": "completed",
        "created_at": completed_at,
        "started_at": None,
        "updated_at": completed_at,
        "completed_at": completed_at,
        "error": None,
        "artifacts": serialize_existing_artifact_paths(artifact_paths),
        "card": card,
        "pipeline_stage": build_pipeline_stage(
            "completed",
            "Using existing ModelCard",
            PIPELINE_TOTAL_STEPS,
            detail="Paper already processed.",
        ),
    }


@router.post("/launch-job", response_model=AskedJobResponse, status_code=202)
def launch_job(asked_job: AskedJob):
    arxiv_id, pdf_url = parse_arxiv_url(str(asked_job.url))
    job_id = str(uuid4())
    status_path = get_status_path(job_id)
    artifact_paths = get_artifact_paths(arxiv_id)
    cached_card = read_existing_modelcard(artifact_paths)

    if cached_card is not None:
        completed_status = build_cached_completed_status(
            job_id,
            arxiv_id,
            pdf_url,
            cached_card,
            artifact_paths,
        )
        write_status_atomic(status_path, completed_status)
        return AskedJobResponse(**completed_status)

    queued_status = create_initial_status_json(
    job_id,
    "arxiv",
    "queued",
    document_id=arxiv_id,
    arxiv_id=arxiv_id,
    url=pdf_url,
)

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
                "pipeline_stage": build_pipeline_stage(
                    "failed",
                    "Failed to enqueue job",
                    detail=str(exc),
                ),
            },
        )
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return AskedJobResponse(**queued_status)



@router.post("/upload-pdf",response_model=UploadedPDFResponse,status_code=201)
async def upload_pdf(file: UploadFile = File(...),) -> UploadedPDFResponse:

    original_filename = Path(file.filename or "uploaded.pdf").name

    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400,detail="Only PDF files are accepted.",)

    upload_dir = DATA_DIR / "raw" / "pdfs"
    upload_dir.mkdir(parents=True, exist_ok=True)

    job_id = str(uuid4())
    temporary_path = upload_dir / f".{job_id}.upload"

    digest = hashlib.sha256()
    size_bytes = 0
    first_chunk = True

    try:
        with temporary_path.open("wb") as output_file:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                if first_chunk:
                    first_chunk = False

                    if not chunk.startswith(b"%PDF-"):
                        raise HTTPException(status_code=400,detail=(
                            "The uploaded file is not ""a valid PDF."
                            )
                        )

                size_bytes += len(chunk)

                if size_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413,detail=(
                            "The PDF exceeds the ""50 MB upload limit."
                        )
                    )

                digest.update(chunk)
                output_file.write(chunk)

        if size_bytes == 0:
            raise HTTPException(status_code=400,
                                detail="The uploaded PDF is empty."
            )

        document_id = f"upload-{digest.hexdigest()}"
        numeric_id = int(digest.hexdigest(), 16) % 100_000_000

        paper_id = (
            f"{9000 + numeric_id // 100_000:04d}."
            f"{numeric_id % 100_000:05d}"
        )

        pdf_url = f"https://upload.p2mc.local/{paper_id}"

        final_filename = f"{paper_id}.pdf"
        final_path = upload_dir / final_filename

        os.replace(temporary_path, final_path)

        pdf_path = str(final_path.relative_to(DATA_DIR))

        queued_status = create_initial_status_json(
            job_id,
            "upload",
            "queued",
            document_id=document_id,
            arxiv_id=paper_id,
            url=pdf_url,
            original_filename=original_filename,
            stored_filename=final_filename,
            pdf_path=pdf_path,
        )

        try:
            publish_job(pdf_url, job_id, paper_id)

        except RabbitMQPublishError as exc:
            failed_at = utc_now()

            write_status_atomic(
                get_status_path(job_id),
                {
                    **queued_status,
                    "status": "failed",
                    "updated_at": failed_at,
                    "completed_at": failed_at,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "pipeline_stage": build_pipeline_stage(
                        "failed",
                        "Failed to enqueue job",
                        detail=str(exc),
                    ),
                },
            )

            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc


    finally:
        temporary_path.unlink(missing_ok=True)
        await file.close()

    return UploadedPDFResponse(
    job_id=job_id,
    document_id=document_id,
    original_filename=original_filename,
    stored_filename=final_filename,
    pdf_path=str(final_path.relative_to(DATA_DIR)),
    size_bytes=size_bytes,
    status=queued_status["status"]
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


@router.get(
    "/{job_id}/artifacts/{artifact_name}",
    response_model=ArtifactContentResponse,
)
def get_artifact_content(
    job_id: str,
    artifact_name: str,
) -> ArtifactContentResponse:
    status = get_existing_job_status(job_id)
    artifact_path = resolve_artifact_path(status, artifact_name)

    try:
        content = artifact_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Artifact could not be read",
        ) from exc

    return ArtifactContentResponse(
        job_id=job_id,
        artifact_name=artifact_name,
        media_type=VIEWABLE_ARTIFACTS[artifact_name],
        content=content,
    )
