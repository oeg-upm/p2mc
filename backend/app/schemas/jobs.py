from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, HttpUrl, field_validator


ARXIV_URL_PATTERN = re.compile(
    r"^https?://(?:www\.|export\.)?arxiv\.org/"
    r"(?:abs|pdf)/"
    r"(?P<arxiv_id>\d{4}\.\d{4,5})"
    r"(?:v\d+)?"
    r"(?:\.pdf)?/?$"
)


def parse_arxiv_url(url: str) -> tuple[str, str]:
    match = ARXIV_URL_PATTERN.fullmatch(url.rstrip("/"))

    if match is None:
        raise ValueError(
            "The URL must be a valid arXiv paper URL, for example: "
            "https://arxiv.org/pdf/1802.09691.pdf"
        )

    arxiv_id = match.group("arxiv_id")
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return arxiv_id, pdf_url



class AskedJob(BaseModel):
    url: HttpUrl

    @field_validator("url")
    @classmethod
    def validate_arxiv_url(cls, url: HttpUrl) -> HttpUrl:
        _, pdf_url = parse_arxiv_url(str(url))

        return HttpUrl(pdf_url)
    
class PipelineStage(BaseModel):
    key: str
    label: str
    step: int | None = None
    total: int | None = None
    detail: str | None = None
    item_current: int | None = None
    item_total: int | None = None


class AskedJobResponse(BaseModel):
    job_id: str
    arxiv_id: str
    url: HttpUrl
    status: str
    created_at: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    error: StatusError | None = None
    artifacts: dict[str, str] | None = None
    card: dict[str, Any] | None = None
    pipeline_stage: PipelineStage | None = None


class StatusError(BaseModel):
    type: str
    message: str


class ArtifactContentResponse(BaseModel):
    job_id: str
    artifact_name: str
    media_type: str
    content: str


class StatusJobResponse(BaseModel):
    job_id: str
    source_type: Literal["arxiv", "upload"] = "arxiv"
    document_id: str | None = None

    arxiv_id: str | None = None
    url: HttpUrl | None = None

    original_filename: str | None = None
    stored_filename: str | None = None
    pdf_path: str | None = None
    size_bytes: int | None = None

    status: str
    created_at: str
    started_at: str | None = None
    updated_at: str
    completed_at: str | None = None

    error: StatusError | None = None
    artifacts: dict[str, str] | None = None
    card: dict[str, Any] | None = None
    pipeline_stage: PipelineStage | None = None


class JobSummaryResponse(BaseModel):
    job_id: str
    source_type: Literal["arxiv", "upload"] = "arxiv"
    document_id: str | None = None

    arxiv_id: str | None = None
    url: HttpUrl | None = None

    original_filename: str | None = None
    stored_filename: str | None = None
    pdf_path: str | None = None
    size_bytes: int | None = None

    status: str
    created_at: str
    started_at: str | None = None
    updated_at: str
    completed_at: str | None = None

    error: StatusError | None = None
    pipeline_stage: PipelineStage | None = None


class JobsListResponse(BaseModel):
    jobs: list[JobSummaryResponse]


class UploadedPDFResponse(BaseModel):
    job_id: str
    document_id: str
    original_filename: str
    stored_filename: str
    pdf_path: str
    size_bytes: int
    status: str