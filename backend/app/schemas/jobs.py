import re
from typing import Any

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
    
    
    
    

class AskedJobResponse(BaseModel):
    job_id: str
    arxiv_id: str
    url: HttpUrl
    status: str


class StatusError(BaseModel):
    type: str
    message: str


class StatusJobResponse(BaseModel):
    job_id: str
    arxiv_id: str
    url: HttpUrl
    status: str
    created_at: str
    started_at: str | None = None
    updated_at: str
    completed_at: str | None = None
    error: StatusError | None = None
    artifacts: dict[str, str] | None = None
    card: dict[str, Any] | None = None
