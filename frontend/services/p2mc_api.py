from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx


API_URL = os.getenv("P2MC_API_URL", "http://localhost:8000").rstrip("/")
LAUNCH_JOB_PATH = "/job/launch-job"
UPLOAD_PDF_PATH = "/job/upload-pdf"
JOBS_LIST_PATH = "/job/jobs"
JOB_STATUS_PATH = "/job/job-status/{job_id}"
ARTIFACT_PATH = "/job/{job_id}/artifacts/{artifact_name}"


class P2MCAPIError(RuntimeError):
    pass


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    detail = payload.get("detail") if isinstance(payload, dict) else None
    return str(detail or response.text)


def launch_job(url: str) -> dict[str, Any]:
    try:
        response = httpx.post(
            f"{API_URL}{LAUNCH_JOB_PATH}",
            json={"url": url},
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as exc:
        raise P2MCAPIError(_extract_error_detail(exc.response)) from exc

    except httpx.RequestError as exc:
        raise P2MCAPIError(
            "Could not connect to the P2MC API."
        ) from exc

def upload_pdf(
    uploaded_file: Any,
) -> dict[str, Any]:
    filename = getattr(
        uploaded_file,
        "name",
        "uploaded.pdf",
    )

    content_type = (
        getattr(uploaded_file, "type", None)
        or "application/pdf"
    )

    files = {
        "file": (
            filename,
            uploaded_file.getvalue(),
            content_type,
        ),
    }

    try:
        response = httpx.post(
            f"{API_URL}{UPLOAD_PDF_PATH}",
            files=files,
            timeout=60.0,
        )

        response.raise_for_status()

        return response.json()

    except httpx.HTTPStatusError as exc:
        raise P2MCAPIError(
            _extract_error_detail(exc.response)
        ) from exc

    except httpx.RequestError as exc:
        raise P2MCAPIError(
            "Could not connect to the P2MC API."
        ) from exc


def get_job_status(job_id: str) -> dict[str, Any]:
    try:
        response = httpx.get(
            f"{API_URL}{JOB_STATUS_PATH.format(job_id=job_id)}",
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as exc:
        raise P2MCAPIError(_extract_error_detail(exc.response)) from exc

    except httpx.RequestError as exc:
        raise P2MCAPIError(
            "Could not connect to the P2MC API."
        ) from exc


def list_jobs() -> dict[str, Any]:
    try:
        response = httpx.get(
            f"{API_URL}{JOBS_LIST_PATH}",
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as exc:
        raise P2MCAPIError(_extract_error_detail(exc.response)) from exc

    except httpx.RequestError as exc:
        raise P2MCAPIError(
            "Could not connect to the P2MC API."
        ) from exc


def get_artifact_content(
    job_id: str,
    artifact_name: str,
) -> dict[str, Any]:
    quoted_job_id = quote(job_id, safe="")
    quoted_artifact_name = quote(artifact_name, safe="")

    path = ARTIFACT_PATH.format(
        job_id=quoted_job_id,
        artifact_name=quoted_artifact_name,
    )

    try:
        response = httpx.get(
            f"{API_URL}{path}",
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as exc:
        raise P2MCAPIError(_extract_error_detail(exc.response)) from exc

    except httpx.RequestError as exc:
        raise P2MCAPIError(
            "Could not connect to the P2MC API."
        ) from exc
