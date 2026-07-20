from __future__ import annotations

import json
from typing import Any

import streamlit as st

from services.p2mc_api import (
    P2MCAPIError,
    artifact_download_url,
    get_job_status,
)


ARTIFACT_LABELS = {
    "pdf": "PDF",
    "xml": "SciPDF XML",
    "lightocr_json": "LightOCR JSON",
    "modelcard": "ModelCard JSON-LD",
}


def render_status(status: str) -> None:
    if status == "completed":
        st.success("Completed")
    elif status == "failed":
        st.error("Failed")
    elif status == "processing":
        st.info("Processing")
    elif status == "queued":
        st.info("Queued")
    else:
        st.warning(f"Status: {status}")


def render_error(error: Any) -> None:
    if not error:
        return

    if isinstance(error, dict):
        error_type = error.get("type", "Error")
        error_message = error.get("message", str(error))
    else:
        error_type = "Error"
        error_message = str(error)

    st.error(f"{error_type}: {error_message}")


def render_artifacts(job_id: str, artifacts: dict[str, str]) -> None:
    if not artifacts:
        return

    st.write("**Artifacts:**")

    for artifact_name, artifact_path in artifacts.items():
        label = ARTIFACT_LABELS.get(artifact_name, artifact_name)
        artifact_col, download_col = st.columns([3, 1])

        with artifact_col:
            st.write(f"**{label}:** `{artifact_path}`")

        with download_col:
            st.link_button(
                "Download",
                url=artifact_download_url(job_id, artifact_name),
                width="stretch",
            )


def render_card(job: dict[str, Any], refresh_button_key: str) -> None:
    card = job.get("card")

    if not card:
        return

    card_json = json.dumps(card, indent=2, ensure_ascii=False)
    file_name = f"{job.get('arxiv_id', 'modelcard')}_modelcard.json"

    st.download_button(
        "Download returned ModelCard",
        data=card_json,
        file_name=file_name,
        mime="application/ld+json",
        key=f"{refresh_button_key}_returned_card",
    )

    with st.container(height=400):
        st.code(card_json, language="json")


def render_job_status_panel(
    job: dict[str, Any],
    *,
    refresh_button_key: str,
) -> dict[str, Any]:
    current_job = job
    job_id = str(current_job.get("job_id", ""))

    with st.container(border=True):
        st.subheader("Job status", anchor=False)

        job_col, refresh_col = st.columns([3, 1])

        with job_col:
            st.write(f"**Job ID:** `{job_id}`")

        with refresh_col:
            if st.button(
                "Refresh status",
                type="primary",
                width="stretch",
                key=refresh_button_key,
            ):
                try:
                    with st.spinner("Refreshing status..."):
                        current_job = get_job_status(job_id)

                except P2MCAPIError as exc:
                    st.error(str(exc))

        render_status(str(current_job.get("status", "unknown")))

        st.write(f"**arXiv ID:** {current_job.get('arxiv_id', '-')}")
        st.write(f"**PDF URL:** {current_job.get('url', '-')}")

        for label, key in [
            ("Created", "created_at"),
            ("Started", "started_at"),
            ("Updated", "updated_at"),
            ("Completed", "completed_at"),
        ]:
            value = current_job.get(key)
            if value:
                st.caption(f"{label}: {value}")

        render_error(current_job.get("error"))
        render_artifacts(job_id, current_job.get("artifacts") or {})
        render_card(current_job, refresh_button_key)

    return current_job
