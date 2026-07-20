from __future__ import annotations

import json
import time
from typing import Any

import streamlit as st

from services.p2mc_api import (
    P2MCAPIError,
    get_artifact_content,
    get_job_status,
)


VIEWABLE_ARTIFACTS = {
    "xml": ("SciPDF XML", "xml"),
    "lightocr_json": ("LightOCR JSON", "json"),
}
ACTIVE_STATUSES = {"queued", "processing"}
AUTO_REFRESH_SECONDS = 3


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


def _progress_ratio(
    current: Any,
    total: Any,
) -> float | None:
    try:
        current_number = float(current)
        total_number = float(total)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if total_number <= 0:
        return None

    return min(
        max(current_number / total_number, 0.0),
        1.0,
    )


def render_pipeline_stage(stage: Any) -> None:
    if not isinstance(stage, dict):
        return

    label = stage.get("label") or stage.get("key")
    step = stage.get("step")
    total = stage.get("total")
    step_ratio = _progress_ratio(step, total)

    if step_ratio is None:
        st.write(f"**Pipeline:** {label}")
    else:
        st.write(f"**Pipeline:** Step {step}/{total}: {label}")
        st.progress(step_ratio)

    detail = stage.get("detail")
    if detail:
        st.caption(str(detail))

    item_ratio = _progress_ratio(
        stage.get("item_current"),
        stage.get("item_total"),
    )
    if item_ratio is not None:
        st.progress(item_ratio)


def render_artifacts(job_id: str, artifacts: dict[str, str]) -> None:
    if not artifacts:
        return

    viewable_artifacts = {
        name: path
        for name, path in artifacts.items()
        if name in VIEWABLE_ARTIFACTS
    }

    if not viewable_artifacts:
        return

    st.write("**Artifacts:**")

    for artifact_name, artifact_path in viewable_artifacts.items():
        label, language = VIEWABLE_ARTIFACTS[artifact_name]
        artifact_col, action_col = st.columns([3, 1])

        with artifact_col:
            st.write(f"**{label}:** `{artifact_path}`")

        show_artifact = False
        with action_col:
            show_artifact = st.button(
                f"Show {label}",
                key=f"{job_id}_{artifact_name}_show",
                width="stretch",
            )

        if show_artifact:
            try:
                with st.spinner(f"Loading {label}..."):
                    artifact = get_artifact_content(job_id, artifact_name)
            except P2MCAPIError as exc:
                st.error(str(exc))
                continue

            content = str(artifact.get("content", ""))
            if language == "json":
                try:
                    content = json.dumps(
                        json.loads(content),
                        indent=2,
                        ensure_ascii=False,
                    )
                except ValueError:
                    pass

            with st.container(height=400):
                st.code(content, language=language)


def render_card(job: dict[str, Any], card_button_key: str) -> None:
    card = job.get("card")

    if not card:
        return

    if not st.button(
        "Show ModelCard",
        key=card_button_key,
        width="stretch",
    ):
        return

    card_json = json.dumps(card, indent=2, ensure_ascii=False)

    st.write("**ModelCard JSON-LD:**")
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
        auto_refresh_enabled = True

        if (
            str(current_job.get("status", "unknown")) in ACTIVE_STATUSES
            and job_id
        ):
            try:
                current_job = get_job_status(job_id)
            except P2MCAPIError as exc:
                auto_refresh_enabled = False
                st.error(str(exc))

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
        render_pipeline_stage(current_job.get("pipeline_stage"))

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
        if str(current_job.get("status", "unknown")) == "completed":
            render_artifacts(job_id, current_job.get("artifacts") or {})
            render_card(current_job, f"{job_id}_modelcard_show")

        if (
            auto_refresh_enabled
            and str(current_job.get("status", "unknown")) in ACTIVE_STATUSES
        ):
            time.sleep(AUTO_REFRESH_SECONDS)
            st.rerun()

    return current_job
