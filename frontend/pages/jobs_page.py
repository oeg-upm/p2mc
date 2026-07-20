from __future__ import annotations

import streamlit as st

from components.job_status import render_job_status_panel
from services.p2mc_api import P2MCAPIError, get_job_status, list_jobs


if "jobs_list" not in st.session_state:
    st.session_state.jobs_list = None

if "jobs_selected_job_id" not in st.session_state:
    st.session_state.jobs_selected_job_id = None

if "jobs_selected_status" not in st.session_state:
    st.session_state.jobs_selected_status = None


def load_jobs() -> None:
    with st.spinner("Loading jobs..."):
        st.session_state.jobs_list = list_jobs().get("jobs", [])


def job_label(job: dict) -> str:
    status = job.get("status", "unknown")
    stage = job.get("pipeline_stage") or {}
    stage_label = stage.get("label") if isinstance(stage, dict) else None
    arxiv_id = job.get("arxiv_id", "-")
    updated_at = job.get("updated_at", "-")
    job_id = str(job.get("job_id", ""))

    if stage_label:
        return f"{status} | {stage_label} | {arxiv_id} | {job_id[:8]}"

    return f"{status} | {arxiv_id} | {updated_at} | {job_id[:8]}"


st.title("Jobs", anchor=False)

refresh_col, _ = st.columns([1, 3])

with refresh_col:
    if st.button("Refresh jobs", type="primary", width="stretch"):
        try:
            load_jobs()
        except P2MCAPIError as exc:
            st.error(str(exc))

if st.session_state.jobs_list is None:
    try:
        load_jobs()
    except P2MCAPIError as exc:
        st.error(str(exc))
        st.stop()

jobs = st.session_state.jobs_list or []

if not jobs:
    st.info("No jobs found yet.")
    st.stop()

st.dataframe(
    [
        {
            "Stage": (
                job.get("pipeline_stage", {}).get("label")
                if isinstance(job.get("pipeline_stage"), dict)
                else None
            ),
            "Status": job.get("status"),
            "arXiv ID": job.get("arxiv_id"),
            "Updated": job.get("updated_at"),
            "Completed": job.get("completed_at"),
            "Job ID": job.get("job_id"),
        }
        for job in jobs
    ],
    hide_index=True,
    use_container_width=True,
)

job_ids = [str(job["job_id"]) for job in jobs]
jobs_by_id = {str(job["job_id"]): job for job in jobs}

current_index = 0
if st.session_state.jobs_selected_job_id in job_ids:
    current_index = job_ids.index(st.session_state.jobs_selected_job_id)

selected_job_id = st.selectbox(
    "Select a job",
    job_ids,
    index=current_index,
    format_func=lambda job_id: job_label(jobs_by_id[job_id]),
)

if selected_job_id != st.session_state.jobs_selected_job_id:
    st.session_state.jobs_selected_job_id = selected_job_id
    st.session_state.jobs_selected_status = None

if (
    st.session_state.jobs_selected_status is None
    or st.session_state.jobs_selected_status.get("job_id") != selected_job_id
):
    try:
        with st.spinner("Loading selected job..."):
            st.session_state.jobs_selected_status = get_job_status(
                selected_job_id
            )
    except P2MCAPIError as exc:
        st.error(str(exc))
        st.stop()

st.session_state.jobs_selected_status = render_job_status_panel(
    st.session_state.jobs_selected_status,
    refresh_button_key=f"jobs_refresh_{selected_job_id}",
)
