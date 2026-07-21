from __future__ import annotations

import streamlit as st

from components.job_status import render_job_status_panel
from services.p2mc_api import P2MCAPIError, get_job_status, list_jobs


PAGE_SIZE = 10


if "jobs_list" not in st.session_state:
    st.session_state.jobs_list = None

if "jobs_selected_job_id" not in st.session_state:
    st.session_state.jobs_selected_job_id = None

if "jobs_selected_status" not in st.session_state:
    st.session_state.jobs_selected_status = None

if "jobs_page_number" not in st.session_state:
    st.session_state.jobs_page_number = 1


def load_jobs() -> None:
    with st.spinner("Loading jobs..."):
        st.session_state.jobs_list = list_jobs().get("jobs", [])


def build_job_table_rows(jobs: list[dict]) -> list[dict]:
    return [
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
    ]


def get_selection_rows(selection: object) -> list[int]:
    selection_value = getattr(selection, "selection", None)

    if isinstance(selection_value, dict):
        rows = selection_value.get("rows", [])
    else:
        rows = getattr(selection_value, "rows", [])

    return list(rows or [])


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

total_pages = max((len(jobs) - 1) // PAGE_SIZE + 1, 1)
current_page = min(st.session_state.jobs_page_number, total_pages)

page_col, count_col = st.columns([1, 3])

with page_col:
    page = int(
        st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=current_page,
            step=1,
        )
    )

st.session_state.jobs_page_number = page

start = (page - 1) * PAGE_SIZE
end = start + PAGE_SIZE
visible_jobs = jobs[start:end]
visible_end = min(end, len(jobs))

with count_col:
    st.caption(
        f"Showing jobs {start + 1}-{visible_end} of {len(jobs)}"
    )

table_selection = st.dataframe(
    build_job_table_rows(visible_jobs),
    hide_index=True,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = get_selection_rows(table_selection)

if not selected_rows:
    st.session_state.jobs_selected_job_id = None
    st.session_state.jobs_selected_status = None
    st.info("Select a job row to inspect its status and ModelCard.")
    st.stop()

selected_job_id = str(visible_jobs[selected_rows[0]]["job_id"])

if (
    selected_job_id != st.session_state.jobs_selected_job_id
    or
    st.session_state.jobs_selected_status is None
    or st.session_state.jobs_selected_status.get("job_id") != selected_job_id
):
    st.session_state.jobs_selected_job_id = selected_job_id
    st.session_state.jobs_selected_status = None

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
    show_card_by_default=True,
)
