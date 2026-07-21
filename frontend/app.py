import streamlit as st

main_page = st.Page(
    "pages/main_page.py",
    title="Generate",
    icon=":material/add_circle:",
)
jobs_page = st.Page(
    "pages/jobs_page.py",
    title="Jobs",
    icon=":material/list_alt:",
)



pg = st.navigation([main_page, jobs_page])
pg.run()
