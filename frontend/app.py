import streamlit as st



main_page = st.Page("pages/main_page.py", title="Main page", icon=":material/home:")



pg = st.navigation([main_page])
st.set_page_config(page_title="Main page", page_icon=":material/edit:")
pg.run()