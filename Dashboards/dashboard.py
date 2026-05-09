import streamlit as st

st.set_page_config(page_title="Data Science Dashboard", layout="wide")

st.title("Data Science Dashboard")

st.write("Use the sidebar to navigate between datasets and tools.")
# st.set_page_config(page_title="Data Science Dashboard", layout="wide")

# Define pages
bigfive_page = st.Page("pages/big5_dashboard.py", title="Big Five")
crimes_page = st.Page("pages/crimes_dashboard.py", title="Crimes")
happiness_page = st.Page("pages/WH_dashboard.py", title="World Happiness")
auto_page = st.Page("pages/Automating_dashboard.py", title="Auto Analyzer")

# Navigation sidebar
pg = st.navigation([
    bigfive_page,
    crimes_page,
    happiness_page,
    auto_page
])

pg.run()