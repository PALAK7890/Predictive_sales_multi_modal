import streamlit as st

from src.dashboard.dashboard import SimilarCampaignDashboard
from app.components.campaign_card import campaign_card
from app.components.metrics import summary_metrics

from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Similar Campaign Dashboard",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 Kickstarter Intelligence Engine")

st.write(
    "Find similar Kickstarter campaigns using semantic search."
)

dashboard = SimilarCampaignDashboard()

query = st.text_input(
    "Describe your campaign",
    placeholder="AI powered fitness application...",
)

top_k = st.slider(
    "Number of similar campaigns",
    min_value=5,
    max_value=25,
    value=10,
)

if st.button("Search Similar Campaigns"):

    if query.strip() == "":
        st.warning("Please enter a campaign description.")
        st.stop()

    with st.spinner("Searching similar campaigns..."):

        report = dashboard.search(
            query=query,
            top_k=top_k,
        )

    st.divider()

    summary_metrics(report["summary"])

    st.divider()

    st.subheader("Most Similar Campaigns")

    for campaign in report["campaigns"]:
        campaign_card(campaign)