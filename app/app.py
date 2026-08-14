import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
st.set_page_config(
    page_title="Kickstarter Intelligence Engine",
    page_icon="🚀",
)

st.title("🚀 Kickstarter Intelligence Engine")

st.markdown(
    """
Welcome!

Use the navigation panel on the left to explore:

- 🚀 Similar Campaign Dashboard
- 📊 Business Recommendation Engine *(Coming Soon)*
- 💰 Funding Goal Optimizer *(Coming Soon)*
- ⚠️ Risk Analyzer *(Coming Soon)*
- 🤖 AI Consultant *(Coming Soon)*
"""
)