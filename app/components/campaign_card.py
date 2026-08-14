import streamlit as st

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
def campaign_card(campaign):

    emoji = "🟢" if campaign["state"] == "successful" else "🔴"

    with st.container(border=True):

        st.subheader(f"{emoji} {campaign['title']}")

        col1, col2 = st.columns(2)

        with col1:

            st.write(f"**Category:** {campaign['category']}")
            st.write(f"**Main Category:** {campaign['main_category']}")
            st.write(f"**Country:** {campaign['country']}")

        with col2:

            st.write(f"**Goal:** ${campaign['goal']:,.0f}")
            st.write(f"**Raised:** ${campaign['pledged']:,.0f}")
            st.write(f"**Backers:** {campaign['backers']}")

        st.progress(campaign["similarity"])

        st.caption(
            f"Similarity Score: {campaign['similarity']:.2f}"
        )