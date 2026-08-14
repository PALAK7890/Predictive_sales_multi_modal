import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def summary_metrics(summary):

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Retrieved",
        summary["retrieved"],
    )

    c2.metric(
        "Successful",
        summary["successful"],
    )

    c3.metric(
        "Failed",
        summary["failed"],
    )

    c4.metric(
        "Success Rate",
        f"{summary['success_rate']}%",
    )