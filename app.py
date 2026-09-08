"""Cloud Cost Predictor: Interactive Streamlit Dashboard."""
from pathlib import Path

import streamlit as st

from dashboard import inject_css, page_model, page_problem, page_recommendations, page_try_it

ROOT = Path(__file__).resolve().parent


def main():
    st.set_page_config(
        page_title="Cloud Cost Predictor",
        page_icon="\U0001f4b8",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    logo_path = ROOT / "assets" / "logo.svg"
    icon_path = ROOT / "assets" / "logo_icon.svg"
    if logo_path.exists() and icon_path.exists():
        st.logo(str(logo_path), icon_image=str(icon_path))

    with st.sidebar:
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["The Problem", "The Model", "Recommendations", "Try It"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            '<p style="color: #6B7280; font-size: 0.75rem;">'
            "XGBoost quantile regression<br>"
            "q=0.95 decision threshold<br>"
            "3x asymmetric loss<br>"
            "5 features after VIF cleanup"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            '<p style="font-size: 0.8rem; color: #8B949E;">'
            '<a href="https://github.com/aroaxinping" target="_blank" '
            'style="color: #C9D1D9; text-decoration: none;">GitHub</a>'
            " &nbsp;|&nbsp; "
            '<a href="https://linkedin.com/in/aroaxinping" target="_blank" '
            'style="color: #C9D1D9; text-decoration: none;">LinkedIn</a>'
            "</p>",
            unsafe_allow_html=True,
        )

    if page == "The Problem":
        page_problem()
    elif page == "The Model":
        page_model()
    elif page == "Recommendations":
        page_recommendations()
    elif page == "Try It":
        page_try_it()


if __name__ == "__main__":
    main()
