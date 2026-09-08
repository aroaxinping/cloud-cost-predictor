"""Page 4: Try It — upload your own data for predictions."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import (
    AMBER,
    GRAY,
    GREEN,
    PLOTLY_LAYOUT,
    RED,
    TEXT,
    footer,
    kpi_card,
    load_models,
    load_utilization,
)
from src.predict import assess_risk, estimate_savings, load_fleet_avg_hourly, recommend


def page_try_it():
    st.markdown("# Try It")
    st.markdown("""
    <p class="story-text">
        Upload your own VM utilization data and get instant recommendations.
        The model runs three XGBoost predictions per VM (low, median, high)
        and classifies each one into <strong>terminate</strong>,
        <strong>downsize</strong>, <strong>review</strong>, or <strong>keep</strong>
        based on the 95th percentile prediction.
    </p>
    """, unsafe_allow_html=True)

    with st.expander("How does it work?", expanded=False):
        st.markdown("""
**1. Feature extraction.** From your CPU columns, the model computes 5 features:
standard deviation (volatility), minimum (floor), median (typical load),
trend (slope over time, defaults to 0 for static data), and coefficient of variation
(relative spread).

**2. Quantile predictions.** Three XGBoost models predict CPU usage at q=0.10
(optimistic), q=0.50 (median), and q=0.95 (pessimistic). The 95th percentile
is the decision threshold: if even the pessimistic estimate says "idle", we're confident.

**3. Decision rules.** If q95 < 5%: terminate (VM is almost certainly idle).
If q95 < 20%: downsize (oversized but not idle). If q50 < 50%: review manually.
Otherwise: keep.

**4. Risk scoring.** Each recommendation gets a risk level (safe / moderate / risky)
based on how far the prediction is from the threshold. A VM predicted at 1% with a
5% threshold is "safe to terminate"; one at 4.5% is "risky".
        """)

    uploaded = st.file_uploader("Upload a VM utilization CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df):,} VMs from {uploaded.name}")

            required = ["instance", "cpu_mean", "cpu_min", "cpu_median"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
                return

            with st.spinner("Running predictions..."):
                models = load_models()
                has_std = "cpu_std" in df.columns
                if not has_std and "cpu_max" in df.columns:
                    df["cpu_std"] = (df["cpu_max"] - df["cpu_min"]) / 4
                elif not has_std:
                    df["cpu_std"] = 0.0

                features = []
                for _, row in df.iterrows():
                    cpu_mean = float(row["cpu_mean"])
                    cpu_std = float(row["cpu_std"])
                    cpu_min = float(row["cpu_min"])
                    cpu_median = float(row["cpu_median"])
                    cv = cpu_std / cpu_mean if cpu_mean > 0 else 0
                    features.append([cpu_std, cpu_min, cpu_median, 0.0, cv])

                X = np.array(features)
                preds = {q: models[q].predict(X) for q in [0.10, 0.50, 0.95]}

                avg_hourly = load_fleet_avg_hourly()
                results = []
                for i in range(len(df)):
                    low = preds[0.10][i]
                    mid = preds[0.50][i]
                    high = preds[0.95][i]

                    action = recommend(high, mid)
                    risk = assess_risk(action, high)
                    savings = estimate_savings(action, avg_hourly, mid)

                    results.append({
                        "instance": df.iloc[i]["instance"],
                        "actual_cpu": df.iloc[i]["cpu_mean"],
                        "pred_low": round(low, 2),
                        "pred_mid": round(mid, 2),
                        "pred_high": round(high, 2),
                        "action": action,
                        "risk": risk,
                        "monthly_savings": round(savings, 2),
                    })

                result_df = pd.DataFrame(results)

            st.markdown('<p class="section-header">Results</p>',
                        unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                n_term = (result_df["action"] == "terminate").sum()
                kpi_card(str(n_term), "Terminate", "", RED)
            with c2:
                n_down = (result_df["action"] == "downsize").sum()
                kpi_card(str(n_down), "Downsize", "", AMBER)
            with c3:
                n_rev = (result_df["action"] == "review").sum()
                kpi_card(str(n_rev), "Review", "", GRAY)
            with c4:
                n_keep = (result_df["action"] == "keep").sum()
                kpi_card(str(n_keep), "Keep", "", GREEN)

            action_summary = result_df["action"].value_counts().reset_index()
            action_summary.columns = ["action", "count"]
            action_colors = {
                "terminate": RED, "downsize": AMBER,
                "review": GRAY, "keep": GREEN,
            }
            fig = go.Figure(go.Bar(
                x=action_summary["action"],
                y=action_summary["count"],
                marker_color=[action_colors.get(a, GRAY) for a in action_summary["action"]],
                text=action_summary["count"],
                textposition="outside",
                textfont=dict(color=TEXT),
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=320,
                yaxis_title="VM Count",
                xaxis=dict(gridcolor="#0A1628"),
                yaxis=dict(gridcolor="#0A1628"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                result_df,
                column_config={
                    "actual_cpu": st.column_config.NumberColumn("Actual CPU", format="%.1f%%"),
                    "pred_low": st.column_config.NumberColumn("Pred Low", format="%.1f%%"),
                    "pred_mid": st.column_config.NumberColumn("Pred Mid", format="%.1f%%"),
                    "pred_high": st.column_config.NumberColumn("Pred High", format="%.1f%%"),
                    "monthly_savings": st.column_config.NumberColumn("Savings", format="$%.2f"),
                },
                use_container_width=True,
                height=400,
            )

            csv_out = result_df.to_csv(index=False)
            st.download_button(
                "Download Results CSV",
                csv_out,
                file_name="vm_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        st.markdown("""
        <p class="story-text">
            No file yet? Here is what the input format looks like:
        </p>
        """, unsafe_allow_html=True)

        sample = load_utilization().head(5)[
            ["instance", "cpu_mean", "cpu_median", "cpu_p5", "cpu_p95",
             "cpu_min", "cpu_max", "mem_mean"]
        ]
        st.dataframe(sample, use_container_width=True)

        sample_csv = sample.to_csv(index=False)
        st.download_button(
            "Download sample CSV to try",
            sample_csv,
            file_name="sample_vm_utilization.csv",
            mime="text/csv",
        )

    footer()
