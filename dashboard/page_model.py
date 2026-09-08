"""Page 2: The Model — quantile regression and explainability."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import (
    AMBER,
    GRAY,
    PLOTLY_LAYOUT,
    RED,
    ROOT,
    TEXT,
    TEXT_DIM,
    footer,
    kpi_card,
    load_models,
    load_recommendations,
    safe_page,
)


@safe_page
def page_model():
    st.markdown("# The Model")
    st.markdown("""
    <p class="story-text">
        The goal: predict each VM's future CPU usage well enough to decide
        what to do with it. But a wrong prediction in the wrong direction
        kills a production workload. So the model is built to be
        <strong>deliberately cautious</strong>.
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("1.71%", "MAE", "Median absolute error on CPU %", TEXT)
    with c2:
        kpi_card("85.6%", "Coverage", "Predictions above actual usage", TEXT)
    with c3:
        kpi_card("5.4%", "Underprediction", "Rate we miss on the low side", AMBER)
    with c4:
        kpi_card("3x", "Penalty", "Asymmetric loss on underprediction", RED)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<p class="section-header">Quantile Regression: Three Predictions per VM</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <p class="story-text">
        Instead of one point estimate, the model gives three:
        a <strong>low bound</strong> (10th percentile),
        a <strong>median</strong> (50th), and a
        <strong>high bound</strong> (95th).
        The decision threshold uses the 95th percentile --
        if even the pessimistic prediction says "this VM is idle," we can be confident.
    </p>
    """, unsafe_allow_html=True)

    rec = load_recommendations()
    samples = []
    for action in ["terminate", "downsize", "review", "keep"]:
        subset = rec[rec["action"] == action]
        if len(subset) > 0:
            samples.append(subset.iloc[len(subset) // 2])

    sample_df = pd.DataFrame(samples).reset_index(drop=True)
    sample_df["vm_label"] = [
        f"VM {i+1} ({row['action']})" for i, row in sample_df.iterrows()
    ]

    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(
        x=sample_df["vm_label"],
        y=sample_df["pred_mid"],
        error_y=dict(
            type="data",
            symmetric=False,
            array=(sample_df["pred_high"] - sample_df["pred_mid"]).tolist(),
            arrayminus=(sample_df["pred_mid"] - sample_df["pred_low"]).tolist(),
            color=TEXT_DIM,
            thickness=3,
            width=10,
        ),
        mode="markers",
        marker=dict(size=12, color=TEXT_DIM, symbol="diamond"),
        name="Prediction range",
        hovertemplate=(
            "%{x}<br>"
            "Low: %{customdata[0]:.1f}%<br>"
            "Mid: %{y:.1f}%<br>"
            "High: %{customdata[1]:.1f}%<extra></extra>"
        ),
        customdata=np.column_stack([sample_df["pred_low"], sample_df["pred_high"]]),
    ))
    fig_q.add_trace(go.Scatter(
        x=sample_df["vm_label"],
        y=sample_df["actual_cpu"],
        mode="markers",
        marker=dict(size=10, color=AMBER, symbol="x"),
        name="Actual CPU",
        hovertemplate="%{x}<br>Actual: %{y:.1f}%<extra></extra>",
    ))
    fig_q.add_hline(y=5, line_dash="dot", line_color=RED,
                    annotation_text="Terminate threshold (5%)",
                    annotation_position="top left",
                    annotation_font_color=RED)
    fig_q.add_hline(y=20, line_dash="dot", line_color=AMBER,
                    annotation_text="Downsize threshold (20%)",
                    annotation_position="top left",
                    annotation_font_color=AMBER)
    fig_q.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        yaxis_title="CPU Utilization (%)",
        xaxis=dict(gridcolor="#0A1628"),
        yaxis=dict(gridcolor="#0A1628"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_q, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Why Asymmetric Loss?</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <p class="story-text">
            Predicting too high means we keep a VM that could be downsized.
            That wastes money. Annoying, but harmless.<br><br>
            Predicting too low means we terminate a VM that was actually busy.
            That kills a workload. <strong>The 3x penalty on underprediction
            makes the model err on the side of caution.</strong>
        </p>
        """, unsafe_allow_html=True)

    with col2:
        errors = np.linspace(-10, 10, 200)
        symmetric_loss = errors ** 2
        asymmetric_loss = np.where(errors < 0, 3 * errors ** 2, errors ** 2)

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=errors, y=symmetric_loss,
            mode="lines", name="Symmetric (MSE)",
            line=dict(color=TEXT_DIM, width=2, dash="dash"),
        ))
        fig_loss.add_trace(go.Scatter(
            x=errors, y=asymmetric_loss,
            mode="lines", name="Asymmetric (3x penalty)",
            line=dict(color=TEXT, width=3),
        ))
        fig_loss.add_vline(x=0, line_color="#0F1D33")
        fig_loss.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            xaxis_title="Prediction Error (negative = underprediction)",
            yaxis_title="Loss",
            xaxis=dict(gridcolor="#0A1628"),
            yaxis=dict(gridcolor="#0A1628"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown('<p class="section-header">Feature Importance</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <p class="story-text">
        After VIF-based multicollinearity cleanup, <strong>5 features</strong> survive.
        The model needs surprisingly little to make accurate predictions.
    </p>
    """, unsafe_allow_html=True)

    models = load_models()
    mid_model = models[0.50]
    feature_names = ["std (volatility)", "min (floor)", "p50 (median)",
                     "trend (slope)", "cv (relative spread)"]
    importances = mid_model.feature_importances_

    fig_feat = go.Figure(go.Bar(
        x=importances,
        y=feature_names,
        orientation="h",
        marker_color=GRAY,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig_feat.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        xaxis_title="Importance (gain)",
        xaxis=dict(gridcolor="#0A1628"),
        yaxis=dict(gridcolor="#0A1628"),
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown('<p class="section-header">Why Did the Model Decide That?</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <p class="story-text">
        Feature importance tells us what matters <em>globally</em>. SHAP values
        tell us what matters <em>for each VM</em>. Below: two real VMs from
        the fleet, one recommended for termination and one to keep. The bars
        show how each feature pushed the prediction up (red) or down (blue).
    </p>
    """, unsafe_allow_html=True)

    shap_img = ROOT / "reports" / "figures" / "shap_waterfall_example.png"
    if shap_img.exists():
        st.image(str(shap_img), use_container_width=True)
        st.caption(
            "SHAP waterfall plots (TreeExplainer on the median model). "
            "Each bar shows a feature's contribution to that VM's predicted CPU usage."
        )
    else:
        st.info("SHAP waterfall plot not found. Run notebook 03 to generate it.")

    footer()
