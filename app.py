"""Cloud Cost Predictor -- Interactive Streamlit Dashboard."""

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clean"
MODELS = ROOT / "models"

# ---------------------------------------------------------------------------
# Theme colors
# ---------------------------------------------------------------------------
TEAL = "#00D4AA"
RED = "#FF6B6B"
YELLOW = "#FFD93D"
TEAL2 = "#4ECDC4"
DARK_BG = "#0E1117"
CARD_BG = "#1E2130"
TEXT = "#FAFAFA"
TEXT_DIM = "#8B949E"

# Plotly template
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Hide hamburger + footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0A0D14;
        border-right: 1px solid #1E2130;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #FAFAFA;
        font-size: 1.05rem;
        padding: 0.5rem 0;
    }

    /* KPI card */
    .kpi-card {
        background: linear-gradient(135deg, #1E2130 0%, #262940 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #2D3250;
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    .kpi-value {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #8B949E;
        margin: 0.3rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-context {
        font-size: 0.75rem;
        color: #6B7280;
        margin-top: 0.3rem;
    }

    /* Story text */
    .story-text {
        font-size: 1.15rem;
        line-height: 1.7;
        color: #C9D1D9;
        max-width: 720px;
        margin: 1rem 0 2rem 0;
    }
    .story-text strong {
        color: #00D4AA;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #FAFAFA;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #00D4AA;
        display: inline-block;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #6B7280;
        font-size: 0.8rem;
        border-top: 1px solid #1E2130;
        margin-top: 3rem;
    }

    /* Reduce top padding */
    .block-container {
        padding-top: 2rem !important;
    }

    /* Table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_utilization():
    return pd.read_csv(DATA / "vm_utilization_summary.csv")


@st.cache_data
def load_classified():
    return pd.read_csv(DATA / "vm_classified.csv")


@st.cache_data
def load_recommendations():
    return pd.read_csv(DATA / "vm_recommendations.csv")


@st.cache_data
def load_fleet_costs():
    return pd.read_csv(DATA / "fleet_cost_estimate.csv")


@st.cache_data
def load_models():
    models = {}
    for q in [0.10, 0.50, 0.95]:
        path = MODELS / f"quantile_{q:.2f}.json"
        model = xgb.XGBRegressor()
        model.load_model(path)
        models[q] = model
    return models


# ---------------------------------------------------------------------------
# Helper: KPI card
# ---------------------------------------------------------------------------
def kpi_card(value, label, context="", color=TEAL):
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-value" style="color: {color};">{value}</p>
        <p class="kpi-label">{label}</p>
        <p class="kpi-context">{context}</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: footer
# ---------------------------------------------------------------------------
def footer():
    st.markdown("""
    <div class="app-footer">
        Built by Aroa  |  Data: SAP Cloud Infrastructure Dataset (CC BY 4.0)
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 1: The Problem
# ---------------------------------------------------------------------------
def page_problem():
    st.markdown("# The Problem")
    st.markdown("""
    <p class="story-text">
        123,363 virtual machines. One month of data. The question:
        <strong>how much of this fleet is actually doing useful work?</strong>
    </p>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("$11.0M/mo", "Total Fleet Spend",
                 "123K VMs across 9 instance types", TEAL2)
    with c2:
        kpi_card("$5.9M/mo", "Wasted",
                 "Enough to fund 50 full-time engineers", RED)
    with c3:
        kpi_card("54.2%", "Waste Rate",
                 "More than half of every dollar, gone", YELLOW)

    st.markdown("")  # spacer

    # Classification donut + CPU histogram side by side
    classified = load_classified()
    class_counts = classified["class"].value_counts().reset_index()
    class_counts.columns = ["class", "count"]

    # Order and color
    class_order = ["idle", "oversized", "zombie", "right-sized", "review", "hot"]
    class_colors = {
        "idle": "#FF6B6B",
        "oversized": "#FFD93D",
        "zombie": "#FF4757",
        "right-sized": "#00D4AA",
        "review": "#4ECDC4",
        "hot": "#FF9F43",
    }
    class_counts["class"] = pd.Categorical(
        class_counts["class"], categories=class_order, ordered=True
    )
    class_counts = class_counts.sort_values("class")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">VM Classification</p>',
                    unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=class_counts["class"],
            values=class_counts["count"],
            hole=0.55,
            marker=dict(
                colors=[class_colors.get(c, TEAL) for c in class_counts["class"]]
            ),
            textinfo="label+percent",
            textfont=dict(size=13),
            hovertemplate="%{label}: %{value:,} VMs (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            showlegend=False,
            annotations=[dict(
                text="123K<br>VMs",
                x=0.5, y=0.5, font_size=18, showarrow=False,
                font=dict(color=TEXT),
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">CPU Utilization Distribution</p>',
                    unsafe_allow_html=True)
        util = load_utilization()
        fig_hist = go.Figure(go.Histogram(
            x=util["cpu_mean"],
            nbinsx=80,
            marker_color=TEAL,
            opacity=0.85,
            hovertemplate="CPU %{x:.0f}%: %{y:,} VMs<extra></extra>",
        ))
        fig_hist.add_vline(x=5, line_dash="dash", line_color=RED,
                           annotation_text="5% threshold",
                           annotation_position="top right",
                           annotation_font_color=RED)
        fig_hist.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            xaxis_title="Mean CPU Utilization (%)",
            yaxis_title="VM Count",
            xaxis=dict(gridcolor="#1E2130"),
            yaxis=dict(gridcolor="#1E2130"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("""
    <p class="story-text">
        The histogram tells the whole story. That massive spike on the left?
        <strong>50.4% of VMs average below 5% CPU.</strong>
        They are either sitting idle or running at a tiny fraction of their capacity.
        Meanwhile, <strong>only 4.4% of the fleet is properly sized.</strong>
        The rest is burning money.
    </p>
    """, unsafe_allow_html=True)

    # Spend breakdown by instance type
    st.markdown('<p class="section-header">Where the Money Goes</p>',
                unsafe_allow_html=True)
    fleet = load_fleet_costs()
    fleet_sorted = fleet.sort_values("monthly_cost", ascending=True)

    fig_spend = go.Figure()
    fig_spend.add_trace(go.Bar(
        y=fleet_sorted["ec2_type"],
        x=fleet_sorted["zombie_savings"],
        name="Zombie waste",
        orientation="h",
        marker_color=RED,
        hovertemplate="%{y}: $%{x:,.0f}/mo zombie waste<extra></extra>",
    ))
    fig_spend.add_trace(go.Bar(
        y=fleet_sorted["ec2_type"],
        x=fleet_sorted["downsize_savings"],
        name="Oversize waste",
        orientation="h",
        marker_color=YELLOW,
        hovertemplate="%{y}: $%{x:,.0f}/mo oversize waste<extra></extra>",
    ))
    fig_spend.add_trace(go.Bar(
        y=fleet_sorted["ec2_type"],
        x=fleet_sorted["monthly_cost"] - fleet_sorted["zombie_savings"] - fleet_sorted["downsize_savings"],
        name="Useful spend",
        orientation="h",
        marker_color=TEAL,
        hovertemplate="%{y}: $%{x:,.0f}/mo useful spend<extra></extra>",
    ))
    fig_spend.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        barmode="stack",
        xaxis_title="Monthly Cost ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#1E2130"),
        yaxis=dict(gridcolor="#1E2130"),
    )
    st.plotly_chart(fig_spend, use_container_width=True)

    st.markdown("""
    <p class="story-text">
        The r5.8xlarge tier alone wastes <strong>$1.6M/month</strong> on VMs that
        could be downsized or terminated. These are big machines doing small jobs.
    </p>
    """, unsafe_allow_html=True)

    footer()


# ---------------------------------------------------------------------------
# PAGE 2: The Model
# ---------------------------------------------------------------------------
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

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("1.71%", "MAE", "Median absolute error on CPU %", TEAL)
    with c2:
        kpi_card("85.6%", "Coverage", "Predictions above actual usage", TEAL2)
    with c3:
        kpi_card("5.4%", "Underprediction", "Rate we miss on the low side", YELLOW)
    with c4:
        kpi_card("3x", "Penalty", "Asymmetric loss on underprediction", RED)

    # Quantile regression explanation
    st.markdown('<p class="section-header">Quantile Regression: Three Predictions per VM</p>',
                unsafe_allow_html=True)
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

    # Sample VMs to show quantile bands
    rec = load_recommendations()
    # Pick a few representative VMs: one terminate, one downsize, one keep
    samples = []
    for action in ["terminate", "downsize", "review", "keep"]:
        subset = rec[rec["action"] == action]
        if len(subset) > 0:
            samples.append(subset.iloc[len(subset) // 2])

    sample_df = pd.DataFrame(samples).reset_index(drop=True)
    sample_df["vm_label"] = [f"VM {i+1} ({row['action']})" for i, row in sample_df.iterrows()]

    fig_q = go.Figure()

    # Error bars showing the prediction range
    fig_q.add_trace(go.Scatter(
        x=sample_df["vm_label"],
        y=sample_df["pred_mid"],
        error_y=dict(
            type="data",
            symmetric=False,
            array=(sample_df["pred_high"] - sample_df["pred_mid"]).tolist(),
            arrayminus=(sample_df["pred_mid"] - sample_df["pred_low"]).tolist(),
            color=TEAL2,
            thickness=3,
            width=10,
        ),
        mode="markers",
        marker=dict(size=12, color=TEAL, symbol="diamond"),
        name="Prediction range",
        hovertemplate=(
            "%{x}<br>"
            "Low: %{customdata[0]:.1f}%<br>"
            "Mid: %{y:.1f}%<br>"
            "High: %{customdata[1]:.1f}%<extra></extra>"
        ),
        customdata=np.column_stack([sample_df["pred_low"], sample_df["pred_high"]]),
    ))

    # Actual value
    fig_q.add_trace(go.Scatter(
        x=sample_df["vm_label"],
        y=sample_df["actual_cpu"],
        mode="markers",
        marker=dict(size=10, color=YELLOW, symbol="x"),
        name="Actual CPU",
        hovertemplate="%{x}<br>Actual: %{y:.1f}%<extra></extra>",
    ))

    # Threshold lines
    fig_q.add_hline(y=5, line_dash="dot", line_color=RED,
                    annotation_text="Terminate threshold (5%)",
                    annotation_position="top left",
                    annotation_font_color=RED)
    fig_q.add_hline(y=20, line_dash="dot", line_color=YELLOW,
                    annotation_text="Downsize threshold (20%)",
                    annotation_position="top left",
                    annotation_font_color=YELLOW)

    fig_q.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        yaxis_title="CPU Utilization (%)",
        xaxis=dict(gridcolor="#1E2130"),
        yaxis=dict(gridcolor="#1E2130"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_q, use_container_width=True)

    # Asymmetric loss visual
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Why Asymmetric Loss?</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <p class="story-text">
            Predicting too high means we keep a VM that could be downsized.
            That wastes money -- annoying, but harmless.<br><br>
            Predicting too low means we terminate a VM that was actually busy.
            That kills a workload. <strong>The 3x penalty on underprediction
            makes the model err on the side of caution.</strong>
        </p>
        """, unsafe_allow_html=True)

    with col2:
        # Loss function visualization
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
            line=dict(color=TEAL, width=3),
        ))
        fig_loss.add_vline(x=0, line_color="#2D3250")
        fig_loss.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            xaxis_title="Prediction Error (negative = underprediction)",
            yaxis_title="Loss",
            xaxis=dict(gridcolor="#1E2130"),
            yaxis=dict(gridcolor="#1E2130"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # Feature importance
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
        marker_color=[TEAL, TEAL2, TEAL, YELLOW, TEAL2],
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig_feat.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        xaxis_title="Importance (gain)",
        xaxis=dict(gridcolor="#1E2130"),
        yaxis=dict(gridcolor="#1E2130"),
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    footer()


# ---------------------------------------------------------------------------
# PAGE 3: Recommendations
# ---------------------------------------------------------------------------
def page_recommendations():
    st.markdown("# Recommendations")
    st.markdown("""
    <p class="story-text">
        The model scored every VM. Now each one gets an action
        and a risk level. The question: <strong>how much can we save,
        and how safely?</strong>
    </p>
    """, unsafe_allow_html=True)

    rec = load_recommendations()
    fleet = load_fleet_costs()

    total_cost = fleet["monthly_cost"].sum()
    zombie_savings = fleet["zombie_savings"].sum()
    downsize_savings = fleet["downsize_savings"].sum()
    total_savings = zombie_savings + downsize_savings

    # KPI row
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card(f"{len(rec):,}", "VMs Scored",
                 "Every VM gets an action + risk level", TEAL2)
    with c2:
        kpi_card(f"${total_savings/1e6:.1f}M/mo", "Potential Savings",
                 f"From ${total_cost/1e6:.1f}M down to ${(total_cost-total_savings)/1e6:.1f}M",
                 TEAL)
    with c3:
        safe_pct = (rec["risk"] == "safe").mean() * 100
        kpi_card(f"{safe_pct:.0f}%", "Safe Actions",
                 "Low-risk recommendations we can act on now", TEAL)

    # Actions + Risk side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Action Breakdown</p>',
                    unsafe_allow_html=True)
        action_counts = rec["action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        action_colors = {
            "terminate": RED,
            "downsize": YELLOW,
            "review": TEAL2,
            "keep": TEAL,
        }
        fig_act = go.Figure(go.Bar(
            x=action_counts["action"],
            y=action_counts["count"],
            marker_color=[action_colors.get(a, TEAL) for a in action_counts["action"]],
            text=action_counts["count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color=TEXT),
            hovertemplate="%{x}: %{y:,} VMs<extra></extra>",
        ))
        fig_act.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            yaxis_title="VM Count",
            xaxis=dict(gridcolor="#1E2130"),
            yaxis=dict(gridcolor="#1E2130"),
        )
        st.plotly_chart(fig_act, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Risk Distribution</p>',
                    unsafe_allow_html=True)
        risk_order = ["safe", "moderate", "risky"]
        risk_colors = {"safe": TEAL, "moderate": YELLOW, "risky": RED}
        risk_data = rec[rec["risk"].isin(risk_order)]
        risk_counts = risk_data["risk"].value_counts().reindex(risk_order).reset_index()
        risk_counts.columns = ["risk", "count"]

        fig_risk = go.Figure(go.Bar(
            x=risk_counts["risk"],
            y=risk_counts["count"],
            marker_color=[risk_colors[r] for r in risk_counts["risk"]],
            text=risk_counts["count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color=TEXT),
            hovertemplate="%{x}: %{y:,} VMs<extra></extra>",
        ))
        fig_risk.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            yaxis_title="VM Count",
            xaxis=dict(gridcolor="#1E2130"),
            yaxis=dict(gridcolor="#1E2130"),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # Savings waterfall
    st.markdown('<p class="section-header">Savings Waterfall</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <p class="story-text">
        Starting from the $11M monthly bill, here is where each optimization
        takes a bite.
    </p>
    """, unsafe_allow_html=True)

    remaining = total_cost - total_savings
    fig_wf = go.Figure(go.Waterfall(
        name="Monthly Cost",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Current Spend", "Terminate Zombies", "Downsize Oversized", "Optimized Spend"],
        y=[total_cost, -zombie_savings, -downsize_savings, remaining],
        text=[
            f"${total_cost/1e6:.1f}M",
            f"-${zombie_savings/1e6:.1f}M",
            f"-${downsize_savings/1e6:.1f}M",
            f"${remaining/1e6:.1f}M",
        ],
        textposition="outside",
        textfont=dict(color=TEXT),
        connector=dict(line=dict(color="#2D3250")),
        decreasing=dict(marker=dict(color=TEAL)),
        increasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=TEAL2)),
        hovertemplate="%{x}: $%{y:,.0f}/mo<extra></extra>",
    ))
    fig_wf.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        yaxis_title="Monthly Cost ($)",
        xaxis=dict(gridcolor="#1E2130"),
        yaxis=dict(gridcolor="#1E2130"),
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Searchable table
    st.markdown('<p class="section-header">VM Explorer</p>',
                unsafe_allow_html=True)

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        action_filter = st.multiselect(
            "Filter by action",
            options=sorted(rec["action"].unique()),
            default=sorted(rec["action"].unique()),
        )
    with fc2:
        risk_filter = st.multiselect(
            "Filter by risk",
            options=["safe", "moderate", "risky", "n/a"],
            default=["safe", "moderate", "risky", "n/a"],
        )
    with fc3:
        search = st.text_input("Search VM ID", "")

    filtered = rec[
        (rec["action"].isin(action_filter)) &
        (rec["risk"].isin(risk_filter))
    ]
    if search:
        filtered = filtered[filtered["instance"].str.contains(search, case=False)]

    st.dataframe(
        filtered.style.format({
            "actual_cpu": "{:.1f}%",
            "pred_low": "{:.1f}%",
            "pred_mid": "{:.1f}%",
            "pred_high": "{:.1f}%",
            "monthly_savings": "${:.2f}",
        }),
        use_container_width=True,
        height=400,
    )
    st.caption(f"Showing {len(filtered):,} of {len(rec):,} VMs")

    footer()


# ---------------------------------------------------------------------------
# PAGE 4: Try It
# ---------------------------------------------------------------------------
def page_try_it():
    st.markdown("# Try It")
    st.markdown("""
    <p class="story-text">
        Upload your own VM utilization data and see what the model recommends.
        The CSV needs the same columns as <code>vm_utilization_summary.csv</code>:
        instance, cpu_mean, cpu_median, cpu_p5, cpu_p95, cpu_min, cpu_max, etc.
    </p>
    """, unsafe_allow_html=True)

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

            # Build features
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

                results = []
                for i in range(len(df)):
                    low = preds[0.10][i]
                    mid = preds[0.50][i]
                    high = preds[0.95][i]

                    if high < 5:
                        action = "terminate"
                    elif high < 20:
                        action = "downsize"
                    elif mid < 50:
                        action = "review"
                    else:
                        action = "keep"

                    if action == "terminate":
                        margin = 5 - high
                        risk = "safe" if margin > 3 else ("moderate" if margin > 1 else "risky")
                    elif action == "downsize":
                        margin = 20 - high
                        risk = "safe" if margin > 8 else ("moderate" if margin > 3 else "risky")
                    else:
                        risk = "n/a"

                    savings = max(0, (X[i, 2] - mid)) * 0.001 * 730

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

            # Summary cards
            st.markdown('<p class="section-header">Results</p>',
                        unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                n_term = (result_df["action"] == "terminate").sum()
                kpi_card(str(n_term), "Terminate", "", RED)
            with c2:
                n_down = (result_df["action"] == "downsize").sum()
                kpi_card(str(n_down), "Downsize", "", YELLOW)
            with c3:
                n_rev = (result_df["action"] == "review").sum()
                kpi_card(str(n_rev), "Review", "", TEAL2)
            with c4:
                n_keep = (result_df["action"] == "keep").sum()
                kpi_card(str(n_keep), "Keep", "", TEAL)

            # Action breakdown chart
            action_summary = result_df["action"].value_counts().reset_index()
            action_summary.columns = ["action", "count"]
            action_colors = {
                "terminate": RED, "downsize": YELLOW,
                "review": TEAL2, "keep": TEAL,
            }
            fig = go.Figure(go.Bar(
                x=action_summary["action"],
                y=action_summary["count"],
                marker_color=[action_colors.get(a, TEAL) for a in action_summary["action"]],
                text=action_summary["count"],
                textposition="outside",
                textfont=dict(color=TEXT),
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=320,
                yaxis_title="VM Count",
                xaxis=dict(gridcolor="#1E2130"),
                yaxis=dict(gridcolor="#1E2130"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Results table
            st.dataframe(
                result_df.style.format({
                    "actual_cpu": "{:.1f}%",
                    "pred_low": "{:.1f}%",
                    "pred_mid": "{:.1f}%",
                    "pred_high": "{:.1f}%",
                    "monthly_savings": "${:.2f}",
                }),
                use_container_width=True,
                height=400,
            )

            # Download
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
        # Show sample format
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

    footer()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Cloud Cost Predictor",
        page_icon="💸",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("## Cloud Cost Predictor")
        st.markdown(
            '<p style="color: #8B949E; font-size: 0.85rem;">'
            "Turning 123K VMs into actionable savings"
            "</p>",
            unsafe_allow_html=True,
        )
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
