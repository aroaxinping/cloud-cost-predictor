"""Page 1: The Problem — fleet waste overview."""
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
    load_classified,
    load_fleet_costs,
    load_utilization,
    safe_page,
)


@safe_page
def page_problem():
    st.markdown("# The Problem")
    st.markdown("""
    <p class="story-text">
        123,363 virtual machines. One month of data. The question:
        <strong>how much of this fleet is actually doing useful work?</strong>
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("$11.0M/mo", "Total Fleet Spend",
                 "123K VMs across 9 instance types", TEXT)
    with c2:
        kpi_card("$5.9M/mo", "Wasted",
                 "Enough to fund 50 full-time engineers", RED)
    with c3:
        kpi_card("54.2%", "Waste Rate",
                 "More than half of every dollar, gone", AMBER)

    st.markdown("<br>", unsafe_allow_html=True)

    classified = load_classified()
    class_counts = classified["class"].value_counts().reset_index()
    class_counts.columns = ["class", "count"]

    class_order = ["idle", "oversized", "zombie", "right-sized", "review", "hot"]
    class_colors = {
        "idle": "#FACC15",
        "oversized": "#F59E0B",
        "zombie": "#EF4444",
        "right-sized": "#22C55E",
        "review": "#94A3B8",
        "hot": "#FF6B00",
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
                colors=[class_colors.get(c, GRAY) for c in class_counts["class"]]
            ),
            textinfo="label+percent",
            textposition="auto",
            insidetextorientation="horizontal",
            outsidetextfont=dict(size=12, color=TEXT),
            insidetextfont=dict(size=12, color="#000000"),
            hovertemplate="%{label}: %{value:,} VMs (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            **PLOTLY_LAYOUT,
            height=450,
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
            marker_color=GRAY,
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
            xaxis=dict(gridcolor="#0A1628"),
            yaxis=dict(gridcolor="#0A1628"),
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
        marker_color=AMBER,
        hovertemplate="%{y}: $%{x:,.0f}/mo oversize waste<extra></extra>",
    ))
    fig_spend.add_trace(go.Bar(
        y=fleet_sorted["ec2_type"],
        x=fleet_sorted["monthly_cost"] - fleet_sorted["zombie_savings"] - fleet_sorted["downsize_savings"],
        name="Useful spend",
        orientation="h",
        marker_color=GREEN,
        hovertemplate="%{y}: $%{x:,.0f}/mo useful spend<extra></extra>",
    ))
    fig_spend.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        barmode="stack",
        xaxis_title="Monthly Cost ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#0A1628"),
        yaxis=dict(gridcolor="#0A1628"),
    )
    st.plotly_chart(fig_spend, use_container_width=True)

    st.markdown("""
    <p class="story-text">
        The r5.8xlarge tier alone wastes <strong>$1.6M/month</strong> on VMs that
        could be downsized or terminated. These are big machines doing small jobs.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0A1628, #0F1D33);
                border: 1px solid #4B5563; border-radius: 12px;
                padding: 2rem; text-align: center; margin: 2rem 0;">
        <p style="font-size: 1.2rem; color: #FAFAFA; margin: 0 0 0.5rem 0;">
            So how do we decide which VMs to act on <strong style="color: #22C55E;">safely</strong>?
        </p>
        <p style="font-size: 0.95rem; color: #8B949E; margin: 0;">
            Head to <strong>The Model</strong> to see how quantile regression
            separates confident savings from risky ones.
        </p>
    </div>
    """, unsafe_allow_html=True)

    footer()
