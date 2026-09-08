"""Page 3: Recommendations — actions, savings, and sensitivity."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import (
    AMBER,
    GRAY,
    GREEN,
    PLOTLY_LAYOUT,
    RED,
    ROOT,
    TEXT,
    footer,
    kpi_card,
    load_fleet_costs,
    load_recommendations,
    safe_page,
)
from src.predict import load_fleet_avg_hourly


@safe_page
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

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card(f"{len(rec):,}", "VMs Scored",
                 "Every VM gets an action + risk level", TEXT)
    with c2:
        kpi_card(f"${total_savings/1e6:.1f}M/mo", "Potential Savings",
                 f"From ${total_cost/1e6:.1f}M down to ${(total_cost-total_savings)/1e6:.1f}M",
                 GREEN)
    with c3:
        safe_pct = (rec["risk"] == "safe").mean() * 100
        kpi_card(f"{safe_pct:.0f}%", "Safe Actions",
                 "Low-risk recommendations we can act on now", GREEN)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Action Breakdown</p>',
                    unsafe_allow_html=True)
        action_counts = rec["action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        action_colors = {
            "terminate": RED, "downsize": AMBER, "review": GRAY, "keep": GREEN,
        }
        fig_act = go.Figure(go.Bar(
            x=action_counts["action"],
            y=action_counts["count"],
            marker_color=[action_colors.get(a, GRAY) for a in action_counts["action"]],
            text=action_counts["count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color=TEXT),
            hovertemplate="%{x}: %{y:,} VMs<extra></extra>",
        ))
        fig_act.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            yaxis_title="VM Count",
            xaxis=dict(gridcolor="#0A1628"),
            yaxis=dict(gridcolor="#0A1628"),
        )
        st.plotly_chart(fig_act, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Risk Distribution</p>',
                    unsafe_allow_html=True)
        risk_order = ["safe", "moderate", "risky"]
        risk_colors = {"safe": GREEN, "moderate": AMBER, "risky": RED}
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
            xaxis=dict(gridcolor="#0A1628"),
            yaxis=dict(gridcolor="#0A1628"),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

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
        connector=dict(line=dict(color="#0F1D33")),
        decreasing=dict(marker=dict(color=GREEN)),
        increasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=GREEN)),
        hovertemplate="%{x}: $%{y:,.0f}/mo<extra></extra>",
    ))
    fig_wf.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        yaxis_title="Monthly Cost ($)",
        xaxis=dict(gridcolor="#0A1628"),
        yaxis=dict(gridcolor="#0A1628"),
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    with st.expander("How are savings calculated?", expanded=False):
        st.markdown("""
**Terminate savings** = full monthly cost of the VM (it is doing nothing useful).

**Downsize savings** = 50% of the monthly cost (move to a smaller instance type
in the same family, e.g. r5.8xlarge to r5.4xlarge).

Costs use real AWS EC2 on-demand prices for us-east-1. Each SAP VM size category
is mapped to a specific EC2 instance type. The savings are conservative: they
assume on-demand pricing, not reserved instances or savings plans.
        """)

    with st.expander("AWS EC2 pricing used", expanded=False):
        pricing_file = ROOT / "data" / "pricing" / "ec2_on_demand.json"
        if pricing_file.exists():
            with open(pricing_file) as f:
                pricing_data = json.load(f)
            prices = pricing_data.get("instances", {})
            price_rows = []
            for itype, info in sorted(prices.items(), key=lambda x: x[1]["usd_per_hour"]):
                hourly = info["usd_per_hour"]
                monthly = hourly * 730
                price_rows.append({
                    "Instance Type": itype,
                    "vCPUs": info.get("vcpus", ""),
                    "RAM (GB)": info.get("ram_gb", ""),
                    "Hourly (USD)": f"${hourly:.4f}",
                    "Monthly (USD)": f"${monthly:,.2f}",
                })
            st.dataframe(pd.DataFrame(price_rows), use_container_width=True, hide_index=True)
            st.caption(
                f"Source: AWS Bulk Pricing API, region "
                f"{pricing_data.get('metadata', {}).get('region', 'us-east-1')}. "
                "Refresh with: python scripts/fetch_ec2_pricing.py"
            )
        else:
            st.info("Pricing file not found. Run scripts/fetch_ec2_pricing.py to generate it.")

    st.markdown('<p class="section-header">Threshold Sensitivity</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <p class="story-text">
        How do the savings change if we adjust the terminate threshold?
        Moving it from 5% to 3% is more conservative (fewer kills, less risk).
        Moving it to 10% is more aggressive (more savings, more risk).
    </p>
    """, unsafe_allow_html=True)

    thresholds_range = list(range(1, 16))
    savings_by_threshold = []
    terminate_counts = []
    avg_hourly = load_fleet_avg_hourly()
    for t in thresholds_range:
        n_term = (rec["pred_high"] < t).sum()
        terminate_counts.append(n_term)
        term_savings = n_term * avg_hourly * 730
        savings_by_threshold.append(term_savings / 1e6)

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=thresholds_range,
        y=savings_by_threshold,
        mode="lines+markers",
        line=dict(color=GREEN, width=3),
        marker=dict(size=8),
        name="Monthly savings ($M)",
        hovertemplate=(
            "Threshold: %{x}%<br>Savings: $%{y:.1f}M<br>"
            "VMs terminated: %{customdata:,}<extra></extra>"
        ),
        customdata=terminate_counts,
    ))
    fig_sens.add_vline(x=5, line_dash="dash", line_color=RED,
                       annotation_text="Current (5%)", annotation_position="top")
    fig_sens.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Terminate threshold (CPU p95 %)",
        yaxis_title="Monthly savings ($M)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", dtick=1),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        height=350,
        showlegend=False,
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    sc1, sc2, sc3 = st.columns(3)
    current_idx = thresholds_range.index(5)
    with sc1:
        conservative_savings = savings_by_threshold[thresholds_range.index(3)]
        st.metric("Conservative (3%)", f"${conservative_savings:.1f}M/mo",
                  f"{terminate_counts[thresholds_range.index(3)]:,} VMs")
    with sc2:
        st.metric("Current (5%)", f"${savings_by_threshold[current_idx]:.1f}M/mo",
                  f"{terminate_counts[current_idx]:,} VMs")
    with sc3:
        aggressive_savings = savings_by_threshold[thresholds_range.index(10)]
        st.metric("Aggressive (10%)", f"${aggressive_savings:.1f}M/mo",
                  f"{terminate_counts[thresholds_range.index(10)]:,} VMs")

    st.markdown('<p class="section-header">VM Explorer</p>',
                unsafe_allow_html=True)

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
        filtered,
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
    st.caption(f"Showing {len(filtered):,} of {len(rec):,} VMs")

    footer()
