"""Page 5: Reserved Instances — on-demand vs RI commitments for "keep" VMs."""
import json

import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import (
    AMBER,
    GREEN,
    PLOTLY_LAYOUT,
    RED,
    ROOT,
    TEXT,
    footer,
    kpi_card,
    load_ri_recommendations,
    safe_page,
)


@safe_page
def page_reserved_instances():
    st.markdown("# Reserved Instances")
    st.markdown("""
    <p class="story-text">
        <strong>Keep</strong> VMs are the ones this project is NOT proposing
        to touch: not terminated, not downsized, not flagged for review.
        That makes them the only workloads stable enough to reserve capacity
        for — everything else could disappear before a 1-3 year commitment
        is up. This page compares what they cost on-demand against real
        AWS Reserved Instance rates.
    </p>
    """, unsafe_allow_html=True)

    ri = load_ri_recommendations()

    on_demand_monthly = ri["on_demand_monthly_cost"].iloc[0]
    ri_monthly = ri["ri_monthly_cost"].iloc[0]
    savings_pct = ri["savings_pct"].iloc[0]
    breakeven_pct = ri["breakeven_utilization_pct"].iloc[0]
    commitment = ri["recommended_commitment"].iloc[0]
    total_monthly_savings = ri["monthly_savings"].sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card(f"{len(ri):,}", "Keep VMs Priced",
                 "RI candidates: stable, not slated for any action", TEXT)
    with c2:
        kpi_card(f"${total_monthly_savings:,.0f}/mo", "Fleet Savings",
                 f"If all keep VMs moved to {commitment}", GREEN)
    with c3:
        kpi_card(f"{breakeven_pct:.0f}%", "Breakeven Utilization",
                 "Minimum uptime for the reservation to pay off", AMBER)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<p class="section-header">On-Demand vs Reserved</p>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <p class="story-text">
        Fleet-weighted on-demand baseline: <strong>${on_demand_monthly:,.2f}/VM/month</strong>.
        Cheapest commitment ({commitment}): <strong>${ri_monthly:,.2f}/VM/month</strong>
        — a <strong>{savings_pct:.0f}%</strong> reduction.
    </p>
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=["On-Demand", commitment],
        y=[on_demand_monthly, ri_monthly],
        marker_color=[RED, GREEN],
        text=[f"${on_demand_monthly:,.2f}", f"${ri_monthly:,.2f}"],
        textposition="outside",
        textfont=dict(color=TEXT),
        hovertemplate="%{x}: $%{y:,.2f}/VM/mo<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        yaxis_title="$/VM/month",
        xaxis=dict(gridcolor="#0A1628"),
        yaxis=dict(gridcolor="#0A1628"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("All term / payment option combinations", expanded=False):
        ri_pricing_file = ROOT / "data" / "pricing" / "ec2_reserved_instances.json"
        if ri_pricing_file.exists():
            with open(ri_pricing_file) as f:
                meta = json.load(f)["metadata"]
            st.caption(
                f"Source: AWS Bulk Pricing API, region {meta.get('region', 'us-east-1')}, "
                f"verified {meta.get('verified', '?')}. "
                "Refresh with: python scripts/fetch_ri_pricing.py"
            )
        st.dataframe(
            ri[["recommended_commitment", "on_demand_monthly_cost", "ri_monthly_cost",
                "monthly_savings", "savings_pct", "breakeven_utilization_pct"]].drop_duplicates(),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Why aren't Savings Plans shown?", expanded=False):
        st.markdown(f"""
{ri["savings_plan"].iloc[0]}
        """)

    st.markdown('<p class="section-header">Keep VMs</p>', unsafe_allow_html=True)
    st.markdown("""
    <p class="story-text">
        The SAP dataset has no per-VM EC2 instance type, only aggregate
        fleet-wide size counts — so every row below shares the same
        fleet-weighted cost estimate; only the instance ID and actual CPU
        are VM-specific.
    </p>
    """, unsafe_allow_html=True)
    st.dataframe(
        ri,
        column_config={
            "actual_cpu": st.column_config.NumberColumn("Actual CPU", format="%.1f%%"),
            "on_demand_monthly_cost": st.column_config.NumberColumn("On-Demand $/mo", format="$%.2f"),
            "ri_monthly_cost": st.column_config.NumberColumn("RI $/mo", format="$%.2f"),
            "monthly_savings": st.column_config.NumberColumn("Savings $/mo", format="$%.2f"),
            "savings_pct": st.column_config.NumberColumn("Savings %", format="%.1f%%"),
            "breakeven_utilization_pct": st.column_config.NumberColumn("Breakeven %", format="%.1f%%"),
        },
        use_container_width=True,
        height=400,
    )
    st.caption(f"Showing all {len(ri):,} 'keep' VMs")

    footer()
