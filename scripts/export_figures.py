"""Generate three LinkedIn-ready figures from cloud cost prediction results."""

import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "clean"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.major.size": 0,
        "ytick.major.size": 0,
    }
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
classified = pd.read_csv(DATA / "vm_classified.csv")
recommendations = pd.read_csv(DATA / "vm_recommendations.csv")
fleet = pd.read_csv(DATA / "fleet_cost_estimate.csv")

# Merge "review" class into "oversized" for display
classified["display_class"] = classified["class"].replace({"review": "oversized"})

# ---------------------------------------------------------------------------
# Figure 1: The Waste Problem
# ---------------------------------------------------------------------------

WASTE_COLORS = {
    "zombie": "#D32F2F",
    "idle": "#E64A19",
    "oversized": "#F57C00",
    "right-sized": "#43A047",
    "hot": "#1565C0",
}
CATEGORY_ORDER = ["zombie", "idle", "oversized", "right-sized", "hot"]

counts = classified["display_class"].value_counts()
total = counts.sum()
pcts = {c: counts.get(c, 0) / total * 100 for c in CATEGORY_ORDER}
labels = {
    "zombie": "Zombie",
    "idle": "Idle",
    "oversized": "Oversized",
    "right-sized": "Right-sized",
    "hot": "Hot",
}

total_cost = fleet["monthly_cost"].sum()
total_savings = fleet["zombie_savings"].sum() + fleet["downsize_savings"].sum()
waste_pct = total_savings / total_cost * 100

fig1, (ax_bar, ax_num) = plt.subplots(
    1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [3, 2]}
)

# Stacked horizontal bar
left = 0.0
for cat in CATEGORY_ORDER:
    width = pcts[cat]
    ax_bar.barh(
        0,
        width,
        left=left,
        color=WASTE_COLORS[cat],
        height=0.5,
        label=labels[cat],
    )
    if width > 8:
        ax_bar.text(
            left + width / 2,
            0,
            f"{labels[cat]}\n{width:.1f}%",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
        )
    elif width > 3:
        ax_bar.text(
            left + width / 2,
            0,
            f"{width:.1f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )
    left += width

# Legend below bar for small-segment categories
ax_bar.legend(
    loc="upper left",
    bbox_to_anchor=(0, -0.15),
    ncol=5,
    frameon=False,
    fontsize=11,
    handlelength=1.2,
    columnspacing=1.5,
)

ax_bar.set_xlim(0, 100)
ax_bar.set_yticks([])
ax_bar.set_xticks([])
ax_bar.set_title(
    "Fleet composition by VM classification",
    fontsize=16,
    fontweight="bold",
    pad=20,
    loc="left",
)
ax_bar.text(
    0,
    -0.45,
    f"{total:,.0f} virtual machines analyzed",
    fontsize=12,
    color="#666666",
    ha="left",
    transform=ax_bar.transData,
)

# Big number on the right
ax_num.set_xlim(0, 1)
ax_num.set_ylim(0, 1)
ax_num.set_xticks([])
ax_num.set_yticks([])

ax_num.text(
    0.5,
    0.6,
    f"${total_savings / 1e6:.1f}M/mo",
    fontsize=44,
    fontweight="bold",
    color="#D32F2F",
    ha="center",
    va="center",
)
ax_num.text(
    0.5,
    0.42,
    f"{waste_pct:.0f}% of a ${total_cost / 1e6:.0f}M fleet",
    fontsize=18,
    color="#666666",
    ha="center",
    va="center",
)
ax_num.text(
    0.5,
    0.28,
    "recoverable monthly spend",
    fontsize=14,
    color="#999999",
    ha="center",
    va="center",
)

fig1.suptitle(
    "The Waste Problem",
    fontsize=22,
    fontweight="bold",
    y=0.96,
    x=0.05,
    ha="left",
)
fig1.tight_layout(rect=[0, 0, 1, 0.92])
fig1.savefig(OUT / "linkedin_waste.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"Saved {OUT / 'linkedin_waste.png'}")


# ---------------------------------------------------------------------------
# Figure 2: Model Performance
# ---------------------------------------------------------------------------

ACTION_COLORS = {
    "terminate": "#D32F2F",
    "downsize": "#E64A19",
    "review": "#FBC02D",
    "keep": "#43A047",
}

fig2, ax2 = plt.subplots(figsize=(12, 7))

rec = recommendations.copy()
mae = np.mean(np.abs(rec["actual_cpu"] - rec["pred_mid"]))
within = np.mean(
    (rec["actual_cpu"] >= rec["pred_low"]) & (rec["actual_cpu"] <= rec["pred_high"])
)

# Subsample for scatter (full dataset is ~47k points)
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(rec), size=min(4000, len(rec)), replace=False)
sample = rec.iloc[sample_idx]

# Plot order: keep first (background), then overlay waste categories
for action in ["keep", "review", "downsize", "terminate"]:
    mask = sample["action"] == action
    if mask.sum() == 0:
        continue
    ax2.scatter(
        sample.loc[mask, "actual_cpu"],
        sample.loc[mask, "pred_mid"],
        c=ACTION_COLORS[action],
        s=12,
        alpha=0.5,
        label=action.capitalize(),
        rasterized=True,
    )

# Diagonal reference
lim = max(rec["actual_cpu"].max(), rec["pred_mid"].max()) * 1.05
ax2.plot([0, lim], [0, lim], color="#CCCCCC", linewidth=1.5, zorder=0)
ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.set_aspect("equal")

ax2.set_xlabel("Actual CPU utilization (%)", fontsize=14)
ax2.set_ylabel("Predicted CPU utilization (%)", fontsize=14)
ax2.set_title(
    "Model Performance",
    fontsize=22,
    fontweight="bold",
    pad=16,
    loc="left",
)

# Annotations
box_props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#DDDDDD")
ax2.text(
    0.97,
    0.12,
    f"MAE: {mae:.2f}%\nCoverage: {within * 100:.1f}%",
    transform=ax2.transAxes,
    fontsize=16,
    fontweight="bold",
    ha="right",
    va="bottom",
    bbox=box_props,
)

ax2.legend(
    loc="upper left",
    frameon=True,
    framealpha=0.9,
    edgecolor="#DDDDDD",
    fontsize=12,
    markerscale=2,
    title="Action",
    title_fontsize=13,
)

# Light grid
ax2.grid(True, alpha=0.15, linewidth=0.5)

fig2.tight_layout()
fig2.savefig(OUT / "linkedin_model.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved {OUT / 'linkedin_model.png'}")


# ---------------------------------------------------------------------------
# Figure 3: Savings Breakdown
# ---------------------------------------------------------------------------

TEAL = "#00D4AA"

# Aggregate savings by action using fleet-level data
# terminate -> zombie_savings, downsize -> downsize_savings
action_counts = recommendations["action"].value_counts()
savings_data = pd.DataFrame(
    {
        "action": ["Downsize", "Terminate", "Review", "Keep"],
        "savings": [
            fleet["downsize_savings"].sum(),
            fleet["zombie_savings"].sum(),
            0,
            0,
        ],
        "vm_count": [
            action_counts.get("downsize", 0),
            action_counts.get("terminate", 0),
            action_counts.get("review", 0),
            action_counts.get("keep", 0),
        ],
    }
)
savings_data = savings_data[savings_data["savings"] > 0].sort_values(
    "savings", ascending=True
)

fig3, ax3 = plt.subplots(figsize=(12, 5))

bars = ax3.barh(
    savings_data["action"],
    savings_data["savings"],
    color=TEAL,
    height=0.55,
    edgecolor="white",
)

# Labels on bars
for bar, (_, row) in zip(bars, savings_data.iterrows()):
    width = bar.get_width()
    ax3.text(
        width + total_savings * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"${width / 1e6:.1f}M   |   {int(row['vm_count']):,} VMs",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#333333",
    )

ax3.set_xlim(0, savings_data["savings"].max() * 1.35)
ax3.set_xticks([])
ax3.tick_params(axis="y", labelsize=16)

ax3.set_title(
    "Savings Breakdown",
    fontsize=22,
    fontweight="bold",
    pad=30,
    loc="left",
)
ax3.text(
    0.0,
    1.06,
    "XGBoost quantile regression with 95% confidence",
    transform=ax3.transAxes,
    fontsize=13,
    color="#888888",
    va="bottom",
)

fig3.tight_layout()
fig3.savefig(
    OUT / "linkedin_savings.png", dpi=200, bbox_inches="tight", facecolor="white"
)
plt.close(fig3)
print(f"Saved {OUT / 'linkedin_savings.png'}")

print("\nAll figures exported.")
