"""Dashboard package — shared state and page functions."""

from dashboard.page_model import page_model
from dashboard.page_problem import page_problem
from dashboard.page_recommendations import page_recommendations
from dashboard.page_try_it import page_try_it
from dashboard.shared import (
    AMBER,
    CARD_BG,
    DARK_BG,
    GRAY,
    GREEN,
    PLOTLY_LAYOUT,
    RED,
    TEXT,
    TEXT_DIM,
    footer,
    inject_css,
    kpi_card,
    load_classified,
    load_fleet_costs,
    load_models,
    load_recommendations,
    load_utilization,
)

__all__ = [
    "AMBER", "CARD_BG", "DARK_BG", "GRAY", "GREEN", "PLOTLY_LAYOUT",
    "RED", "TEXT", "TEXT_DIM",
    "footer", "inject_css", "kpi_card",
    "load_classified", "load_fleet_costs", "load_models",
    "load_recommendations", "load_utilization",
    "page_model", "page_problem", "page_recommendations", "page_try_it",
]
