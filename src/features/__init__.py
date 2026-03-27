"""Feature engineering modules."""

from .advanced import (
    attach_dynamic_motivation,
    attach_referee_features,
    attach_style_engineered_features,
    build_style_aware_rolling_features,
)
from .basic import (
    attach_context_features,
    attach_fatigue,
    attach_h2h_features,
    attach_position_features,
    build_rolling_features,
)
from .style_analysis import (
    calculate_fixture_importance,
    calculate_opponent_strength,
    calculate_recent_form,
    classify_team_style,
    estimate_squad_depth,
)
from .style_calculations import (
    calculate_style_weighted_context,
    opponent_style_interaction,
    style_defense_expectation,
    style_xg_expectation,
)

__all__ = [
    # Basic features
    "attach_fatigue",
    "build_rolling_features",
    "attach_position_features",
    "attach_h2h_features",
    "attach_context_features",
    # Style analysis
    "classify_team_style",
    "calculate_recent_form",
    "calculate_opponent_strength",
    "calculate_fixture_importance",
    "estimate_squad_depth",
    # Style calculations
    "style_xg_expectation",
    "style_defense_expectation",
    "opponent_style_interaction",
    "calculate_style_weighted_context",
    # Advanced features
    "build_style_aware_rolling_features",
    "attach_style_engineered_features",
    "attach_referee_features",
    "attach_dynamic_motivation",
]
