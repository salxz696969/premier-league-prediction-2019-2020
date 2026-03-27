"""Style-based feature calculations and transformations."""

from __future__ import annotations

import pandas as pd


def style_xg_expectation(team_style: dict) -> float:
    """
    Adjust xG expectations based on team style.
    Counter teams may have lower xG but be equally effective.
    Possession teams should create more xG.
    Returns multiplier [0.7, 1.3].
    """
    if team_style.get("counter_prone"):
        return 0.75  # Counter teams okay with lower xG
    if team_style.get("possession_heavy"):
        return 1.1  # Possession teams expected higher xG
    return 1.0


def style_defense_expectation(team_style: dict) -> float:
    """
    Adjust defensive xGA/clean sheet expectations based on style.
    Defensive teams should concede less.
    Counter teams may concede more but defend compact.
    Returns multiplier [0.7, 1.3].
    """
    if team_style.get("defensive"):
        return 0.75  # Defensive teams expected lower xGA
    if team_style.get("counter_prone"):
        return 1.05  # Counters may concede more on balance
    return 1.0


def opponent_style_interaction(team_style: dict, opp_style: dict) -> float:
    """
    Calculate matchup bonus/penalty based on style interaction.
    Returns adjustment [-0.15, +0.15].
    """
    adjustment = 0.0

    # Counter team vs possession team: counters thrive
    if team_style.get("counter_prone") and opp_style.get("possession_heavy"):
        adjustment += 0.1

    # Possession team vs defensive team: harder to break down
    if team_style.get("possession_heavy") and opp_style.get("defensive"):
        adjustment -= 0.08

    # Defensive vs possession: defensive absorbs play better
    if team_style.get("defensive") and opp_style.get("possession_heavy"):
        adjustment += 0.06

    # Defensive vs counter: could be vulnerable on transitions
    if team_style.get("defensive") and opp_style.get("counter_prone"):
        adjustment -= 0.07

    return adjustment


def calculate_style_weighted_context(row: pd.Series, team_style: dict) -> dict:
    """Return weights for context features based on team style."""
    weights = {
        "Possession": 0.8 if not team_style.get("possession_heavy") else 1.3,
        "Shot_Creating_Actions": 1.0,
        "Successful_Dribbles": 1.2 if team_style.get("counter_prone") else 0.9,
        "Final_Third_Entries": 1.1 if team_style.get("possession_heavy") else 0.8,
        "Final_Third_Entries_Allowed": 1.2 if team_style.get("defensive") else 0.9,
        "Aerial_Battles_Won_Pct": 1.2 if team_style.get("defensive") else 0.9,
        "Save_Pct": 1.3 if team_style.get("defensive") else 0.85,
        "PPDA": 1.1 if team_style.get("defensive") else 0.9,
    }
    return weights
