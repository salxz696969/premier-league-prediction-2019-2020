"""Team style classification and analysis functions."""

from __future__ import annotations

import pandas as pd
import numpy as np

from ..utils import col_or_nan


def classify_team_style(df: pd.DataFrame, team: str, window: int = 10) -> dict:
    """
    Classify team playing style based on historical stats.
    Returns style dict with properties: possession_heavy, defensive, counter_prone, etc.
    """
    team_df = df[df["Team"].eq(team)].sort_values("Date").tail(window)
    if team_df.empty:
        return {
            "possession_heavy": False,
            "defensive": False,
            "counter_prone": False,
            "possession_pct": 50.0,
            "ppda": 6.0,
            "defensive_actions": 10.0,
            "clean_sheet_rate": 0.3,
        }

    avg_possession = pd.to_numeric(
        col_or_nan(team_df, "Possession"), errors="coerce"
    ).mean()
    avg_ppda = pd.to_numeric(col_or_nan(team_df, "PPDA"), errors="coerce").mean()
    avg_defensive_actions = pd.to_numeric(
        col_or_nan(team_df, "Defensive_Actions"), errors="coerce"
    ).mean()
    clean_sheet_col = col_or_nan(team_df, "Clean_Sheet")
    avg_clean_sheets = (
        (clean_sheet_col == 1).sum() / len(team_df) if len(team_df) > 0 else 0.0
    )

    # Defaults if missing
    avg_possession = avg_possession if pd.notna(avg_possession) else 50.0
    avg_ppda = avg_ppda if pd.notna(avg_ppda) else 6.0
    avg_defensive_actions = (
        avg_defensive_actions if pd.notna(avg_defensive_actions) else 10.0
    )

    style = {
        "possession_heavy": avg_possession > 52,
        "defensive": avg_defensive_actions > 15,
        "counter_prone": avg_ppda < 5.5,
        "possession_pct": int(avg_possession),
        "ppda": avg_ppda,
        "defensive_actions": avg_defensive_actions,
        "clean_sheet_rate": float(avg_clean_sheets),
    }
    return style


def calculate_recent_form(df: pd.DataFrame, team: str, window: int = 5) -> dict:
    """Calculate recent form: points, xG, goals, streaks from last N matches."""
    team_df = df[df["Team"].eq(team)].sort_values("Date").copy()
    if len(team_df) < 1:
        return {"points_per_game": 0.0, "streak": 0, "recent_xg": 0.0}

    team_df["_win_val"] = team_df["Result"].map({"w": 1.0, "d": 0.5, "l": 0.0})
    recent = team_df.tail(window)

    points = recent["_win_val"].sum()
    points_per_game = points / len(recent) if len(recent) > 0 else 0.0

    recent_xg = pd.to_numeric(col_or_nan(recent, "xG"), errors="coerce").mean()
    recent_goals = pd.to_numeric(col_or_nan(recent, "Scored"), errors="coerce").mean()

    # Win streak (positive for W, negative for L)
    streak = 0
    for result in recent["Result"].iloc[::-1]:
        if result == "w":
            streak += 1
        elif result == "l":
            streak -= 1
        else:
            break

    return {
        "points_per_game": points_per_game,
        "streak": streak,
        "recent_xg": recent_xg if pd.notna(recent_xg) else 0.0,
        "recent_goals": recent_goals if pd.notna(recent_goals) else 0.0,
    }


def calculate_opponent_strength(df: pd.DataFrame, opponent: str) -> float:
    """Estimate opponent strength from recent win rate. Returns score [0, 1]."""
    opp_df = df[df["Team"].eq(opponent)].sort_values("Date").tail(10)
    if opp_df.empty:
        return 0.5

    opp_df["_win_val"] = opp_df["Result"].map({"w": 1.0, "d": 0.5, "l": 0.0})
    win_rate = opp_df["_win_val"].mean()
    return float(win_rate)


def calculate_fixture_importance(
    team_pos: float, opponent_pos: float, row: pd.Series
) -> float:
    """
    Assess fixture importance based on league position gap and situation.
    Returns score [0, 1].
    """
    if pd.isna(team_pos) or pd.isna(opponent_pos):
        return 0.5

    importance = 0.0

    # Title race (positions 1-2)
    if team_pos <= 2:
        importance += 0.7

    # Relegation fight (positions 16+)
    if team_pos >= 16:
        importance += 0.8

    # European race (positions 3-7)
    if 3 <= team_pos <= 7:
        importance += 0.5

    # Derby or big match (opponent in top 6 or within 2 positions)
    pos_gap = abs(team_pos - opponent_pos)
    if opponent_pos <= 6:
        importance += 0.4
    if pos_gap <= 2:
        importance += 0.3

    return min(1.0, importance)


def estimate_squad_depth(df: pd.DataFrame, team: str, window: int = 15) -> float:
    """
    Estimate squad depth from lineup rotation rate.
    Higher rotation → better squad depth. Returns [0, 1].
    """
    team_df = df[df["Team"].eq(team)].sort_values("Date").tail(window)
    if len(team_df) < 3:
        return 0.5  # Default

    # Proxy: if lineups/formation changes frequently, squad depth is better
    formations = col_or_nan(team_df, "Formation").dropna().nunique()
    rotation_rate = min(1.0, formations / max(1, len(team_df)))
    return rotation_rate
