"""Advanced feature engineering with style awareness and interaction terms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.config import ROLLING_WINDOW, TEAM_BOOST_OVERRIDES
from ..utils import col_or_nan, log
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


DEFAULT_POSSESSION_PCT = 50.0
DEFAULT_PPDA = 6.0


def _row_numeric(row: pd.Series, key: str, default: float = np.nan) -> float:
    """Safely read a numeric value from a row, falling back to `default`."""
    value = pd.to_numeric(pd.Series([row.get(key, default)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return default
    return float(value)


def _style_weighted_context_values(row: pd.Series, team_style: dict) -> dict[str, float]:
    """Compute style-weighted context metrics for one match row."""
    weights = calculate_style_weighted_context(row, team_style)

    possession = _row_numeric(row, "Possession")
    sca = _row_numeric(row, "Shot_Creating_Actions")
    dribbles = _row_numeric(row, "Successful_Dribbles")
    final_third_entries = _row_numeric(row, "Final_Third_Entries")

    return {
        "Possession_Style_Weighted": (
            possession * weights.get("Possession", 1.0)
            if pd.notna(possession)
            else np.nan
        ),
        "SCA_Style_Weighted": (
            sca * weights.get("Shot_Creating_Actions", 1.0) if pd.notna(sca) else np.nan
        ),
        "Dribbles_Style_Weighted": (
            dribbles * weights.get("Successful_Dribbles", 1.0)
            if pd.notna(dribbles)
            else np.nan
        ),
        "Final_Third_Style_Weighted": (
            final_third_entries * weights.get("Final_Third_Entries", 1.0)
            if pd.notna(final_third_entries)
            else np.nan
        ),
        "Possession_Style_Delta": (
            possession - float(team_style.get("possession_pct", DEFAULT_POSSESSION_PCT))
            if pd.notna(possession)
            else np.nan
        ),
    }


def _style_adjusted_rolling_values(
    row: pd.Series, team_style: dict, rolling_window: int
) -> dict[str, float]:
    """Adjust rolling xG/xGA based on expected values from team style."""
    rolling_xg = _row_numeric(row, f"rolling_xG_{rolling_window}")
    rolling_xga = _row_numeric(row, f"rolling_xGA_{rolling_window}")

    xg_expectation = style_xg_expectation(team_style)
    xga_expectation = style_defense_expectation(team_style)

    xg_adjusted = (
        rolling_xg / max(xg_expectation, 1e-6) if pd.notna(rolling_xg) else np.nan
    )
    xga_adjusted = rolling_xga * xga_expectation if pd.notna(rolling_xga) else np.nan

    return {
        "rolling_xG_style_adj": xg_adjusted,
        "rolling_xGA_style_adj": xga_adjusted,
        "xG_diff": (
            xg_adjusted - xga_adjusted
            if pd.notna(xg_adjusted) and pd.notna(xga_adjusted)
            else np.nan
        ),
    }


def _referee_style_impact_for_row(row: pd.Series) -> float:
    """Estimate how referee bias may interact with pressing/aggression style."""
    is_home = str(row.get("home_advantage", "a")).lower() == "h"
    foul_col = "Home_Fouls" if is_home else "Away_Fouls"

    team_fouls = _row_numeric(row, foul_col, default=0.0)
    ppda_value = _row_numeric(row, "PPDA", default=DEFAULT_PPDA)
    referee_bias = _row_numeric(row, "Referee_Bias_Score", default=0.0)

    aggression = team_fouls + max(0.0, 8.0 - ppda_value)
    return referee_bias * (1.0 + aggression / 10.0)


def _momentum_features_for_row(
    row: pd.Series,
    rolling_window: int,
    recent_form: dict,
    opponent_strength: float,
) -> dict[str, float]:
    """Create compact momentum/interaction features for one match row."""
    rolling_points = _row_numeric(row, f"rolling_win_rate_{rolling_window}")
    if pd.notna(rolling_points):
        rolling_points *= 3.0
    else:
        rolling_points = recent_form["points_per_game"] * 3.0

    return {
        "form_vs_strength": recent_form["points_per_game"] * opponent_strength,
        "momentum": rolling_points + recent_form["streak"],
    }


def build_style_aware_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build rolling features with style-aware interpretation."""
    df = df.sort_values(["Team", "Date"]).copy()

    def _lagged_rolling_mean(series: pd.Series, window: int) -> pd.Series:
        return series.shift(1).rolling(window, min_periods=1).mean()

    w = ROLLING_WINDOW
    grp = df.groupby("Team", group_keys=False)
    roll = lambda col: grp[col].transform(lambda s: _lagged_rolling_mean(s, w))

    # Base rolling features
    df[f"rolling_xG_{w}"] = roll("xG")
    df[f"rolling_xGA_{w}"] = roll("xGA")
    df[f"rolling_scored_{w}"] = roll("_scored")
    df[f"rolling_conceded_{w}"] = roll("_conceded")
    df[f"rolling_win_rate_{w}"] = roll("_win_val")
    df[f"rolling_ppda_{w}"] = roll("_ppda")

    if "_xpts" in df.columns:
        df[f"rolling_xpts_{w}"] = roll("_xpts")

    log(f"Style-aware rolling features built (window={w})")
    return df


def attach_style_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject style-aware engineered features directly into training matrix inputs.
    All historical computations use Date < current match Date to avoid leakage.
    """
    out = df.sort_values(["Date", "Team", "Opponent"]).copy()
    w = ROLLING_WINDOW
    feature_rows: list[dict[str, float]] = []

    for _, row in out.iterrows():
        current_date = row["Date"]
        history = out[out["Date"] < current_date]

        team = row["Team"]
        opponent = row["Opponent"]

        team_style = classify_team_style(history, team, window=10)
        opp_style = classify_team_style(history, opponent, window=10)

        recent_form = calculate_recent_form(history, team, window=5)
        opp_strength = calculate_opponent_strength(history, opponent)

        _ = opponent_style_interaction(team_style, opp_style)
        _ = estimate_squad_depth(history, team, window=15)

        row_features: dict[str, float] = {}
        row_features.update(_style_weighted_context_values(row, team_style))
        row_features.update(_style_adjusted_rolling_values(row, team_style, w))
        row_features.update(
            _momentum_features_for_row(
                row,
                rolling_window=w,
                recent_form=recent_form,
                opponent_strength=opp_strength,
            )
        )
        row_features["referee_style_impact"] = _referee_style_impact_for_row(row)
        feature_rows.append(row_features)

    engineered = pd.DataFrame(feature_rows, index=out.index)
    out = pd.concat([out, engineered], axis=1)

    log("Style engineered features attached")
    return out


def attach_referee_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add style-aware referee bias features."""
    if "Referee" not in df.columns:
        df["Referee_Bias_Score"] = 0.0
        log("No Referee column — Referee_Bias_Score = 0")
        return df

    out = df.sort_values(["Date", "Team", "Opponent"]).copy()
    out["_ref"] = (
        out["Referee"].astype(str).str.lower().str.strip().replace("", "unknown")
    )

    home_mask = out["home_advantage"].astype(str).str.lower().eq("h")

    home_y = pd.to_numeric(col_or_nan(out, "Home_Yellow"), errors="coerce")
    home_r = pd.to_numeric(col_or_nan(out, "Home_Red"), errors="coerce")
    away_y = pd.to_numeric(col_or_nan(out, "Away_Yellow"), errors="coerce")
    away_r = pd.to_numeric(col_or_nan(out, "Away_Red"), errors="coerce")

    # Weighted card pressure: red = 2 yellows
    home_cards = home_y.fillna(0.0) + 2.0 * home_r.fillna(0.0)
    away_cards = away_y.fillna(0.0) + 2.0 * away_r.fillna(0.0)

    team_cards = np.where(home_mask, home_cards, away_cards)
    opp_cards = np.where(home_mask, away_cards, home_cards)

    pk_for = pd.to_numeric(col_or_nan(out, "PK"), errors="coerce").fillna(0.0)
    pk_against = pd.to_numeric(col_or_nan(out, "PK_Allowed"), errors="coerce").fillna(
        0.0
    )

    # Raw signal: positive means favorable officiating
    raw_signal = (opp_cards - team_cards) + 2.0 * (pk_for - pk_against)
    raw_signal = pd.Series(raw_signal, index=out.index).fillna(0.0)

    # Normalize by referee
    std = raw_signal.std()
    out["_ref_signal_norm"] = (
        (raw_signal - raw_signal.mean()) / std if pd.notna(std) and std > 0 else 0.0
    )

    # Historical bias: per referee-team pair (shift-1 for no leakage)
    out["Referee_Bias_Score"] = (
        out.groupby(["_ref", "Team"])["_ref_signal_norm"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0.0)
    )

    # Adjust for league situation: relegation-zone teams or title-chasers get boosted sensitivity
    relegation_mask = out["team_position"].fillna(10) >= 16
    title_mask = out["team_position"].fillna(10) <= 2
    situation_multiplier = np.where(
        relegation_mask | title_mask, 1.3, 1.0
    )  # More sensitive when stakes are high
    out["Referee_Bias_Score"] = out["Referee_Bias_Score"] * situation_multiplier

    out.drop(columns=["_ref", "_ref_signal_norm"], inplace=True)
    log("Style-aware referee bias features attached")
    return out


def attach_dynamic_motivation(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate dynamic motivation score with style awareness."""
    out = df.sort_values(["Date", "Team"]).copy()
    motivation_scores = []

    def _position_component(team_pos: float) -> float:
        """Map league position to a simple urgency/motivation value."""
        if pd.isna(team_pos):
            return 0.0
        if team_pos == 1:
            return 0.4
        if team_pos <= 5:
            return 0.25
        if team_pos <= 8:
            return 0.15
        if team_pos >= 16:
            return 0.5
        return 0.0

    for _, row in out.iterrows():
        current_date = row["Date"]
        history = out[out["Date"] < current_date]

        team = row["Team"]
        opp = row["Opponent"]
        team_pos = row.get("team_position", np.nan)
        opp_pos = row.get("opponent_position", np.nan)

        # Classify styles from historical data
        team_style = classify_team_style(history, team, window=10)
        opp_style = classify_team_style(history, opp, window=10)

        # Recent form component (style-adjusted)
        form = calculate_recent_form(history, team, window=5)
        form_score = 0.3 * form["points_per_game"]
        form_score += 0.05 * max(form["streak"], -3)

        # League position context
        position_score = _position_component(team_pos)

        # Opponent strength (style matters)
        opp_strength = calculate_opponent_strength(history, opp)
        opp_score = 0.15 * opp_strength

        # Fixture importance
        fixture_score = 0.2 * calculate_fixture_importance(team_pos, opp_pos, row)

        # Style interaction: bonus/penalty
        style_interaction = opponent_style_interaction(team_style, opp_style)

        # Squad depth: reduces fatigue impact on motivation
        squad_depth = estimate_squad_depth(history, team, window=15)
        depth_factor = 0.9 + 0.2 * squad_depth  # [0.9, 1.1]

        # Team boosters
        boost = TEAM_BOOST_OVERRIDES.get(team.lower(), 0.0)

        # Combine components
        total_motivation = (
            form_score
            + position_score
            + opp_score
            + fixture_score
            + style_interaction
            + boost
        ) * depth_factor

        motivation_scores.append(min(1.0, max(0.0, total_motivation)))

    out["Motivation_Score"] = motivation_scores
    log("Dynamic style-aware motivation scores calculated")
    return out
